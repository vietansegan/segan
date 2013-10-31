package data;

import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class RegressionTextDataset extends TextDataset {

    protected double[] responses;

    public RegressionTextDataset(String name, String folder) {
        super(name, folder);
    }

    public RegressionTextDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public double[] getResponses() {
        return this.responses;
    }

    public void setResponses(double[] responses) {
        this.responses = responses;
    }

    public double[] getResponses(ArrayList<Integer> instances) {
        double[] res = new double[instances.size()];
        for (int i = 0; i < res.length; i++) {
            int idx = instances.get(i);
            res[i] = responses[idx];
        }
        return res;
    }

    public void loadResponses(String responseFilepath) throws Exception {
        if (verbose) {
            logln("--- Loading response from file " + responseFilepath);
        }

        if (this.docIdList == null) {
            throw new RuntimeException("docIdList is null. Load text data first.");
        }

        this.responses = new double[this.docIdList.size()];
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(responseFilepath);
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            String docId = sline[0];
            double docResponse = Double.parseDouble(sline[1]);
            int index = this.docIdList.indexOf(docId);
            this.responses[index] = docResponse;
        }
        reader.close();
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        if (verbose) {
            logln("--- Outputing document info ... " + outputFile);
        }

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex)
                    + "\t" + this.responses[docIndex]
                    + "\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File filepath) throws Exception {
        if (verbose) {
            logln("--- Reading document info from " + filepath);
        }

        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<Double> responseList = new ArrayList<Double>();

        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            responseList.add(Double.parseDouble(sline[1]));
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.responses = new double[responseList.size()];
        for (int i = 0; i < this.responses.length; i++) {
            this.responses[i] = responseList.get(i);
        }
    }

    /**
     * Create cross validation
     *
     * @param cvFolder Cross validation folder
     * @param numFolds Number of folds
     * @param trToDevRatio Ratio between the number of training and the number
     * of test data
     */
    @Override
    public void createCrossValidation(String cvFolder, int numFolds, double trToDevRatio)
            throws Exception {
        ArrayList<Instance<String>> instanceList = new ArrayList<Instance<String>>();
        ArrayList<Integer> groupIdList = new ArrayList<Integer>();
        for (int d = 0; d < this.docIdList.size(); d++) {
            instanceList.add(new Instance<String>(docIdList.get(d)));
            groupIdList.add(0); // random, no stratified
        }

        String cvName = "";
        CrossValidation<String, Instance<String>> cv =
                new CrossValidation<String, Instance<String>>(
                cvFolder,
                cvName,
                instanceList);

        cv.stratify(groupIdList, numFolds, trToDevRatio);
        cv.outputFolds();

        for (Fold<String, Instance<String>> fold : cv.getFolds()) {
            // processor
            CorpusProcessor cp = new CorpusProcessor(corpProc);

            // training data
            RegressionTextDataset trainData = new RegressionTextDataset(fold.getFoldName(), cv.getFolderPath(), cp);
            trainData.setFormatFilename(fold.getFoldName() + Fold.TrainingExt);
            ArrayList<String> trDocIds = new ArrayList<String>();
            ArrayList<String> trDocTexts = new ArrayList<String>();
            double[] trResponses = new double[fold.getNumTrainingInstances()];
            for (int ii = 0; ii < fold.getNumTrainingInstances(); ii++) {
                int idx = fold.getTrainingInstances().get(ii);
                trDocIds.add(this.docIdList.get(idx));
                trDocTexts.add(this.textList.get(idx));
                trResponses[ii] = responses[idx];
            }
            trainData.setTextData(trDocIds, trDocTexts);
            trainData.setResponses(trResponses);
            trainData.format(fold.getFoldFolderPath());

            // development data
            RegressionTextDataset devData = new RegressionTextDataset(fold.getFoldName(), cv.getFolderPath(), cp);
            devData.setFormatFilename(fold.getFoldName() + Fold.DevelopExt);
            ArrayList<String> deDocIds = new ArrayList<String>();
            ArrayList<String> deDocTexts = new ArrayList<String>();
            double[] deResponses = new double[fold.getNumDevelopmentInstances()];
            for (int ii = 0; ii < fold.getNumDevelopmentInstances(); ii++) {
                int idx = fold.getDevelopmentInstances().get(ii);
                deDocIds.add(this.docIdList.get(idx));
                deDocTexts.add(this.textList.get(idx));
                deResponses[ii] = responses[idx];
            }
            devData.setTextData(deDocIds, deDocTexts);
            devData.setResponses(deResponses);
            devData.format(fold.getFoldFolderPath());

            // test data
            RegressionTextDataset testData = new RegressionTextDataset(fold.getFoldName(), cv.getFolderPath(), cp);
            testData.setFormatFilename(fold.getFoldName() + Fold.TestExt);
            ArrayList<String> teDocIds = new ArrayList<String>();
            ArrayList<String> teDocTexts = new ArrayList<String>();
            double[] teResponses = new double[fold.getNumTestingInstances()];
            for (int ii = 0; ii < fold.getNumTestingInstances(); ii++) {
                int idx = fold.getTestingInstances().get(ii);
                teDocIds.add(this.docIdList.get(idx));
                teDocTexts.add(this.textList.get(idx));
                teResponses[ii] = responses[idx];
            }
            testData.setTextData(teDocIds, teDocTexts);
            testData.setResponses(teResponses);
            testData.format(fold.getFoldFolderPath());
        }
    }

    public static void main(String[] args) {
        try {
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("dataset", "Dataset");
            addOption("data-folder", "Folder that stores the processed data");
            addOption("text-data", "Directory of the text data");
            addOption("format-folder", "Folder that stores formatted data");
            addOption("format-file", "Formatted file name");
            addOption("response-file", "Directory of the response file");

            // text processing
            addOption("u", "The minimum count of raw unigrams");
            addOption("b", "The minimum count of raw bigrams");
            addOption("bs", "The minimum score of bigrams");
            addOption("V", "Maximum vocab size");
            addOption("min-tf", "Term frequency minimum cutoff");
            addOption("max-tf", "Term frequency maximum cutoff");
            addOption("min-df", "Document frequency minimum cutoff");
            addOption("max-df", "Document frequency maximum cutoff");
            addOption("min-doc-length", "Document minimum length");

            // cross validation
            addOption("num-folds", "Number of folds. Default 5.");
            addOption("tr2dev-ratio", "Training-to-development ratio. Default 0.8.");
            addOption("cv-folder", "Folder to store cross validation folds");

            addOption("run-mode", "Run mode");

            options.addOption("v", false, "Verbose");
            options.addOption("d", false, "Debug");
            options.addOption("s", false, "Whether stopwords are filtered");
            options.addOption("l", false, "Whether lemmatization is performed");
            options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData -help", options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            String runMode = cmd.getOptionValue("run-mode");
            if (runMode.equals("process")) {
                process(args);
            } else if (runMode.equals("load")) {
                load(args);
            } else if (runMode.equals("cross-validation")) {
                crossValidate(args);
            } else {
                throw new RuntimeException("Run mode " + runMode + " is not supported");
            }

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void crossValidate(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String responseFile = cmd.getOptionValue("response-file");

        int unigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "u", 0);
        int bigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "b", 0);
        double bigramScoreCutoff = CLIUtils.getDoubleArgument(cmd, "bs", 5.0);
        int maxVocabSize = CLIUtils.getIntegerArgument(cmd, "V", Integer.MAX_VALUE);
        int vocTermFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-tf", 0);
        int vocTermFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-tf", Integer.MAX_VALUE);
        int vocDocFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-df", 0);
        int vocDocFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-df", Integer.MAX_VALUE);
        int docTypeCountCutoff = CLIUtils.getIntegerArgument(cmd, "min-doc-length", 1);

        boolean stopwordFilter = cmd.hasOption("s");
        boolean lemmatization = cmd.hasOption("l");

        int numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
        double trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 0.8);
        String cvFolder = cmd.getOptionValue("cv-folder");
        IOUtils.createFolder(cvFolder);

        CorpusProcessor corpProc = new CorpusProcessor(
                unigramCountCutoff,
                bigramCountCutoff,
                bigramScoreCutoff,
                maxVocabSize,
                vocTermFreqMinCutoff,
                vocTermFreqMaxCutoff,
                vocDocFreqMinCutoff,
                vocDocFreqMaxCutoff,
                docTypeCountCutoff,
                stopwordFilter,
                lemmatization);

        RegressionTextDataset dataset = new RegressionTextDataset(datasetName, datasetFolder, corpProc);
        // load text data
        if (cmd.hasOption("file")) {
            dataset.loadTextDataFromFile(textInputData);
        } else {
            dataset.loadTextDataFromFolder(textInputData);
        }
        dataset.loadResponses(responseFile); // load response data
        dataset.createCrossValidation(cvFolder, numFolds, trToDevRatio);
    }

    public static RegressionTextDataset load(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        RegressionTextDataset data = new RegressionTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        return data;
    }

    public static void process(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        String responseFile = cmd.getOptionValue("response-file");

        int unigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "u", 0);
        int bigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "b", 0);
        double bigramScoreCutoff = CLIUtils.getDoubleArgument(cmd, "bs", 5.0);
        int maxVocabSize = CLIUtils.getIntegerArgument(cmd, "V", Integer.MAX_VALUE);
        int vocTermFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-tf", 0);
        int vocTermFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-tf", Integer.MAX_VALUE);
        int vocDocFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-df", 0);
        int vocDocFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-df", Integer.MAX_VALUE);
        int docTypeCountCutoff = CLIUtils.getIntegerArgument(cmd, "min-doc-length", 1);

        boolean stopwordFilter = cmd.hasOption("s");
        boolean lemmatization = cmd.hasOption("l");

        CorpusProcessor corpProc = new CorpusProcessor(
                unigramCountCutoff,
                bigramCountCutoff,
                bigramScoreCutoff,
                maxVocabSize,
                vocTermFreqMinCutoff,
                vocTermFreqMaxCutoff,
                vocDocFreqMinCutoff,
                vocDocFreqMaxCutoff,
                docTypeCountCutoff,
                stopwordFilter,
                lemmatization);

        RegressionTextDataset dataset = new RegressionTextDataset(datasetName, datasetFolder, corpProc);
        dataset.setFormatFilename(formatFile);

        // load text data
        if (cmd.hasOption("file")) {
            dataset.loadTextDataFromFile(textInputData);
        } else {
            dataset.loadTextDataFromFolder(textInputData);
        }
        dataset.loadResponses(responseFile); // load response data
        dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder));
    }
}
