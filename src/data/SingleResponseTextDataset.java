package data;

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
public class SingleResponseTextDataset extends TextDataset {

    protected double[] responses;

    public SingleResponseTextDataset(String name, String folder) {
        super(name, folder);
    }

    public SingleResponseTextDataset(String name, String folder,
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
        logln("--- Loading response from file " + responseFilepath);

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
        logln("--- Outputing document info ... " + outputFile);

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
        logln("--- Reading document info from " + filepath);

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

    public static void main(String[] args) {
        try {
//            run(args);
            
            test(args);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void test(String[] args) throws Exception {
        parser = new BasicParser();

        // create the Options
        options = new Options();

        addOption("dataset", "Dataset");
        addOption("data-folder", "Folder that stores the processed data");
        addOption("format-folder", "Folder that stores formatted data");
        addOption("format-file", "Formatted file name");

        cmd = parser.parse(options, args);
        if (cmd.hasOption("help")) {
            CLIUtils.printHelp("java -cp dist/segan.jar sampler.supervised.regression.SLDA -help", options);
            return;
        }
        
        verbose = true;
        debug = true;

        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        SingleResponseTextDataset data = new SingleResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
    }

    public static void run(String[] args) throws Exception {
        parser = new BasicParser();

        // create the Options
        options = new Options();

        addOption("dataset", "Dataset");
        addOption("data-folder", "Folder that stores the processed data");
        addOption("text-data", "Directory of the text data");
        addOption("response-file", "Directory of the response file");
        addOption("format-folder", "Folder that stores formatted data");
        addOption("format-file", "Formatted file name");

        addOption("u", "The minimum count of raw unigrams");
        addOption("b", "The minimum count of raw bigrams");
        addOption("bs", "The minimum score of bigrams");
        addOption("V", "Maximum vocab size");
        addOption("min-tf", "Term frequency minimum cutoff");
        addOption("max-tf", "Term frequency maximum cutoff");
        addOption("min-df", "Document frequency minimum cutoff");
        addOption("max-df", "Document frequency maximum cutoff");
        addOption("min-doc-length", "Document minimum length");

        addOption("L", "Maximum label vocab size");
        addOption("min-label-df", "Minimum count of raw labels");

        options.addOption("s", false, "Whether stopwords are filtered");
        options.addOption("l", false, "Whether lemmatization is performed");
        options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
        options.addOption("help", false, "Help");

        cmd = parser.parse(options, args);
        if (cmd.hasOption("help")) {
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData -help", options);
            return;
        }

        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

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

        String responseFile = cmd.getOptionValue("response-file");
        SingleResponseTextDataset dataset = new SingleResponseTextDataset(datasetName, datasetFolder, corpProc);
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
