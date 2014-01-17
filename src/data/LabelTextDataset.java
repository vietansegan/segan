package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.DataUtils;
import util.IOUtils;
import util.RankingItem;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 *
 * @author vietan
 */
public class LabelTextDataset extends TextDataset {

    public static final String labelVocabExt = ".lvoc";
    protected ArrayList<ArrayList<String>> labelList;
    protected ArrayList<String> labelVocab;
    protected int[][] labels;
    protected int maxLabelVocSize = Integer.MAX_VALUE;
    protected int minLabelDocFreq = 1;

    public LabelTextDataset(String name, String folder) {
        super(name, folder);
    }

    public LabelTextDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public void setMaxLabelVocabSize(int L) {
        this.maxLabelVocSize = L;
    }

    public void setMinLabelDocFreq(int f) {
        this.minLabelDocFreq = f;
    }

    public int[][] getLabels() {
        return this.labels;
    }

    public ArrayList<String> getLabelVocab() {
        return this.labelVocab;
    }

    public void setLabelVocab(ArrayList<String> lVoc) {
        this.labelVocab = lVoc;
    }

    public ArrayList<ArrayList<String>> getLabelList() {
        return this.labelList;
    }

    public void setLabelList(ArrayList<ArrayList<String>> lblList) {
        this.labelList = lblList;
    }

    /**
     * Filter document labels. The remaining labels only come from a given set
     * of labels
     *
     * @param labVoc The given set of labels
     */
    public void filterLabels(ArrayList<String> labVoc) {
        int D = words.length;
        this.labelVocab = labVoc;

        int[][] filterLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            ArrayList<Integer> docFilterLabels = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String label = labelVocab.get(labels[d][ii]);
                int filterLabelIndex = this.labelVocab.indexOf(label);
                if (filterLabelIndex >= 0) {
                    docFilterLabels.add(filterLabelIndex);
                }
            }

            filterLabels[d] = new int[docFilterLabels.size()];
            for (int ii = 0; ii < docFilterLabels.size(); ii++) {
                filterLabels[d][ii] = docFilterLabels.get(ii);
            }
        }

        this.labels = filterLabels;
    }

    /**
     * Filter labels that do not meet the minimum frequency requirement.
     *
     * @param minLabelFreq Minimum frequency
     */
    public void filterLabelsByFrequency(int minLabelFreq) {
        int D = words.length;
        int L = labelVocab.size();
        int[] labelFreqs = new int[L];
        for (int dd = 0; dd < D; dd++) {
            for (int ii = 0; ii < labels[dd].length; ii++) {
                labelFreqs[labels[dd][ii]]++;
            }
        }

        ArrayList<String> filterLabelVocab = new ArrayList<String>();
        for (int ll = 0; ll < L; ll++) {
            if (labelFreqs[ll] > minLabelFreq) {
                filterLabelVocab.add(labelVocab.get(ll));
            }
        }
        Collections.sort(filterLabelVocab);

        int[][] filterLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            ArrayList<Integer> docFilterLabels = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String label = labelVocab.get(labels[d][ii]);
                int filterLabelIndex = filterLabelVocab.indexOf(label);
                if (filterLabelIndex >= 0) {
                    docFilterLabels.add(filterLabelIndex);
                }
            }

            filterLabels[d] = new int[docFilterLabels.size()];
            for (int ii = 0; ii < docFilterLabels.size(); ii++) {
                filterLabels[d][ii] = docFilterLabels.get(ii);
            }
        }

        this.labels = filterLabels;
        this.labelVocab = filterLabelVocab;
    }

    public void loadLabels(File labelFile) throws Exception {
        loadLabels(labelFile.getAbsolutePath());
    }

    public void loadLabels(String labelFile) throws Exception {
        logln("--- Loading labels from " + labelFile);

        if (this.docIdList == null) {
            throw new RuntimeException("docIdList is null. Load text data first.");
        }

        HashMap<String, ArrayList<String>> docLabelMap = new HashMap<String, ArrayList<String>>();
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(labelFile);
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            String docId = sline[0];

            ArrayList<String> docLabels = new ArrayList<String>();
            for (int ii = 1; ii < sline.length; ii++) {
                docLabels.add(sline[ii]);
            }
            docLabelMap.put(docId, docLabels);
        }
        reader.close();

        this.labelList = new ArrayList<ArrayList<String>>();
        for (int ii = 0; ii < docIdList.size(); ii++) {
            ArrayList<String> docLabels = docLabelMap.get(docIdList.get(ii));
            this.labelList.add(docLabels);
        }
    }

    @Override
    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);

        formatLabels(outputFolder);

        // perform normal processing
        super.format(outputFolder);
    }

    public void formatLabels(String outputFolder) throws Exception {
        logln("Formatting labels ...");
        if (this.labelVocab == null) {
            createLabelVocab();
        }

        // output label vocab
        outputLabelVocab(outputFolder);

        // get label indices
        this.labels = new int[this.labelList.size()][];
        for (int ii = 0; ii < labels.length; ii++) {
            ArrayList<Integer> docLabels = new ArrayList<Integer>();
            for (int jj = 0; jj < labelList.get(ii).size(); jj++) {
                int labelIndex = labelVocab.indexOf(labelList.get(ii).get(jj));
                if (labelIndex >= 0) { // filter out labels not in label vocab
                    docLabels.add(labelIndex);
                }
            }

            this.labels[ii] = new int[docLabels.size()];
            for (int jj = 0; jj < labels[ii].length; jj++) {
                this.labels[ii][jj] = docLabels.get(jj);
            }
        }
    }

    /**
     * Output the list of unique labels
     *
     * @param outputFolder Output folder
     */
    protected void outputLabelVocab(String outputFolder) throws Exception {
        File labelVocFile = new File(outputFolder, formatFilename + labelVocabExt);
        logln("--- Outputing label vocab ... " + labelVocFile.getAbsolutePath());
        DataUtils.outputVocab(labelVocFile.getAbsolutePath(),
                this.labelVocab);
    }

    /**
     * Create label vocabulary
     */
    public void createLabelVocab() throws Exception {
        logln("--- Creating label vocab ...");
        createLabelVocabByFrequency();
    }

    protected void createLabelVocabByFrequency() throws Exception {
        HashMap<String, Integer> labelFreqs = new HashMap<String, Integer>();
        for (int ii = 0; ii < this.labelList.size(); ii++) {
            for (String label : this.labelList.get(ii)) {
                Integer count = labelFreqs.get(label);
                if (count == null) {
                    labelFreqs.put(label, 1);
                } else {
                    labelFreqs.put(label, count + 1);
                }
            }
        }

        ArrayList<RankingItem<String>> rankLabels = new ArrayList<RankingItem<String>>();
        for (String label : labelFreqs.keySet()) {
            int freq = labelFreqs.get(label);
            if (freq >= this.minLabelDocFreq) {
                rankLabels.add(new RankingItem<String>(label, labelFreqs.get(label)));
            }
        }
        Collections.sort(rankLabels);

        this.labelVocab = new ArrayList<String>();
        for (int k = 0; k < Math.min(this.maxLabelVocSize, rankLabels.size()); k++) {
            this.labelVocab.add(rankLabels.get(k).getObject());
        }
        Collections.sort(this.labelVocab);
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        logln("--- Outputing document info ... " + outputFile);

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex));
            for (int label : labels[docIndex]) {
                infoWriter.write("\t" + label);
            }
            infoWriter.write("\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File file) throws Exception {
        logln("--- Reading document info from " + file);

        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<int[]> labelIndexList = new ArrayList<int[]>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            int[] labelIndices = new int[sline.length - 1];
            for (int ii = 0; ii < sline.length - 1; ii++) {
                labelIndices[ii] = Integer.parseInt(sline[ii + 1]);
            }
            labelIndexList.add(labelIndices);
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.labels = new int[labelIndexList.size()][];
        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii] = labelIndexList.get(ii);
        }
    }

    public void outputArffFile(File filepath) {
        if (verbose) {
            logln("Outputing to " + filepath);
        }

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int ii = 0; ii < wordVocab.size(); ii++) {
            attributes.add(new Attribute("voc_" + wordVocab.get(ii)));
        }
        for (int ii = 0; ii < labelVocab.size(); ii++) {
            ArrayList<String> attVals = new ArrayList<String>();
            attVals.add("0");
            attVals.add("1");
            attributes.add(new Attribute("label_" + labelVocab.get(ii), attVals));
        }

        Instances data = new Instances(name, attributes, 0);
        for (int dd = 0; dd < docIds.length; dd++) {
            double[] vals = new double[wordVocab.size() + labelVocab.size()];

            // words
            SparseCount count = new SparseCount();
            for (int w : words[dd]) {
                count.increment(w);
            }
            for (int idx : count.getIndices()) {
                vals[idx] = count.getCount(idx);
            }

            // labels
            ArrayList<String> lbls = labelList.get(dd);
            for (int ll = 0; ll < labelVocab.size(); ll++) {
                if (lbls.contains(labelVocab.get(ll))) {
                    vals[ll + wordVocab.size()] = 1;
                }
            }

            data.add(new SparseInstance(1.0, vals));
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            writer.write(data.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing ARFF file");
        }
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            this.inputLabelVocab(new File(fFolder, formatFilename + labelVocabExt));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    protected void inputLabelVocab(File file) throws Exception {
        labelVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        while ((line = reader.readLine()) != null) {
            labelVocab.add(line);
        }
        reader.close();
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar:dist/lib/*' " + LabelTextDataset.class.getName() + " -help";
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
            addOption("label-file", "Directory of the label file");

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
            addOption("min-word-length", "Word minimum length");

            // cross validation
//            addOption("num-folds", "Number of folds. Default 5.");
//            addOption("tr2dev-ratio", "Training-to-development ratio. Default 0.8.");
//            addOption("cv-folder", "Folder to store cross validation folds");
//            addOption("num-classes", "Number of classes that the response");

            addOption("run-mode", "Run mode");

            options.addOption("v", false, "Verbose");
            options.addOption("d", false, "Debug");
            options.addOption("s", false, "Whether stopwords are filtered");
            options.addOption("l", false, "Whether lemmatization is performed");
            options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            String runMode = cmd.getOptionValue("run-mode");
            if (runMode.equals("process")) {
                process(args);
            } else if (runMode.equals("load")) {
                load(args);
            } else {
                throw new RuntimeException("Run mode " + runMode + " is not supported");
            }

        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static void process(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        String labelFile = cmd.getOptionValue("label-file");

        int unigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "u", 5);
        int bigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "b", 10);
        double bigramScoreCutoff = CLIUtils.getDoubleArgument(cmd, "bs", 5.0);
        int maxVocabSize = CLIUtils.getIntegerArgument(cmd, "V", Integer.MAX_VALUE);
        int vocTermFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-tf", 5);
        int vocTermFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-tf", Integer.MAX_VALUE);
        int vocDocFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-df", 5);
        int vocDocFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-df", Integer.MAX_VALUE);
        int docTypeCountCutoff = CLIUtils.getIntegerArgument(cmd, "min-doc-length", 10);

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

        LabelTextDataset dataset = new LabelTextDataset(datasetName, datasetFolder, corpProc);
        dataset.setFormatFilename(formatFile);

        // load text data
        if (cmd.hasOption("file")) {
            dataset.loadTextDataFromFile(textInputData);
        } else {
            dataset.loadTextDataFromFolder(textInputData);
        }
        dataset.loadLabels(labelFile); // load response data
        dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder));
    }

    public static LabelTextDataset load(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        LabelTextDataset data = new LabelTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        return data;
    }
}
