package data;

import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.DataUtils;
import util.IOUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 * A set of documents
 *
 * @author vietan
 */
public class TextDataset extends AbstractTokenizeDataset {

    protected ArrayList<String> docIdList; // original list of document ids
    protected ArrayList<String> textList; // raw text of documents
    protected ArrayList<Integer> processedDocIndices; // list of document ids after pre-processing
    protected ArrayList<String> wordVocab;
    protected String[] docIds;
    protected int[][] words;
    protected int[][][] sentWords;
    protected String[][] sentRawWords;
    protected MimnoTopicCoherence topicCoherence;
    protected double[] tfidfs;
    protected double[] idfs;

    public TextDataset(String name) {
        super(name);
    }

    public TextDataset(
            String name,
            String folder) {
        super(name, folder);

        this.docIdList = new ArrayList<String>();
        this.textList = new ArrayList<String>();
        this.processedDocIndices = new ArrayList<Integer>();
    }

    public TextDataset(
            String name,
            String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);

        this.docIdList = new ArrayList<String>();
        this.textList = new ArrayList<String>();
        this.processedDocIndices = new ArrayList<Integer>();
    }

    /**
     * Compute the TF-IDF score of each item in the vocabulary.
     */
    public void computeTFIDFs() {
        int V = this.wordVocab.size();
        int D = this.words.length;
        SparseCount tfs = new SparseCount();
        SparseCount dfs = new SparseCount();
        for (int d = 0; d < D; d++) {
            Set<Integer> uniqueWords = new HashSet<Integer>();
            for (int n = 0; n < words[d].length; n++) {
                uniqueWords.add(words[d][n]);
                tfs.increment(words[d][n]);
            }

            for (int w : uniqueWords) {
                dfs.increment(w);
            }
        }

        this.tfidfs = new double[V];
        this.idfs = new double[V];
        for (int v = 0; v < V; v++) {
            double tf = Math.log(tfs.getCount(v));
            double idf = Math.log(D) - Math.log(dfs.getCount(v));
            this.tfidfs[v] = tf * idf;
            this.idfs[v] = idf;
        }
    }

    public double[] getTFIDFs() {
        return this.tfidfs;
    }

    public double[] getIDFs() {
        return this.idfs;
    }

    public String[][] getRawSentences() {
        return this.sentRawWords;
    }

    public ArrayList<String> getDocIdList() {
        return this.docIdList;
    }

    public ArrayList<String> getTextList() {
        return this.textList;
    }

    public void prepareTopicCoherence(int numTopWords) {
        this.topicCoherence = new MimnoTopicCoherence(words, wordVocab.size(), numTopWords);
        this.topicCoherence.prepare();
    }

    public MimnoTopicCoherence getTopicCoherence() {
        return this.topicCoherence;
    }

    public int[][][] getSentenceWords() {
        return this.sentWords;
    }

    public ArrayList<String> getWordVocab() {
        return this.wordVocab;
    }

    public int[][][] getDocSentWords(ArrayList<Integer> instances) {
        int[][][] revSentWords = new int[instances.size()][][];
        for (int i = 0; i < revSentWords.length; i++) {
            int idx = instances.get(i);
            revSentWords[i] = this.sentWords[idx];
        }
        return revSentWords;
    }

    public int[][] getDocWords(ArrayList<Integer> instances) {
        int[][] revWords = new int[instances.size()][];
        for (int i = 0; i < revWords.length; i++) {
            int idx = instances.get(i);
            revWords[i] = this.words[idx];
        }
        return revWords;
    }

    /**
     * Set the raw texts with their IDs
     *
     * @param docIdList List of document IDs
     * @param textList List of texts
     */
    public void setTextData(ArrayList<String> docIdList, ArrayList<String> textList) {
        this.docIdList = docIdList;
        this.textList = textList;
    }

    /**
     * Load text data from a single file where each line has the following
     * format <doc_Id>\t<text>\n
     *
     * @param textFile The input data file
     */
    public void loadTextDataFromFile(File textFile) throws Exception {
        loadTextDataFromFile(textFile.getAbsolutePath());
    }

    /**
     * Load text data from a single file where each line has the following
     * format <doc_Id>\t<text>\n
     *
     * @param textFilepath The input data file
     */
    public void loadTextDataFromFile(String textFilepath) throws Exception {
        if (verbose) {
            logln("--- Loading text data from file " + textFilepath);
        }

        BufferedReader reader = IOUtils.getBufferedReader(textFilepath);
        String line;
        while ((line = reader.readLine()) != null) {
            docIdList.add(line.substring(0, line.indexOf("\t")));
            textList.add(line.substring(line.indexOf("\t") + 1));
        }
        reader.close();

        if (verbose) {
            logln("--- --- Loaded " + docIdList.size() + " document(s)");
        }
    }

    /**
     * Load text data from a folder where each file contains the text of a
     * document, each filename is in the form of <doc_Id>.txt
     *
     * @param textFolder The input data folder
     */
    public void loadTextDataFromFolder(File textFolder) throws Exception {
        loadTextDataFromFolder(textFolder.getAbsolutePath());
    }

    /**
     * Load text data from a folder where each file contains the text of a
     * document, each filename is in the form of <doc_Id>.txt
     *
     * @param textFolderPath The input data folder
     */
    public void loadTextDataFromFolder(String textFolderPath) throws Exception {
        if (verbose) {
            logln("--- Loading text data from folder " + textFolderPath);
        }

        File fd = new File(textFolderPath);
        String[] filenames = fd.list();
        BufferedReader reader;
        String line;
        StringBuilder docText;
        for (String filename : filenames) {
            docIdList.add(IOUtils.removeExtension(filename));
            reader = IOUtils.getBufferedReader(new File(fd, filename));
            docText = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                docText.append(line).append("\n");
            }
            reader.close();

            textList.add(docText.toString());
        }

        if (verbose) {
            logln("--- --- Loaded " + docIdList.size() + " document(s)");
        }
    }

    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    /**
     * Format input data
     *
     * @param outputFolder The directory of the folder that processed data will
     * be stored
     */
    public void format(String outputFolder) throws Exception {
        if (verbose) {
            logln("--- Processing data ...");
        }
        IOUtils.createFolder(outputFolder);

        String[] rawTexts = textList.toArray(new String[textList.size()]);
        corpProc.setRawTexts(rawTexts);
        corpProc.process();

        outputWordVocab(outputFolder);
        outputTextData(outputFolder);
        outputDocumentInfo(outputFolder);
        outputSentTextData(outputFolder);
    }

    /**
     * Output the word vocabulary
     *
     * @param outputFolder Output folder
     */
    protected void outputWordVocab(String outputFolder) throws Exception {
        File wordVocFile = new File(outputFolder, formatFilename + wordVocabExt);
        if (verbose) {
            logln("--- Outputing word vocab ... " + wordVocFile.getAbsolutePath());
        }
        DataUtils.outputVocab(wordVocFile.getAbsolutePath(),
                corpProc.getVocab());
    }

    /**
     * Output the formatted document data.
     *
     * @param outputFolder Output folder
     */
    protected void outputTextData(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + numDocDataExt);
        if (verbose) {
            logln("--- Outputing main numeric data ... " + outputFile);
        }

        // output main numeric
        int[][] numDocs = corpProc.getNumerics();
        BufferedWriter dataWriter = IOUtils.getBufferedWriter(outputFile);
        for (int d = 0; d < numDocs.length; d++) {
            HashMap<Integer, Integer> typeCounts = new HashMap<Integer, Integer>();
            for (int j = 0; j < numDocs[d].length; j++) {
                Integer count = typeCounts.get(numDocs[d][j]);
                if (count == null) {
                    typeCounts.put(numDocs[d][j], 1);
                } else {
                    typeCounts.put(numDocs[d][j], count + 1);
                }
            }

            // skip short documents
            if (typeCounts.size() < corpProc.docTypeCountCutoff) {
                continue;
            }

            // write main data
            dataWriter.write(Integer.toString(typeCounts.size()));
            for (int type : typeCounts.keySet()) {
                dataWriter.write(" " + type + ":" + typeCounts.get(type));
            }
            dataWriter.write("\n");

            // save the doc id
            this.processedDocIndices.add(d);
        }
        dataWriter.close();
    }

    /**
     * Output the formatted data.
     *
     * @param outputFolder Output folder
     */
    protected void outputSentTextData(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + numSentDataExt);
        if (verbose) {
            logln("--- Outputing sentence data ... " + outputFile);
        }

        int[][][] numSents = corpProc.getNumericSentences();
        String[][] rawSents = corpProc.getRawSentences();
        BufferedWriter rawSentWriter = IOUtils.getBufferedWriter(outputFile + ".raw");
        BufferedWriter sentWriter = IOUtils.getBufferedWriter(outputFile);
        for (int d : this.processedDocIndices) {
            StringBuilder docStr = new StringBuilder();
            ArrayList<String> docRawSents = new ArrayList<String>();

            for (int s = 0; s < numSents[d].length; s++) {
                HashMap<Integer, Integer> sentTypeCounts = new HashMap<Integer, Integer>();
                for (int w = 0; w < numSents[d][s].length; w++) {
                    Integer count = sentTypeCounts.get(numSents[d][s][w]);
                    if (count == null) {
                        sentTypeCounts.put(numSents[d][s][w], 1);
                    } else {
                        sentTypeCounts.put(numSents[d][s][w], count + 1);
                    }
                }

                if (sentTypeCounts.size() > 0) {
                    // store numeric sentence
                    StringBuilder str = new StringBuilder();
                    for (int type : sentTypeCounts.keySet()) {
                        str.append(type).append(":").append(sentTypeCounts.get(type)).append(" ");
                    }
                    docStr.append(str.toString().trim()).append("\t");

                    // store raw sentence
                    docRawSents.add(rawSents[d][s]);
                }
            }
            // write numeric sentence
            sentWriter.write(docStr.toString().trim() + "\n");

            // write raw sentence
            rawSentWriter.write(docRawSents.size() + "\n");
            for (String docRawSent : docRawSents) {
                rawSentWriter.write(docRawSent.trim().replaceAll("\n", " ") + "\n");
            }
        }
        sentWriter.close();
        rawSentWriter.close();
    }

    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        if (verbose) {
            logln("--- Outputing document info ... " + outputFile.getAbsolutePath());
        }

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex) + "\n");
        }
        infoWriter.close();
    }

    public String[] getDocIds() {
        return docIds;
    }

    public int[][] getWords() {
        return this.words;
    }

    public void loadFormattedData(File fFolder) {
        this.loadFormattedData(fFolder.getAbsolutePath());
    }

    public void loadFormattedData(String fFolder) {
        if (verbose) {
            logln("--- Loading formatted data from " + fFolder);
        }
        try {
            inputWordVocab(new File(fFolder, formatFilename + wordVocabExt));
            inputTextData(new File(fFolder, formatFilename + numDocDataExt));
            inputDocumentInfo(new File(fFolder, formatFilename + docInfoExt));
            inputSentenceTextData(new File(fFolder, formatFilename + numSentDataExt));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    protected void inputWordVocab(File file) throws Exception {
        if (verbose) {
            logln("--- Loading word vocab from " + file);
        }

        wordVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        while ((line = reader.readLine()) != null) {
            wordVocab.add(line);
        }
        reader.close();

        if (verbose) {
            logln("--- --- # unique words: " + wordVocab.size());
        }
    }

    protected void inputTextData(File file) throws Exception {
        if (verbose) {
            logln("--- Reading text data from " + file);
        }

        BufferedReader reader = IOUtils.getBufferedReader(file);

        ArrayList<int[]> wordList = new ArrayList<int[]>();
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            sline = line.split(" ");

            int numTypes = Integer.parseInt(sline[0]);
            int[] types = new int[numTypes];
            int[] counts = new int[numTypes];

            int numTokens = 0;
            for (int ii = 0; ii < numTypes; ++ii) {
                String[] entry = sline[ii + 1].split(":");
                int count = Integer.parseInt(entry[1]);
                int id = Integer.parseInt(entry[0]);
                numTokens += count;
                types[ii] = id;
                counts[ii] = count;
            }

            int[] gibbsString = new int[numTokens];
            int index = 0;
            for (int ii = 0; ii < numTypes; ++ii) {
                for (int jj = 0; jj < counts[ii]; ++jj) {
                    gibbsString[index++] = types[ii];
                }
            }
            wordList.add(gibbsString);
        }
        words = wordList.toArray(new int[wordList.size()][]);
        reader.close();

        if (verbose) {
            logln("--- --- # docs: " + words.length);
            int numTokens = 0;
            for (int dd = 0; dd < words.length; dd++) {
                numTokens += words[dd].length;
            }
            logln("--- --- # tokens: " + numTokens);
        }
    }

    protected void inputSentenceTextData(File file) throws Exception {
        if (verbose) {
            logln("--- Reading sentence text data from " + file);
        }

        BufferedReader numSentReader = IOUtils.getBufferedReader(file);
        ArrayList<int[][]> sentWordList = new ArrayList<int[][]>();
        String line;
        String[] sline;
        while ((line = numSentReader.readLine()) != null) {
            sline = line.split("\t");
            int numSents = sline.length;
            int[][] sents = new int[numSents][];
            for (int s = 0; s < numSents; s++) {
                String[] sSent = sline[s].split(" ");
                int numTokens = 0;
                HashMap<Integer, Integer> typeCounts = new HashMap<Integer, Integer>();

                for (String sSentWord : sSent) {
                    int type = Integer.parseInt(sSentWord.split(":")[0]);
                    int count = Integer.parseInt(sSentWord.split(":")[1]);
                    numTokens += count;
                    typeCounts.put(type, count);
                }

                int[] tokens = new int[numTokens];
                int idx = 0;
                for (int type : typeCounts.keySet()) {
                    for (int ii = 0; ii < typeCounts.get(type); ii++) {
                        tokens[idx++] = type;
                    }
                }
                sents[s] = tokens;
            }
            sentWordList.add(sents);
        }
        numSentReader.close();

        sentWords = new int[sentWordList.size()][][];
        for (int i = 0; i < sentWords.length; i++) {
            sentWords[i] = sentWordList.get(i);
        }

        if (verbose) {
            logln("--- --- # docs: " + sentWords.length);
            int numSents = 0;
            int numTokens = 0;
            for (int d = 0; d < sentWords.length; d++) {
                numSents += sentWords[d].length;
                for (int s = 0; s < sentWords[d].length; s++) {
                    numTokens += sentWords[d][s].length;
                }
            }
            logln("--- --- # sents: " + numSents);
            logln("--- --- # tokens: " + numTokens);
        }


        File rawSentFile = new File(file + ".raw");
        if (verbose) {
            logln("--- Reading sentence raw text data from " + rawSentFile);
        }
        if (rawSentFile.exists()) {
            try {
                sentRawWords = new String[sentWords.length][];
                int count = 0;
                BufferedReader rawSentReader = IOUtils.getBufferedReader(rawSentFile);
                while ((line = rawSentReader.readLine()) != null) {
                    int numSents = Integer.parseInt(line);
                    String[] docRawSents = new String[numSents];
                    for (int ii = 0; ii < numSents; ii++) {
                        docRawSents[ii] = rawSentReader.readLine();
                    }
                    sentRawWords[count++] = docRawSents;
                }
                rawSentReader.close();

                if (verbose) {
                    logln("--- --- # docs: " + sentRawWords.length);
                    int numSents = 0;
                    for (int dd = 0; dd < sentRawWords.length; dd++) {
                        numSents += sentRawWords[dd].length;
                    }
                    logln("--- --- # sents: " + numSents);
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("Exception while loading raw sentences from "
                        + rawSentFile);
            } finally {
                return;
            }
        }
    }

    protected void inputDocumentInfo(File file) throws Exception {
        if (verbose) {
            logln("--- Reading document info from " + file);
        }

        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        ArrayList<String> dIdList = new ArrayList<String>();

        while ((line = reader.readLine()) != null) {
            dIdList.add(line.split("\t")[0]);
        }
        reader.close();

        this.docIds = dIdList.toArray(new String[dIdList.size()]);
    }

    /**
     * Create cross validation
     *
     * @param cvFolder Cross validation folder
     * @param numFolds Number of folds
     * @param trToDevRatio Ratio between the number of training and the number
     * of test data
     */
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
            TextDataset trainData = new TextDataset(fold.getFoldName(), cv.getFolderPath(), cp);
            trainData.setFormatFilename(fold.getFoldName() + Fold.TrainingExt);
            ArrayList<String> trDocIds = new ArrayList<String>();
            ArrayList<String> trDocTexts = new ArrayList<String>();
            for (int idx : fold.getTrainingInstances()) {
                trDocIds.add(this.docIdList.get(idx));
                trDocTexts.add(this.textList.get(idx));
            }
            trainData.setTextData(trDocIds, trDocTexts);
            trainData.format(fold.getFoldFolderPath());

            // development data
            TextDataset devData = new TextDataset(fold.getFoldName(), cv.getFolderPath(), cp);
            devData.setFormatFilename(fold.getFoldName() + Fold.DevelopExt);
            ArrayList<String> deDocIds = new ArrayList<String>();
            ArrayList<String> deDocTexts = new ArrayList<String>();
            for (int idx : fold.getDevelopmentInstances()) {
                deDocIds.add(this.docIdList.get(idx));
                deDocTexts.add(this.textList.get(idx));
            }
            devData.setTextData(deDocIds, deDocTexts);
            devData.format(fold.getFoldFolderPath());

            // test data
            TextDataset testData = new TextDataset(fold.getFoldName(), cv.getFolderPath(), cp);
            testData.setFormatFilename(fold.getFoldName() + Fold.TestExt);
            ArrayList<String> teDocIds = new ArrayList<String>();
            ArrayList<String> teDocTexts = new ArrayList<String>();
            for (int idx : fold.getTestingInstances()) {
                teDocIds.add(this.docIdList.get(idx));
                teDocTexts.add(this.textList.get(idx));
            }
            testData.setTextData(teDocIds, teDocTexts);
            testData.format(fold.getFoldFolderPath());
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar:dist/lib/*' " + TextDataset.class.getName() + " -help";
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
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "process");
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
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static void crossValidate(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");

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

        TextDataset dataset = new TextDataset(datasetName, datasetFolder, corpProc);
        // load text data
        if (cmd.hasOption("file")) {
            dataset.loadTextDataFromFile(textInputData);
        } else {
            dataset.loadTextDataFromFolder(textInputData);
        }

        dataset.createCrossValidation(cvFolder, numFolds, trToDevRatio);
    }

    public static TextDataset load(String[] args) throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        TextDataset data = new TextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        return data;
    }

    public static void process(String[] args) throws Exception {
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon-data");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "demo");
        String textInputData = CLIUtils.getStringArgument(cmd, "text-data", "demo/amazon-data/raw/text.txt");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
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

        TextDataset dataset = new TextDataset(datasetName, datasetFolder, corpProc);
        dataset.setFormatFilename(formatFile);

        // load text data
        if (cmd.hasOption("file")) {
            dataset.loadTextDataFromFile(textInputData);
        } else {
            dataset.loadTextDataFromFolder(textInputData);
        }
        dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder));
    }
}
