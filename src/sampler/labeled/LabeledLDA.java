package sampler.labeled;

import core.AbstractSampler;
import data.LabelTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 * This is an implementation of a Gibbs sampler for Labeled LDA (Ramage et. al.
 * EMNLP 09).
 *
 * Each document is associated with a set of labels.
 *
 * @author vietan
 */
public class LabeledLDA extends AbstractSampler implements Serializable {
    private static final long serialVersionUID = 1123581321L;

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int L;
    protected int V;
    protected int D;
    private DirMult[] doc_labels;
    private DirMult[] label_words;
    private int[][] z;
    private ArrayList<String> labelVocab;
    private int numTokens;
    private int numTokensChange;

    public void setLabelVocab(ArrayList<String> labelVocab) {
        this.labelVocab = labelVocab;
    }

    public void configure(String folder,
            int[][] words, int[][] labels,
            int V, int L,
            double alpha,
            double beta,
            InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.words = words;
        this.labels = labels;

        this.L = L;
        this.V = V;
        this.D = this.words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInterval;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- label vocab:\t" + L);
            logln("--- word vocab:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            logln("--- beta:\t" + MiscUtils.formatDouble(beta));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_L-LDA")
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(MiscUtils.formatDouble(hyperparams.get(ALPHA)))
                .append("_b-").append(MiscUtils.formatDouble(hyperparams.get(BETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public DirMult[] getTopicWordDistributions() {
        return this.label_words;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    private void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        label_words = new DirMult[L];
        for (int ll = 0; ll < L; ll++) {
            label_words[ll] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        doc_labels = new DirMult[D];
        for (int d = 0; d < D; d++) {
            doc_labels[d] = new DirMult(L, hyperparams.get(ALPHA) * L, 1.0 / L);
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                int[] docLabels = labels[d];
                if (docLabels.length > 0) {
                    z[d][n] = docLabels[rand.nextInt(docLabels.length)];
                } else {
                    z[d][n] = rand.nextInt(L);
                }
                doc_labels[d].increment(z[d][n]);
                label_words[z[d][n]].increment(words[d][n]);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        try {
            if (report) {
                IOUtils.createFolder(new File(getSamplerFolderPath(), ReportFolder));
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChange = 0;

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD);
                }
            }

            if (debug) {
                validate("iter " + iter);
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % REP_INTERVAL == 0) {
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t tokens changed: " + ((double) numTokensChange / numTokens)
                        + "\t" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str + "\n");
                } else {
                    logln("--- Sampling. " + str + "\n");
                }
            }

            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    if (verbose) {
                        logln("*** *** Optimizing hyperparameters by slice sampling ...");
                        logln("*** *** cur param:" + MiscUtils.listToString(hyperparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }

                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        logln("*** *** new param:" + MiscUtils.listToString(sparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }
                }
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(
                        new File(getSamplerFolderPath(), ReportFolder),
                        "iter-" + iter + ".zip"));
            }
        }

        if (report) {
            outputState(new File(
                    new File(getSamplerFolderPath(), ReportFolder),
                    "iter-" + iter + ".zip"));
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (log) {
                BufferedWriter writer = IOUtils.getBufferedWriter(
                        new File(getSamplerFolderPath(), "likelihoods.txt"));
                for (int i = 0; i < logLikelihoods.size(); i++) {
                    writer.write(i + "\t" + logLikelihoods.get(i) + "\n");
                }
                writer.close();

                if (paramOptimized) {
                    outputSampledHyperparameters(new File(getSamplerFolderPath(),
                            "hyperparameters.txt"));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Sample topic assignment for each token.
     *
     * @param d Document index
     * @param n Token index
     * @param removeFromModel Whether the current assignment should be removed
     * from the model (i.e., label-word distributions)
     * @param addToModel Whether the new assignment should be added to the model
     * @param removeFromData Whether the current assignment should be removed
     * from the data (i.e., doc-label distributions)
     * @param addToData Whether the new assignment should be added to the data
     */
    public void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            label_words[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            doc_labels[d].decrement(z[d][n]);
        }

        int sampledZ;
        if (labels[d].length > 0) {
            double[] logprobs = new double[labels[d].length];
            for (int ii = 0; ii < labels[d].length; ii++) {
                logprobs[ii] = doc_labels[d].getLogLikelihood(labels[d][ii])
                        + label_words[labels[d][ii]].getLogLikelihood(words[d][n]);
            }
            sampledZ = labels[d][SamplerUtils.logMaxRescaleSample(logprobs)];
        } else {
            double[] logprobs = new double[L];
            for (int ll = 0; ll < L; ll++) {
                logprobs[ll] = doc_labels[d].getLogLikelihood(ll)
                        + label_words[ll].getLogLikelihood(words[d][n]);
            }
            sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
        }

        if (sampledZ != z[d][n]) {
            numTokensChange++;
        }
        z[d][n] = sampledZ;

        if (addToModel) {
            label_words[z[d][n]].increment(words[d][n]);
        }
        if (addToData) {
            doc_labels[d].increment(z[d][n]);
        }
    }

    public double[] predictNewDocument(int[] newDoc) throws Exception{
        // initialize assignments
        DirMult docTopic = new DirMult(L, hyperparams.get(ALPHA) * L, 1.0 / L);
        int[] newZ = new int[newDoc.length];
        for (int n = 0; n < newZ.length; n++) {
            newZ[n] = rand.nextInt(L);
            docTopic.increment(newZ[n]);
        }
        // sample
        for (iter = 0; iter < MAX_ITER; iter++) {
            for(int n=0; n<newZ.length; n++) {
                // decrement
                docTopic.decrement(newZ[n]);
                
                // sample
                double[] logprobs = new double[L];
                for(int l=0; l<L; l++) {
                    logprobs[l] = docTopic.getLogLikelihood(l)
                            + label_words[l].getLogLikelihood(newDoc[n]);
                }
                newZ[n] = SamplerUtils.logMaxRescaleSample(logprobs);
                
                // increment
                docTopic.increment(newZ[n]);
            }
        }
        
        return docTopic.getDistribution();
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += doc_labels[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int l = 0; l < L; l++) {
            topicWordLlh += label_words[l].getLogLikelihood();
        }

        double llh = docTopicLlh + topicWordLlh;
        if (verbose) {
            logln(">>> doc-topic: " + MiscUtils.formatDouble(docTopicLlh)
                    + "\ttopic-word: " + MiscUtils.formatDouble(topicWordLlh)
                    + "\tllh: " + MiscUtils.formatDouble(llh));
        }
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double llh = 0;
        for (int d = 0; d < D; d++) {
            llh += doc_labels[d].getLogLikelihood(newParams.get(ALPHA) * L, 1.0 / L);
        }
        for (int l = 0; l < L; l++) {
            llh += label_words[l].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.doc_labels[d].setConcentration(this.hyperparams.get(ALPHA) * L);
        }
        for (int l = 0; l < L; l++) {
            this.label_words[l].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            this.doc_labels[d].validate(msg);
        }
        for (int l = 0; l < L; l++) {
            this.label_words[l].validate(msg);
        }

        int total = 0;
        for (int d = 0; d < D; d++) {
            total += doc_labels[d].getCountSum();
        }
        if (total != numTokens) {
            throw new RuntimeException("Token counts mismatch. "
                    + total + " vs. " + numTokens);
        }

        total = 0;
        for (int l = 0; l < L; l++) {
            total += label_words[l].getCountSum();
        }
        if (total != numTokens) {
            throw new RuntimeException("Token counts mismatch. "
                    + total + " vs. " + numTokens);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int k = 0; k < L; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(label_words[k])).append("\n");
            }

            // data
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(doc_labels[d])).append("\n");

                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }

        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        validate("Done reading state from " + filepath);
    }

    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            // initialize
            this.initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < L; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                label_words[k] = DirMult.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                doc_labels[d] = DirMult.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (this.labelVocab == null) {
            throw new RuntimeException("The topic vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < L; k++) {
            double[] distrs = label_words[k].getDistribution();
            String[] topWords = getTopWords(distrs, numTopWords);
            writer.write("[" + k
                    + ", " + labelVocab.get(k)
                    + ", " + label_words[k].getCountSum()
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");
        }
        writer.close();
    }

    public void outputTopicCoherence(
            File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing topic coherence to file " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < L; k++) {
            double[] distribution = this.label_words[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + label_words[k].getCountSum()
                    + "\t" + MiscUtils.formatDouble(score));
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public static void main(String[] args) {
        run(args);
    }

    public static void run(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("dataset", "Dataset");
            addOption("data-folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("format-file", "Formatted file name");
            addOption("output", "Output folder");

            // sampling configurations
            addOption("burnIn", "Burn-in");
            addOption("maxIter", "Maximum number of iterations");
            addOption("sampleLag", "Sample lag");
            addOption("report", "Report interval");

            // model parameters
            addOption("K", "Number of topics");
            addOption("numTopwords", "Number of top words per topic");
            addOption("min-label-freq", "Minimum label frequency");

            // model hyperparameters
            addOption("alpha", "Hyperparameter of the symmetric Dirichlet prior "
                    + "for topic distributions");
            addOption("beta", "Hyperparameter of the symmetric Dirichlet prior "
                    + "for word distributions");

            // running configurations
//            addOption("cv-folder", "Cross validation folder");
//            addOption("num-folds", "Number of folds");
//            addOption("fold", "The cross-validation fold to run");
//            addOption("run-mode", "Running mode");

            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            runModel();
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + LabeledLDA.class.getName() + " -help";
    }

    private static void runModel() throws Exception {
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "112");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "L:/Dropbox/github/data");
        String outputFolder = CLIUtils.getStringArgument(cmd, "output", "L:/Dropbox/github/data/112/format-label/model");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format-label");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        int minLabelFreq = CLIUtils.getIntegerArgument(cmd, "min-label-freq", 300);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);

        boolean verbose = true;
        boolean debug = true;

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        LabelTextDataset data = new LabelTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
        data.filterLabelsByFrequency(minLabelFreq);
        data.prepareTopicCoherence(numTopWords);

        int V = data.getWordVocab().size();
        int K = data.getLabelVocab().size();
        boolean paramOpt = cmd.hasOption("paramOpt");
        InitialState initState = InitialState.RANDOM;

        if (verbose) {
            System.out.println("\tRunning Labeled-LDA sampler ...");
        }
        LabeledLDA sampler = new LabeledLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(data.getWordVocab());
        sampler.setLabelVocab(data.getLabelVocab());

        sampler.configure(outputFolder, data.getWords(), data.getLabels(),
                V, K, alpha, beta, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File lldaFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(lldaFolder);
        sampler.sample();
        sampler.outputTopicTopWords(
                new File(lldaFolder, TopWordFile),
                numTopWords);
        sampler.outputTopicCoherence(
                new File(lldaFolder, TopicCoherenceFile),
                data.getTopicCoherence());
    }
}
