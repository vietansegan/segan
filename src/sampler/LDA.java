package sampler;

import core.AbstractSampler;
import data.TextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
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
 * Implementation of a Gibbs sampler for LDA.
 *
 * @author vietan
 */
public class LDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected int K;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int[][] words;  // [D] x [Nd]: words
    protected int[][] z;
    protected DirMult[] doc_topics;
    protected DirMult[] topic_words;
    protected int numTokens;
    protected int numTokensChanged;

    public void configure(String folder, int[][] words,
            int V, int K,
            double alpha,
            double beta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.words = words;

        this.K = K;
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
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (verbose && folder != null) {
            logln("--- folder\t" + folder);
            logln("--- # documents:\t" + D);
            logln("--- # topics:\t" + K);
            logln("--- # tokens:\t" + numTokens);
            logln("--- vocab size:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_LDA"
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(this.hyperparams.get(ALPHA))
                + "_b-" + formatter.format(this.hyperparams.get(BETA))
                + "_opt-" + this.paramOptimized;
    }

    public int[][] getZ() {
        return this.z;
    }

    /**
     * Configure for new documents.
     *
     * @param ws New document words
     */
    public void configure(int[][] ws) {
        this.words = ws;
        this.D = this.words.length;

        this.initializeDataStructure(null);

        // initialize assignments for new documents
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
            }
        }
    }

    /**
     * Sample assignments for new documents given a learned model.
     *
     * @param ws New document words
     */
    public void sample(int[][] ws) {
        if (verbose) {
            logln("Sampling for new documents ...");
        }

        configure(ws);

        // sample
        logLikelihoods = new ArrayList<Double>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    sampleZ(d, t, !REMOVE, !ADD);
                }
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % REP_INTERVAL == 0) {
                double changeRatio = (double) numTokensChanged / numTokens;
                String str = "Iter " + iter
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChanged
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio);
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }
        }
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure(null);

        initializeDataStructure(null);

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    public void initialize(double[][] docTopicPrior, double[][] topicWordPrior) {
        if (verbose) {
            logln("Initializing with pre-defined topics ...");
        }

        initializeModelStructure(topicWordPrior);

        initializeDataStructure(docTopicPrior);

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeModelStructure(double[][] topics) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        topic_words = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (topics != null) {
                topic_words[k] = new DirMult(V, hyperparams.get(BETA) * V, topics[k]);
            } else {
                topic_words[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }
    }

    protected void initializeDataStructure(double[][] docTopicPrior) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        doc_topics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            if (docTopicPrior != null) {
                doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, docTopicPrior[d]);
            } else {
                doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
            }
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
                topic_words[z[d][n]].increment(words[d][n]);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    sampleZ(d, t, REMOVE, ADD);
                }
            }

            if (debug) {
                validate("Iter " + iter);
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                double changeRatio = (double) numTokensChanged / numTokens;
                String str = "Iter " + iter
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChanged
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio);
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }
        }
    }

    /**
     * Sample the topic assignment for each token
     *
     * @param d The document index
     * @param n The token index
     * @param remove Whether this token should be removed from the current
     * assigned topic
     * @param add Whether this token should be added to the sampled topic
     */
    protected void sampleZ(int d, int n, boolean remove, boolean add) {
        doc_topics[d].decrement(z[d][n]);
        if (remove) {
            topic_words[z[d][n]].decrement(words[d][n]);
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k] =
                    doc_topics[d].getLogLikelihood(k)
                    + topic_words[k].getLogLikelihood(words[d][n]);
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
        if (sampledZ != z[d][n]) {
            numTokensChanged++;
        }
        z[d][n] = sampledZ;

        doc_topics[d].increment(z[d][n]);
        if (add) {
            topic_words[z[d][n]].increment(words[d][n]);
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += doc_topics[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int k = 0; k < K; k++) {
            topicWordLlh += topic_words[k].getLogLikelihood();
        }
        return docTopicLlh + topicWordLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double llh = 0;
        for (int d = 0; d < D; d++) {
            llh += doc_topics[d].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }
        for (int k = 0; k < K; k++) {
            llh += topic_words[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.doc_topics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topic_words[k].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing topics to file " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            String[] topWords = getTopWords(topic_words[k].getDistribution(), numTopWords);
            // output top words
            writer.write("[Topic " + k + ": " + topic_words[k].getCountSum() + "]");
            for (String tw : topWords) {
                writer.write(" " + tw);
            }
            writer.write("\n\n");
        }
        writer.close();
    }

    public void outputTopicTopWordsCummProbs(String filepath, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        double[][] distrs = new double[K][];
        for (int k = 0; k < K; k++) {
            distrs[k] = topic_words[k].getDistribution();
        }
        IOUtils.outputTopWordsWithProbs(distrs, wordVocab, numTopWords, filepath);
    }

    public void outputTopicWordDistribution(String outputFile) throws Exception {
        double[][] pi = new double[K][];
        for (int k = 0; k < K; k++) {
            pi[k] = this.topic_words[k].getDistribution();
        }
        IOUtils.outputDistributions(pi, outputFile);
    }

    public double[][] inputTopicWordDistribution(String inputFile) throws Exception {
        return IOUtils.inputDistributions(inputFile);
    }

    public void outputDocumentTopicDistribution(String outputFile) throws Exception {
        double[][] theta = new double[D][];
        for (int d = 0; d < D; d++) {
            theta[d] = this.doc_topics[d].getDistribution();
        }
        IOUtils.outputDistributions(theta, outputFile);
    }

    public double[][] inputDocumentTopicDistribution(String inputFile) throws Exception {
        return IOUtils.inputDistributions(inputFile);
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            doc_topics[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            topic_words[k].validate(msg);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        try {
            StringBuilder modelStr = new StringBuilder();
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(topic_words[k])).append("\n");
            }

            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(doc_topics[d])).append("\n");
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
            this.initializeModelStructure(null);

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < K; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                topic_words[k] = DirMult.input(reader.readLine());
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
            this.initializeDataStructure(null);

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                doc_topics[d] = DirMult.input(reader.readLine());

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

    /**
     * Output topic coherence
     *
     * @param file Output file
     * @param topicCoherence Topic coherence
     */
    public void outputTopicCoherence(
            File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            System.out.println("Outputing topic coherence to file " + file);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            double[] distribution = this.topic_words[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + topic_words[k].getCountSum()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public DirMult[] getTopics() {
        return this.topic_words;
    }

    public double[][] getDocumentEmpiricalDistributions() {
        double[][] docEmpDists = new double[D][K];
        for (int d = 0; d < D; d++) {
            docEmpDists[d] = doc_topics[d].getEmpiricalDistribution();
        }
        return docEmpDists;
    }

    public void outputDocTopicDistributions(File file) throws Exception {
        if (verbose) {
            logln("Outputing per-document topic distribution to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int d = 0; d < D; d++) {
            writer.write(Integer.toString(d));
            double[] docTopicDist = this.doc_topics[d].getDistribution();
            for (int k = 0; k < K; k++) {
                writer.write("\t" + docTopicDist[k]);
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void outputTopicWordDistributions(File file) throws Exception {
        if (verbose) {
            logln("Outputing per-topic word distribution to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            writer.write(Integer.toString(k));
            double[] topicWordDist = this.topic_words[k].getDistribution();
            for (int v = 0; v < V; v++) {
                writer.write("\t" + topicWordDist[v]);
            }
            writer.write("\n");
        }
        writer.close();
    }

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + LDA.class.getName() + " -help";
    }

    public static void run(String[] args) throws Exception {
        // create the command line parser
        parser = new BasicParser();

        // create the Options
        options = new Options();

        // directories
        addOption("dataset", "Dataset");
        addOption("output", "Output folder");
        addOption("data-folder", "Processed data folder");
        addOption("format-folder", "Folder holding formatted data");
        addOption("format-file", "Format file name");

        // sampling configurations
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");

        // model parameters
        addOption("K", "Number of topics");
        addOption("numTopwords", "Number of top words per topic");

        // model hyperparameters
        addOption("alpha", "Hyperparameter of the symmetric Dirichlet prior "
                + "for topic distributions");
        addOption("beta", "Hyperparameter of the symmetric Dirichlet prior "
                + "for word distributions");

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

        // data 
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon-data");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "demo");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
        String outputFolder = CLIUtils.getStringArgument(cmd, "output", "demo/"
                + datasetName + "/" + formatFolder + "-model");
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        // sampler
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 25);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 50);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");
        verbose = true;
        debug = true;

        if (verbose) {
            System.out.println("Loading data ...");
        }
        TextDataset dataset = new TextDataset(datasetName, datasetFolder);
        dataset.setFormatFilename(formatFile);
        dataset.loadFormattedData(new File(dataset.getDatasetFolderPath(), formatFolder));
        dataset.prepareTopicCoherence(numTopWords);

        int V = dataset.getWordVocab().size();
        InitialState initState = InitialState.RANDOM;

        if (verbose) {
            System.out.println("Running LDA ...");
        }
        LDA sampler = new LDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(dataset.getWordVocab());
        sampler.setPrefix("prior_");

        sampler.configure(outputFolder, dataset.getWords(),
                V, K, alpha, beta, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File ldaFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(ldaFolder);

        double ratio = 100;
        double prob = 1.0 / (K - 1 + ratio);
        double[][] docTopicPrior = new double[dataset.getWords().length][K];
        for (int d = 0; d < dataset.getWords().length; d++) {
            docTopicPrior[d][0] = ratio * prob;
            for (int k = 1; k < K; k++) {
                docTopicPrior[d][k] = prob;
            }
        }

        sampler.initialize(docTopicPrior, null);
        sampler.iterate();
        sampler.outputTopicTopWords(new File(ldaFolder, TopWordFile), numTopWords);
        sampler.outputTopicCoherence(new File(ldaFolder, TopicCoherenceFile), dataset.getTopicCoherence());
        sampler.outputDocTopicDistributions(new File(ldaFolder, "doc-topic.txt"));
        sampler.outputTopicWordDistributions(new File(ldaFolder, "topic-word.txt"));
    }
}
