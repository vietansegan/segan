package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractExperiment;
import core.AbstractSampler;
import data.LabelTextDataset;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import optimization.RidgeLinearRegressionOptimizable;
import optimization.RidgeLogisticRegressionOptimizable;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;
import sampling.likelihood.CascadeDirMult.PathAssumption;

/**
 *
 * @author vietan
 */
public class SNLDA extends AbstractSampler {

    public static final int POSITVE = 1;
    public static final int NEGATIVE = -1;
    // hyperparameters for fixed-height tree
    protected double[] alphas;          // [L-1]
    protected double[] betas;           // [L]
    protected double[] gamma_means;     // [L-1] mean of bias coins
    protected double[] gamma_scales;    // [L-1] scale of bias coins
    protected double rho;
    protected double mu;
    protected double[] sigmas;

    // inputs
    protected int[][] words; // all words
    protected double[] responses; // [D] continous responses
    protected int[] labels; // [D] binary responses
    protected ArrayList<Integer> docIndices; // indices of docs under consideration
    protected int V;    // vocabulary size
    protected int[] Ks; // [L-1]: number of children per node at each level
    protected PathAssumption path;
    // derived
    protected int D; // number of documents
    protected int L;
    // latent
    Node[][] z;
    Node root;
    // internal
    private int numTokensAccepted;
    private double[] background;
    private double[] docMeans;
    private boolean isBinary;
    private Set<Integer> positives;
    private double uniform;

    public SNLDA() {
        this.basename = "SNLDA";
    }

    public SNLDA(String bname) {
        this.basename = bname;
    }

    public void configure(SNLDA sampler) {
        this.isBinary = sampler.isBinary;
        if (this.isBinary) {
            this.configureBinary(sampler.folder,
                    sampler.V,
                    sampler.Ks,
                    sampler.alphas,
                    sampler.betas,
                    sampler.gamma_means,
                    sampler.gamma_scales,
                    sampler.mu,
                    sampler.sigmas,
                    sampler.initState,
                    sampler.path,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        } else {
            this.configureContinuous(sampler.folder,
                    sampler.V,
                    sampler.Ks,
                    sampler.alphas,
                    sampler.betas,
                    sampler.gamma_means,
                    sampler.gamma_scales,
                    sampler.rho,
                    sampler.mu,
                    sampler.sigmas,
                    sampler.initState,
                    sampler.path,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        }
    }

    public void configureBinary(String folder,
            int V, int[] Ks,
            double[] alphas,
            double[] betas,
            double[] gamma_means,
            double[] gamma_scales,
            double mu,
            double[] sigmas,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / this.V;
        this.Ks = Ks;
        this.L = this.Ks.length + 1;

        this.alphas = alphas;
        this.betas = betas;
        this.gamma_means = gamma_means;
        this.gamma_scales = gamma_scales;
        this.mu = mu;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.path = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.isBinary = true;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(gamma_means));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gamma_scales));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
        }

        validateInputHyperparameters();
    }

    public void configureContinuous(String folder,
            int V, int[] Ks,
            double[] alphas,
            double[] betas,
            double[] gamma_means,
            double[] gamma_scales,
            double rho,
            double mu,
            double[] sigmas,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / this.V;
        this.Ks = Ks;
        this.L = this.Ks.length + 1;

        this.alphas = alphas;
        this.betas = betas;
        this.gamma_means = gamma_means;
        this.gamma_scales = gamma_scales;
        this.rho = rho;
        this.mu = mu;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.path = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.isBinary = false;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(gamma_means));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gamma_scales));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
        }

        validateInputHyperparameters();
    }

    private void validateInputHyperparameters() {
        if (alphas.length != L - 1) {
            throw new MismatchRuntimeException(alphas.length, L - 1);
        }
        if (betas.length != L) {
            throw new MismatchRuntimeException(betas.length, L);
        }
        if (gamma_means.length != L - 1) {
            throw new MismatchRuntimeException(gamma_means.length, L - 1);
        }
        if (gamma_scales.length != L - 1) {
            throw new MismatchRuntimeException(gamma_scales.length, L - 1);
        }
        if (sigmas.length != L - 1) {
            throw new MismatchRuntimeException(sigmas.length, L - 1);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename);
        str.append("_Ks");
        for (int K : Ks) {
            str.append("-").append(K);
        }
        str.append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG);
        str.append("_a");
        for (double la : alphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_gm");
        for (double gm : gamma_means) {
            str.append("-").append(MiscUtils.formatDouble(gm));
        }
        str.append("_gs");
        for (double gs : gamma_scales) {
            str.append("-").append(MiscUtils.formatDouble(gs));
        }
        str.append("_r-").append(MiscUtils.formatDouble(rho));
        str.append("_m-").append(MiscUtils.formatDouble(mu));
        str.append("_s");
        for (double s : sigmas) {
            str.append("-").append(MiscUtils.formatDouble(s));
        }
        str.append("_opt-").append(this.paramOptimized);
        str.append("_bin-").append(this.isBinary);
        str.append("_path-").append(this.path);
        this.name = str.toString();
    }

    protected double getAlpha(int l) {
        return this.alphas[l];
    }

    protected double getBeta(int l) {
        return this.betas[l];
    }

    protected double getGammaMean(int l) {
        return this.gamma_means[l];
    }

    protected double getGammaScale(int l) {
        return this.gamma_scales[l];
    }

    protected double getSigma(int l) {
        return this.sigmas[l - 1];
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath() + "\n"
                + printGlobalTreeSummary() + "\n";
    }

    public boolean isLeafNode(int level) {
        return level == L - 1;
    }

    public double[] getPredictedValues() {
        return docMeans;
    }

    /**
     * Set up training data with continuous responses.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     * @param docResponses Continuous responses
     */
    public void train(int[][] docWords,
            ArrayList<Integer> docIndices,
            double[] docResponses) {
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.numTokens = 0;
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.responses = new double[D];
        this.background = new double[V];
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = docWords[dd];
            this.responses[ii] = docResponses[dd];
            this.numTokens += this.words[ii].length;
            for (int nn = 0; nn < words[ii].length; nn++) {
                this.background[words[ii][nn]]++;
            }
        }
        for (int vv = 0; vv < V; vv++) {
            this.background[vv] /= this.numTokens;
        }

        if (verbose) {
            logln("--- # all docs:\t" + words.length);
            logln("--- # selected docs:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- responses:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatUtils.standardDeviation(responses)));
            int[] histogram = StatUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }
        }
    }

    /**
     * Set up training data with binary responses.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     * @param docLabels Binary labels
     */
    public void train(int[][] docWords,
            ArrayList<Integer> docIndices,
            int[] docLabels) {
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.numTokens = 0;
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.labels = new int[D];
        this.positives = new HashSet<Integer>();
        this.background = new double[V];
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = docWords[dd];
            this.labels[ii] = docLabels[dd];
            if (this.labels[ii] == POSITVE) {
                this.positives.add(ii);
            }
            this.numTokens += this.words[ii].length;
            for (int nn = 0; nn < this.words[ii].length; nn++) {
                this.background[words[ii][nn]]++;
            }
        }
        for (int vv = 0; vv < V; vv++) {
            this.background[vv] /= this.numTokens;
        }

        if (verbose) {
            logln("--- # all docs:\t" + words.length);
            logln("--- # selected docs:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- responses:");
            int posCount = this.positives.size();
            logln("--- --- # postive: " + posCount + " (" + ((double) posCount / D) + ")");
            logln("--- --- # negative: " + (D - posCount));
        }
    }

    /**
     * Set up test data.
     *
     * @param docWords Test documents
     * @param docIndices Indices of test documents
     * @param stateFile Input file storing trained model
     * @param testStateFile Output file to store assignments
     * @param predictionFile Output file to store predictions at different test
     * iterations using the given trained model
     * @return Prediction on all documents using the given model
     */
    public double[] test(int[][] docWords, ArrayList<Integer> docIndices,
            File stateFile,
            File testStateFile,
            File predictionFile) {
        setTestConfigurations(BURN_IN / 2, MAX_ITER / 2, LAG / 2);
        // input stored model
        if (stateFile == null) {
            stateFile = getFinalStateFile();
        }
        inputModel(stateFile.toString());

        // setup data
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.numTokens = 0;
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = docWords[dd];
            this.numTokens += this.words[ii].length;
        }

        // initialize data
        initializeDataStructure();

        // store predictions at different test iterations
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();

        // sample topic assignments for test document
        for (iter = 0; iter < this.testMaxIter; iter++) {
            numTokensChanged = 0;
            numTokensAccepted = 0;
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter
                        + ". current thread: " + Thread.currentThread().getId();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            long topicTime;
            if (iter == 0) {
                topicTime = sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);
            } else {
                topicTime = sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
            }

            if (isReporting) {
                logln("--- --- Time (s). sample topic: " + topicTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ") "
                        + ". # token accepted: " + numTokensAccepted
                        + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ") "
                        + "\n");
            }

            // store prediction (on all documents) at a test iteration
            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                double[] predResponses = new double[D];
                System.arraycopy(docMeans, 0, predResponses, 0, D);
                predResponsesList.add(predResponses);
            }
        }

        // output state file containing the assignments for test documents
        if (testStateFile != null) {
            outputState(testStateFile);
        }

        // store predictions if necessary
        if (predictionFile != null) {
            PredictionUtils.outputSingleModelRegressions(predictionFile, predResponsesList);
        }

        // average over all stored predictions
        double[] predictions = new double[D];
        for (int dd = 0; dd < D; dd++) {
            for (double[] predResponses : predResponsesList) {
                predictions[dd] += predResponses[dd] / predResponsesList.size();
            }
        }
        return predictions;
    }

    @Override
    public void initialize() {
        initialize(null);
    }

    public void initialize(double[][] priorTopics) {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;
        initializeModelStructure(priorTopics);
        initializeDataStructure();
        initializeAssignments();
        updateTopics();
        updateEtas();

        if (verbose) {
            logln("--- Done initializing.\n" + printGlobalTree());
            logln("\n" + printGlobalTreeSummary() + "\n");
            getLogLikelihood();
        }

        outputTopicTopWords(new File(getSamplerFolderPath(), "init-" + TopWordFile), 20);
        validate("Initialized");
    }

    protected void initializeModelStructure(double[][] priorTopics) {
        if (verbose) {
            logln("--- Initializing model structure using Recursive LDA ...");
        }

        double ldaAlpha = this.alphas[0];
        double ldaBeta = this.betas[0];
        LDA lda = runLDA(words, Ks[0], V, null, null, ldaAlpha, ldaBeta, 100, 250, 30);

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, background);
        this.root = new Node(iter, 0, 0, rootTopic, null, 0.0);

        Queue<Node> queue = new LinkedList<>();
        for (int kk = 0; kk < Ks[0]; kk++) {
            DirMult topic = new DirMult(V, getBeta(1) * V, lda.getTopicWords()[kk].getDistribution());
            Node node = new Node(iter, kk, 1, topic, root, SamplerUtils.getGaussian(mu, sigmas[0]));
            this.root.addChild(kk, node);

            queue.add(node);
        }
        while (!queue.isEmpty()) {
            Node node = queue.poll();

            int level = node.getLevel();
            if (level < L - 1) {
                for (int kk = 0; kk < Ks[level]; kk++) {
                    DirMult topic = new DirMult(V, getBeta(level + 1) * V, 1.0 / V);
                    Node child = new Node(iter, kk, level + 1, topic, node, SamplerUtils.getGaussian(mu, sigmas[level]));
                    node.addChild(kk, child);
                    queue.add(child);
                }
            }
        }

        // initialize pi's and theta's
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getLevel() < L - 1) {
                node.initializeGlobalTheta();
                node.initializeGlobalPi();
                for (Node child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        if (verbose) {
            logln("--- --- Initialized model structure.\n" + printGlobalTreeSummary());
        }
    }
    
    protected void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }
        this.z = new Node[D][];
        for (int dd = 0; dd < D; dd++) {
            this.z[dd] = new Node[words[dd].length];
        }
        this.docMeans = new double[D];
    }

    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments. " + initState);
        }
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
    }

    /**
     * Add a token to a node.
     *
     * @param dd
     * @param nn
     * @param node
     * @param addToData
     * @param addToModel
     */
    private void addToken(int dd, int nn, Node node,
            boolean addToData, boolean addToModel) {
        if (addToModel) {
            node.getContent().increment(words[dd][nn]);
        }
        if (addToData) {
            docMeans[dd] += node.eta / this.words[dd].length;
            node.tokenCounts.increment(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeTokenCounts.increment(dd);
                tempNode = tempNode.getParent();
            }
        }
    }

    /**
     * Remove a token from a node.
     *
     * @param dd
     * @param nn
     * @param node
     * @param removeFromData
     * @param removeFromModel
     */
    private void removeToken(int dd, int nn, Node node,
            boolean removeFromData, boolean removeFromModel) {
        if (removeFromData) {
            docMeans[dd] -= node.eta / this.words[dd].length;
            node.tokenCounts.decrement(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeTokenCounts.decrement(dd);
                tempNode = tempNode.getParent();
            }
        }
        if (removeFromModel) {
            node.getContent().decrement(words[dd][nn]);
        }
    }

    @Override
    public void iterate() {
        if (isReporting) {
            System.out.println("\n");
            logln("Iteration " + iter + " / " + MAX_ITER);
        }

        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);
        updateTopics();
        updateEtas();
    }

    /**
     * Sample node assignment for all tokens.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observed
     * @return Elapsed time
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData, boolean observed) {
        if (isReporting) {
            logln("+++ Sampling Zs ...");
        }
        numTokensChanged = 0;
        numTokensAccepted = 0;

        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                Node sampledNode = sampleNode(dd, nn, root);
                boolean accept = false;
                if (z[dd][nn] == null) {
                    accept = true;
                    numTokensChanged ++;
                    numTokensAccepted++;
                } else if (sampledNode.equals(z[dd][nn])) {
                    accept = true;
                    numTokensAccepted++;
                } else {
                    double[] curLogprobs = getLogProbabilities(dd, nn, z[dd][nn], observed);
                    double[] newLogprobs = getLogProbabilities(dd, nn, sampledNode, observed);
                    double ratio = Math.min(1.0,
                            Math.exp(newLogprobs[ACTUAL_INDEX] - curLogprobs[ACTUAL_INDEX]
                                    + curLogprobs[PROPOSAL_INDEX] - newLogprobs[PROPOSAL_INDEX]));
                    if (rand.nextDouble() < ratio) {
                        accept = true;
                        numTokensAccepted++;
                    }
                }

                if (accept) {
                    if (z[dd][nn] != null && !z[dd][nn].equals(sampledNode)) {
                        numTokensChanged++;
                    }
                    z[dd][nn] = sampledNode;
                }

                // add
                addToken(dd, nn, z[dd][nn], addToData, addToModel);
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # changed: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                    + ". # accepted: " + numTokensAccepted
                    + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ")");
        }
        return eTime;
    }

    /**
     * Recursively sample a node from a current node. The sampled node can be
     * either the same node or one of its children. If the current node is a
     * leaf node, return it.
     *
     * @param dd Document index
     * @param nn Token index
     * @param curNode Current node
     */
    private Node sampleNode(int dd, int nn, Node curNode) {
        if (curNode.isLeaf()) {
            return curNode;
        }
        int level = curNode.getLevel();
        double lAlpha = getAlpha(level);
        double gammaScale = getGammaScale(level);
        double stayprob = (curNode.tokenCounts.getCount(dd) + gammaScale * curNode.pi)
                / (curNode.subtreeTokenCounts.getCount(dd) + gammaScale);
        double passprob = 1.0 - stayprob;

        int KK = curNode.getNumChildren();
        double[] probs = new double[KK + 1];
        double norm = curNode.getPassingCount(dd) + lAlpha * KK;
        for (Node child : curNode.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * KK * curNode.theta[kk]) / norm;
            double wordprob = child.getPhi(words[dd][nn]);
            probs[kk] = passprob * pathprob * wordprob;
        }
        double wordprob = curNode.getPhi(words[dd][nn]);
        probs[KK] = stayprob * wordprob;

        int sampledIdx = SamplerUtils.scaleSample(probs);
        if (sampledIdx == KK) {
            return curNode;
        } else {
            return sampleNode(dd, nn, curNode.getChild(sampledIdx));
        }
    }

    /**
     * Compute both the proposal log probabilities and the actual log
     * probabilities of assigning a token to a node.
     *
     * @param dd Document index
     * @param nn Token index
     * @param observed
     * @param node The node to be assigned to
     */
    private double[] getLogProbabilities(int dd, int nn, Node node, boolean observed) {
        double[] logprobs = getTransLogProbabilities(dd, nn, node, node);
        logprobs[ACTUAL_INDEX] = Math.log(node.getPhi(words[dd][nn]));
        if (observed) {
            logprobs[ACTUAL_INDEX] += getResponseLogLikelihood(dd, node);
        }
        Node source = node.getParent();
        Node target = node;
        while (source != null) {
            double[] lps = getTransLogProbabilities(dd, nn, source, target);
            logprobs[PROPOSAL_INDEX] += lps[PROPOSAL_INDEX];
            logprobs[ACTUAL_INDEX] += lps[ACTUAL_INDEX];

            source = source.getParent();
            target = target.getParent();
        }
        return logprobs;
    }

    /**
     * Compute the log probabilities of (1) the proposal move and (2) the actual
     * move from source to target. The source and target nodes can be the same.
     *
     * @param dd Document index
     * @param nn Token index
     * @param source The source node
     * @param target The target node
     */
    private double[] getTransLogProbabilities(int dd, int nn, Node source, Node target) {
        int level = source.getLevel();
        if (level == L - 1) { // leaf node
            if (!source.equals(target)) {
                throw new RuntimeException("At leaf node. " + source.toString()
                        + ". " + target.toString());
            }
            return new double[2]; // stay with probabilities 1
        }

        int KK = source.getNumChildren();
        double lAlpha = getAlpha(level);
        double gammaScale = getGammaScale(level);
        double stayprob = (source.tokenCounts.getCount(dd) + gammaScale * source.pi)
                / (source.subtreeTokenCounts.getCount(dd) + gammaScale);
        double passprob = 1.0 - stayprob;

        double pNum = 0.0;
        double pDen = 0.0;
        double aNum = 0.0;
        double aDen = 0.0;
        double norm = source.subtreeTokenCounts.getCount(dd)
                - source.tokenCounts.getCount(dd) + lAlpha * KK;
        for (Node child : source.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * KK * source.theta[kk]) / norm;
            double wordprob = child.getPhi(words[dd][nn]);

            double aVal = passprob * pathprob;
            aDen += aVal;

            double pVal = passprob * pathprob * wordprob;
            pDen += pVal;

            if (target.equals(child)) {
                pNum = pVal;
                aNum = aVal;
            }
        }
        double wordprob = source.getPhi(words[dd][nn]);
        double pVal = stayprob * wordprob;
        pDen += pVal;
        aDen += stayprob;

        if (target.equals(source)) {
            pNum = pVal;
            aNum = stayprob;
        }

        double[] lps = new double[2];
        lps[PROPOSAL_INDEX] = Math.log(pNum / pDen);
        lps[ACTUAL_INDEX] = Math.log(aNum / aDen);
        return lps;
    }

    private double getResponseLogLikelihood(int dd, Node node) {
        double aMean = docMeans[dd] + node.eta / this.words[dd].length;
        double resLLh;
        if (isBinary) {
            resLLh = getLabelLogLikelihood(labels[dd], aMean);
        } else {
            resLLh = StatUtils.logNormalProbability(responses[dd], aMean, Math.sqrt(rho));
        }
        return resLLh;
    }

    private double getLabelLogLikelihood(int label, double dotProb) {
        double logNorm = Math.log(Math.exp(dotProb) + 1);
        if (label == POSITVE) {
            return dotProb - logNorm;
        } else {
            return -logNorm;
        }
    }

    protected long updateTopics() {
        if (isReporting) {
            logln("+++ Updating topics ...");
        }
        long sTime = System.currentTimeMillis();

        // get all leaves of the tree
        ArrayList<Node> leaves = new ArrayList<Node>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getChildren().isEmpty()) {
                leaves.add(node);
            }
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // bottom-up smoothing to compute pseudo-counts from children
        Queue<Node> queue = new LinkedList<Node>();
        for (Node leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            Node parent = node.getParent();
            if (!node.isRoot() && !queue.contains(parent)) {
                queue.add(parent);
            }
            if (node.isLeaf()) {
                continue;
            }
            node.computePropagatedCountsFromChildren();
        }

        // top-down sampling to get topics
        queue = new LinkedList<Node>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            for (Node child : node.getChildren()) {
                queue.add(child);
            }
            node.updateTopic();
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    /**
     * Update regression parameters using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        if (isReporting) {
            logln("+++ Updating eta's ...");
        }
        long sTime = System.currentTimeMillis();

        // list of nodes
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        // design matrix
        SparseVector[] designMatrix = new SparseVector[D];
        for (int aa = 0; aa < D; aa++) {
            designMatrix[aa] = new SparseVector(N);
        }
        for (int kk = 0; kk < N; kk++) {
            Node node = nodeList.get(kk);
            for (int dd : node.tokenCounts.getIndices()) {
                int count = node.tokenCounts.getCount(dd);
                double val = (double) count / this.words[dd].length;
                designMatrix[dd].change(kk, val);
            }
        }

        // current params
        double[] etaArray = new double[N];
        double[] sigmaArray = new double[N];
        for (int kk = 0; kk < N; kk++) {
            etaArray[kk] = nodeList.get(kk).eta;
            sigmaArray[kk] = getSigma(nodeList.get(kk).getLevel());
        }

        boolean converged = false;
        if (isBinary) {
            RidgeLogisticRegressionOptimizable optimizable = new RidgeLogisticRegressionOptimizable(
                    labels, etaArray, designMatrix, mu, sigmaArray);
            LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            // update regression parameters
            for (int kk = 0; kk < N; kk++) {
                nodeList.get(kk).eta = optimizable.getParameter(kk);
            }
        } else {
            RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                    responses, etaArray, designMatrix, rho, mu, sigmaArray);
            LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            // update regression parameters
            for (int kk = 0; kk < N; kk++) {
                nodeList.get(kk).eta = optimizable.getParameter(kk);
            }
        }

        // update document means
        for (int dd = 0; dd < D; dd++) {
            docMeans[dd] = 0.0;
            for (int kk : designMatrix[dd].getIndices()) {
                docMeans[dd] += designMatrix[dd].get(kk) * nodeList.get(kk).eta;
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
            logln("--- --- time: " + eTime);
            evaluatePerformances();
        }
        return eTime;
    }

    protected void evaluatePerformances() {
        if (isBinary) {
        } else {
            RegressionEvaluation eval = new RegressionEvaluation(responses, docMeans);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeMeanAbsoluteError();
            eval.computeRSquared();
            eval.computePredictiveRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
            }
        }
    }

    private ArrayList<Node> getNodeList() {
        ArrayList<Node> nodeList = new ArrayList<>();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            if (!node.isRoot()) {
                nodeList.add(node);
            }
        }
        return nodeList;
    }

    @Override
    public double getLogLikelihood() {
        return 0.0;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg);
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.validate(msg);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        // model string
        StringBuilder modelStr = new StringBuilder();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            modelStr.append(Integer.toString(node.born)).append("\n");
            modelStr.append(node.getPathString()).append("\n");
            modelStr.append(node.eta).append("\n");
            modelStr.append(node.pi).append("\n");
            modelStr.append(SparseCount.output(node.tokenCounts)).append("\n");
            modelStr.append(SparseCount.output(node.subtreeTokenCounts)).append("\n");
            if (node.theta != null) {
                modelStr.append(MiscUtils.arrayToString(node.theta));
            }
            modelStr.append("\n");
            modelStr.append(DirMult.output(node.getContent())).append("\n");
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // assignment string
        StringBuilder assignStr = new StringBuilder();
        for (int dd = 0; dd < z.length; dd++) {
            for (int nn = 0; nn < z[dd].length; nn++) {
                assignStr.append(dd)
                        .append("\t").append(nn)
                        .append("\t").append(z[dd][nn].getPathString()).append("\n");
            }
        }

        try { // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);

            this.outputZipFile(filepath, contentStrs, entryFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + filepath);
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
            throw new RuntimeException("Exception while inputing from " + filepath);
        }
    }

    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            HashMap<String, Node> nodeMap = new HashMap<String, Node>();
            String line;
            while ((line = reader.readLine()) != null) {
                int born = Integer.parseInt(line);
                String pathStr = reader.readLine();
                double eta = Double.parseDouble(reader.readLine());
                double pi = Double.parseDouble(reader.readLine());
                SparseCount tokenCounts = SparseCount.input(reader.readLine());
                SparseCount subtreeTokenCounts = SparseCount.input(reader.readLine());
                line = reader.readLine().trim();
                double[] theta = null;
                if (!line.isEmpty()) {
                    theta = MiscUtils.stringToDoubleArray(line);
                }
                DirMult topic = DirMult.input(reader.readLine());

                // create node
                int lastColonIndex = pathStr.lastIndexOf(":");
                Node parent = null;
                if (lastColonIndex != -1) {
                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
                }
                String[] pathIndices = pathStr.split(":");
                int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
                int nodeLevel = pathIndices.length - 1;

                Node node = new Node(born, nodeIndex, nodeLevel, topic, parent, eta);
                node.pi = pi;
                node.theta = theta;
//                node.tokenCounts = tokenCounts;
//                node.subtreeTokenCounts = subtreeTokenCounts;
                node.setPhi(topic.getDistribution());

                if (node.getLevel() == 0) {
                    root = node;
                }
                if (parent != null) {
                    parent.addChild(node.getIndex(), node);
                }
                nodeMap.put(pathStr, node);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    /**
     * Input a set of assignments.
     *
     * @param zipFilepath Compressed learned state file
     */
    public void inputAssignments(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }
        try {
            z = new Node[D][];
            for (int d = 0; d < D; d++) {
                z[d] = new Node[words[d].length];
            }

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int dd = 0; dd < z.length; dd++) {
                for (int nn = 0; nn < z[dd].length; nn++) {
                    String[] sline = reader.readLine().split("\t");
                    if (dd != Integer.parseInt(sline[0])) {
                        throw new MismatchRuntimeException(Integer.parseInt(sline[0]), dd);
                    }
                    if (nn != Integer.parseInt(sline[1])) {
                        throw new MismatchRuntimeException(Integer.parseInt(sline[1]), nn);
                    }
                    String pathStr = sline[2];
                    z[dd][nn] = getNode(pathStr);
                    addToken(dd, nn, z[dd][nn], ADD, !ADD);
                }
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Parse the node path string.
     *
     * @param nodePath The node path string
     * @return
     */
    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    /**
     * Get a node in the tree given a parsed path
     *
     * @param parsedPath The parsed path
     */
    private Node getNode(int[] parsedPath) {
        Node node = root;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    private Node getNode(String pathStr) {
        return getNode(parseNodePath(pathStr));
    }

    /**
     * Summary of the current tree.
     *
     * @return Summary of the current tree
     */
    public String printGlobalTreeSummary() {
        StringBuilder str = new StringBuilder();
        SparseCount nodeCountPerLevel = new SparseCount();
        SparseCount obsCountPerLevel = new SparseCount();
        SparseCount subtreeObsCountPerLvl = new SparseCount();

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        int totalObs = 0;
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            int level = node.getLevel();

            nodeCountPerLevel.increment(level);
            obsCountPerLevel.changeCount(level, node.getContent().getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeTokenCounts.getCountSum());

            totalObs += node.getContent().getCountSum();

            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("global tree:\n\t>>> node count per level:\n");
        for (int l : nodeCountPerLevel.getSortedIndices()) {
            int obsCount = obsCountPerLevel.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLevel.getCount(l);
            str.append("\t>>> >>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * obsCount / numTokens)).append("%")
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * subtreeObsCount / numTokens)).append("%")
                    .append("]\n");
        }
        str.append("\n");
        str.append("\t>>> # observations = ").append(totalObs).append("\n");
        str.append("\t>>> # nodes = ").append(nodeCountPerLevel.getCountSum()).append("\n");
        return str.toString();
    }

    /**
     * The current tree.
     *
     * @return The current tree
     */
    public String printGlobalTree() {
        SparseCount nodeCountPerLvl = new SparseCount();
        SparseCount obsCountPerLvl = new SparseCount();
        SparseCount subtreeObsCountPerLvl = new SparseCount();
        int totalNumObs = 0;

        StringBuilder str = new StringBuilder();
        str.append("global tree\n");

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            int level = node.getLevel();

            nodeCountPerLvl.increment(level);
            obsCountPerLvl.changeCount(level, node.getContent().getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeTokenCounts.getCountSum());

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString()).append("\n");

            // top words according to distribution
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            String[] topWords = node.getTopWords(10);
            for (String w : topWords) {
                str.append(w).append(" ");
            }
            str.append("\n");

            // top assigned words
            if (!node.getContent().isEmpty()) {
                for (int i = 0; i < node.getLevel(); i++) {
                    str.append("\t");
                }
                str.append(node.getTopObservations()).append("\n");
            }
            str.append("\n");

            totalNumObs += node.getContent().getCountSum();

        }
        str.append("Tree summary").append("\n");
        for (int l : nodeCountPerLvl.getSortedIndices()) {
            int obsCount = obsCountPerLvl.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLvl.getCount(l);
            str.append("\t>>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * obsCount / numTokens)).append("%")
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * subtreeObsCount / numTokens)).append("%")
                    .append("]\n");
        }
        str.append("\t>>> # observations = ").append(totalNumObs).append("\n");
        str.append("\t>>> # nodes = ").append(nodeCountPerLvl.getCountSum()).append("\n");
        return str.toString();
    }

    /**
     * Output top words for each topic in the tree to text file.
     *
     * @param outputFile The output file
     * @param numWords Number of top words
     */
    @Override
    public void outputTopicTopWords(File outputFile, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();

            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            double[] nodeTopic = node.phi;
            String[] topWords = getTopWords(nodeTopic, numWords);

            // top words according to the distribution
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.born)
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.eta))
                    .append(")");
            str.append("\n");

            // words with highest probabilities
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            for (String topWord : topWords) {
                str.append(topWord).append(" ");
            }
            str.append("\n");

            // top assigned words
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getTopObservations()).append("\n\n");
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(str.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topics "
                    + outputFile);
        }
    }

    /**
     * Output posterior distribution over non-rooted nodes in the tree of all
     * documents.
     *
     * @param outputFile Output file
     */
    public void outputNodePosteriors(File outputFile) {
        ArrayList<Node> nodeList = getNodeList();
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int dd = 0; dd < D; dd++) {
                double[] nodePos = new double[nodeList.size()];
                for (int kk = 0; kk < nodeList.size(); kk++) {
                    Node node = nodeList.get(kk);
                    nodePos[kk] = (double) node.tokenCounts.getCount(dd) / words[dd].length;
                }
                writer.write(Integer.toString(dd));
                for (int kk = 0; kk < nodePos.length; kk++) {
                    writer.write("\t" + nodePos[kk]);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while output to " + outputFile);
        }
    }

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected SparseCount subtreeTokenCounts;
        protected SparseCount tokenCounts;
        protected double[] theta;
        protected double pi;
        protected double eta;
        protected SparseCount propagatedCounts;

        // estimated topics after training, which is used for test
        protected double[] phi;

        public Node(int iter, int index, int level, DirMult content, Node parent,
                double eta) {
            super(index, level, content, parent);
            this.born = iter;
            this.subtreeTokenCounts = new SparseCount();
            this.tokenCounts = new SparseCount();
            this.eta = eta;
            this.propagatedCounts = new SparseCount();
        }

        void initializeGlobalPi() {
            this.pi = getGammaMean(level);
        }

        void initializeGlobalTheta() {
            int KK = getNumChildren();
            this.theta = new double[KK];
            Arrays.fill(this.theta, 1.0 / KK);
        }

        void setPhi(double[] ph) {
            this.phi = ph;
        }

        /**
         * Return the number of tokens of a given document which are assigned to
         * any nodes below this node.
         *
         * @param dd Document index
         */
        int getPassingCount(int dd) {
            return subtreeTokenCounts.getCount(dd) - tokenCounts.getCount(dd);
        }

        /**
         * Get the probability of a word type given this node. During training,
         * this probability is computed on-the-fly using counts and
         * pseudo-counts. During test, it comes from the learned distribution.
         *
         * @param v word type
         */
        double getPhi(int v) {
            if (this.phi == null) {
                return getContent().getProbability(v);
            }
            return phi[v];
        }

        void updateTopic() {
            phi = new double[V];
            double beta = getBeta(getLevel());
            double[] meanPrior = this.content.getCenterVector();

            double norm = this.getContent().getCountSum()
                    + this.propagatedCounts.getCountSum()
                    + beta * V;
            for (int vv = 0; vv < V; vv++) {
                phi[vv] = (getContent().getCount(vv)
                        + propagatedCounts.getCount(vv)
                        + beta * V * meanPrior[vv]) / norm;
            }
        }

        void computePropagatedCountsFromChildren() {
            this.propagatedCounts = new SparseCount();
            for (Node child : this.getChildren()) {
                SparseCount childCount = child.getContent().getSparseCounts();
                childCount.add(child.propagatedCounts);
                for (int vv : childCount.getIndices()) {
                    if (path == PathAssumption.MINIMAL) {
                        this.propagatedCounts.increment(vv);
                    } else if (path == PathAssumption.MAXIMAL) {
                        this.propagatedCounts.changeCount(vv, childCount.getCount(vv));
                    } else if (path == PathAssumption.NONE) {
                        // do nothing, no propagation
                    } else {
                        throw new RuntimeException("Path assumption " + path
                                + " is not supported.");
                    }
                }
            }
        }

        boolean isEmpty() {
            return this.getContent().isEmpty();
        }

        String[] getTopWords(int numTopWords) {
            ArrayList<RankingItem<String>> topicSortedVocab
                    = IOUtils.getSortedVocab(phi, wordVocab);
            String[] topWords = new String[numTopWords];
            for (int i = 0; i < numTopWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
        }

        String getTopObservations() {
            return getTopObservations(getContent().getSparseCounts());
        }

        String getTopObservations(SparseCount counts) {
            ArrayList<RankingItem<Integer>> rankObs = new ArrayList<RankingItem<Integer>>();
            for (int obs : counts.getIndices()) {
                rankObs.add(new RankingItem<Integer>(obs, counts.getCount(obs)));
            }
            Collections.sort(rankObs);
            StringBuilder str = new StringBuilder();
            for (int ii = 0; ii < Math.min(10, rankObs.size()); ii++) {
                RankingItem<Integer> obs = rankObs.get(ii);
                str.append(wordVocab.get(obs.getObject())).append(":")
                        .append(obs.getPrimaryValue()).append(" ");
            }
            return str.toString();
        }

        void validate(String msg) {
            this.tokenCounts.validate(msg);
            this.subtreeTokenCounts.validate(msg);
            if (theta != null && theta.length != getNumChildren()) {
                throw new RuntimeException(msg + ". MISMATCH. " + this.toString());
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[").append(getPathString());
            str.append(", ").append(born);
            str.append(", c (").append(getChildren().size()).append(")");
            // word types
            str.append(", (").append(getContent().getCountSum()).append(")");
            // token counts
            str.append(", (").append(subtreeTokenCounts.getCountSum());
            str.append(", ").append(tokenCounts.getCountSum()).append(")");
            str.append(", ").append(MiscUtils.formatDouble(eta));
            str.append("]");
            return str.toString();
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + SNLDA.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        String example = new String();

        example += "For continuous responses:\n";
        example += "java -cp \"dist/segan.jar:lib/*\" sampler.supervised.regression.SNLDA "
                + "--dataset amazon-data "
                + "--word-voc-file demo/amazon-data/format-supervised/amazon-data.wvoc "
                + "--word-file demo/amazon-data/format-supervised/amazon-data.dat "
                + "--info-file demo/amazon-data/format-supervised/amazon-data.docinfo "
                + "--output-folder demo/amazon-data/model-supervised  "
                + "--Ks 15,4 "
                + "--burnIn 50 "
                + "--maxIter 100 "
                + "--sampleLag 25 "
                + "--report 5 "
                + "--init random "
                + "--alphas 0.1,0.1 "
                + "--betas 1.0,0.5,0.1 "
                + "--gamma-means 0.2,0.2 "
                + "--gamma-scales 100,10 "
                + "--rho 1.0 "
                + "--mu 0.0 "
                + "--sigma 2.5 "
                + "-v -d -z -train";
        example += "\n\n";
        example += "For binary responses:\n";
        example += "java -cp \"dist/segan.jar:lib/*\" sampler.supervised.regression.SNLDA "
                + "--dataset amazon-data "
                + "--word-voc-file demo/amazon-data/format-binary/amazon-data.wvoc "
                + "--word-file demo/amazon-data/format-binary/amazon-data.dat "
                + "--info-file demo/amazon-data/format-binary/amazon-data.docinfo "
                + "--output-folder demo/amazon-data/model-binary "
                + "--Ks 15,4 "
                + "--burnIn 50 "
                + "--maxIter 100 "
                + "--sampleLag 25 "
                + "--report 5 "
                + "--init random "
                + "--alphas 0.1,0.1 "
                + "--betas 1.0,0.5,0.1 "
                + "--gamma-means 0.2,0.2 "
                + "--gamma-scales 100,10 "
                + "--mu 0.0 "
                + "--sigma 2.5 "
                + "-v -d -binary -train";
        return example;
    }

    private static void addOpitions() throws Exception {
        parser = new BasicParser();
        options = new Options();

        // data input
        addOption("dataset", "Dataset");
        addOption("word-voc-file", "Word vocabulary file");
        addOption("word-file", "Document word file");
        addOption("info-file", "Document info file");
        addOption("selected-docs-file", "(Optional) Indices of selected documents");
        addOption("prior-topic-file", "File containing prior topics");

        // data output
        addOption("output-folder", "Output folder");

        // sampling
        addSamplingOptions();

        // parameters
        addOption("alphas", "Alpha");
        addOption("betas", "Beta");
        addOption("gamma-means", "Gamma means");
        addOption("gamma-scales", "Gamma scales");
        addOption("rho", "Rho");
        addOption("mu", "Mu");
        addOption("sigmas", "Sigmas");
        addOption("Ks", "Number of topics");
        addOption("num-top-words", "Number of top words per topic");

        // configurations
        addOption("init", "Initialization");

        options.addOption("train", false, "train");
        options.addOption("test", false, "test");
        options.addOption("parallel", false, "parallel");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "z-normalize");
        options.addOption("help", false, "Help");
        options.addOption("example", false, "Example command");
        options.addOption("binary", false, "Binary responses");
    }

    private static void runModel() throws Exception {
        // sampling configurations
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "num-top-words", 20);
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 500);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 50);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 25);
        boolean paramOpt = cmd.hasOption("paramOpt");
        String init = CLIUtils.getStringArgument(cmd, "init", "random");
        InitialState initState;
        switch (init) {
            case "random":
                initState = InitialState.RANDOM;
                break;
            case "preset":
                initState = InitialState.PRESET;
                break;
            default:
                throw new RuntimeException("Initialization " + init + " not supported");
        }

        // model parameters
        int[] Ks = CLIUtils.getIntArrayArgument(cmd, "Ks", new int[]{15, 4}, ",");
        int L = Ks.length + 1;

        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas", new double[]{2.0, 1.0}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", new double[]{0.5, 0.25, 0.1}, ",");
        double[] gamma_means = CLIUtils.getDoubleArrayArgument(cmd, "gamma-means", new double[]{0.2, 0.2}, ",");
        double[] gamma_scales = CLIUtils.getDoubleArrayArgument(cmd, "gamma-scales", new double[]{100, 10}, ",");
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", new double[]{0.5, 2.5}, ",");
        String path = CLIUtils.getStringArgument(cmd, "path", "none");
        PathAssumption pathAssumption = getPathAssumption(path);

        // data input
        String datasetName = cmd.getOptionValue("dataset");
        String wordVocFile = cmd.getOptionValue("word-voc-file");
        String docWordFile = cmd.getOptionValue("word-file");

        // data output
        String outputFolder = cmd.getOptionValue("output-folder");

        double[][] priorTopics = null;
        if (cmd.hasOption("prior-topic-file")) {
            String priorTopicFile = cmd.getOptionValue("prior-topic-file");
            priorTopics = IOUtils.input2DArray(new File(priorTopicFile));
        }

        File docInfoFile = null;
        if (cmd.hasOption("info-file")) {
            docInfoFile = new File(cmd.getOptionValue("info-file"));
        }

        SNLDA sampler = new SNLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);

        boolean isBinary = cmd.hasOption("binary");
        ResponseTextDataset contData = new ResponseTextDataset(datasetName);
        LabelTextDataset binData = new LabelTextDataset(datasetName);
        int V;
        if (isBinary) {
            binData.loadFormattedData(new File(wordVocFile),
                    new File(docWordFile),
                    docInfoFile,
                    null);
            V = binData.getWordVocab().size();
            sampler.setWordVocab(binData.getWordVocab());
            sampler.configureBinary(outputFolder, V, Ks,
                    alphas, betas, gamma_means, gamma_scales, mu, sigmas,
                    initState, pathAssumption, paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);
        } else {
            contData.loadFormattedData(new File(wordVocFile),
                    new File(docWordFile),
                    docInfoFile,
                    null);
            V = contData.getWordVocab().size();
            sampler.setWordVocab(contData.getWordVocab());
            sampler.configureContinuous(outputFolder, V, Ks,
                    alphas, betas, gamma_means, gamma_scales, rho, mu, sigmas,
                    initState, pathAssumption, paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);
        }

        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (isTraining()) {
            ArrayList<Integer> trainDocIndices;
            if (isBinary) {
                trainDocIndices = sampler.getSelectedDocIndices(binData.getDocIds());
                sampler.train(binData.getWords(), trainDocIndices, binData.getSingleLabels());
            } else {
                trainDocIndices = sampler.getSelectedDocIndices(contData.getDocIds());
                double[] docResponses = contData.getResponses();
                if (cmd.hasOption("z")) { // z-normalization
                    ZNormalizer zNorm = new ZNormalizer(docResponses);
                    docResponses = zNorm.normalize(docResponses);
                }
                sampler.train(contData.getWords(), trainDocIndices, docResponses);
            }

            sampler.initialize(priorTopics);
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            sampler.outputNodePosteriors(new File(samplerFolder, "train-node-posteriors.txt"));
        }

        if (isTesting()) {
            int[][] testWords;
            ArrayList<Integer> testDocIndices;
            if (isBinary) {
                testWords = binData.getWords();
                testDocIndices = sampler.getSelectedDocIndices(binData.getDocIds());

            } else {
                testWords = contData.getWords();
                testDocIndices = sampler.getSelectedDocIndices(contData.getDocIds());
            }

            File testAssignmentFolder = new File(samplerFolder, AbstractSampler.IterAssignmentFolder);
            IOUtils.createFolder(testAssignmentFolder);

            File testPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
            IOUtils.createFolder(testPredFolder);

            double[] predictions;
            if (cmd.hasOption("parallel")) { // using multiple stored models
                predictions = SNLDA.parallelTest(testWords, testDocIndices, testPredFolder, testAssignmentFolder, sampler);
            } else { // using the last model
                File stateFile = sampler.getFinalStateFile();
                File outputPredFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".txt");
                File outputStateFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".zip");
                predictions = sampler.test(testWords, testDocIndices, stateFile, outputStateFile, outputPredFile);
                sampler.outputNodePosteriors(new File(samplerFolder, "test-node-posteriors.txt"));
            }

            if (isBinary) {
                // TODO
            } else {
                File teResultFolder = new File(samplerFolder,
                        AbstractExperiment.TEST_PREFIX + AbstractExperiment.RESULT_FOLDER);
                IOUtils.createFolder(teResultFolder);
                PredictionUtils.outputRegressionPredictions(
                        new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                        contData.getDocIds(), contData.getResponses(), predictions);
                PredictionUtils.outputRegressionResults(
                        new File(teResultFolder, AbstractExperiment.RESULT_FILE), contData.getResponses(),
                        predictions);
            }
        }
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOpitions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            } else if (cmd.hasOption("example")) {
                System.out.println(getExampleCmd());
                return;
            }

            runModel();

            // date and time
            DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
            Date dateobj = new Date();
            long eTime = (System.currentTimeMillis() - sTime) / 1000;
            System.out.println("Elapsed time: " + eTime + "s");
            System.out.println("End time: " + df.format(dateobj));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param newDocIndices Indices of test documents
     * @param iterPredFolder Output folder
     * @param iterStateFolder Folder to store assignments
     * @param sampler The configured sampler
     */
    public static double[] parallelTest(int[][] newWords,
            ArrayList<Integer> newDocIndices,
            File iterPredFolder,
            File iterStateFolder,
            SNLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        double[] avgPredictions = null;
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            ArrayList<File> partPredFiles = new ArrayList<>();
            for (String filename : filenames) { // all learned models
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);

                String stateFilename = IOUtils.removeExtension(filename);
                File iterOutputPredFile = new File(iterPredFolder, stateFilename + ".txt");
                File iterOutputStateFile = new File(iterStateFolder, stateFilename + ".zip");

                SNLDATestRunner runner = new SNLDATestRunner(sampler,
                        newWords, newDocIndices,
                        stateFile.getAbsolutePath(),
                        iterOutputStateFile.getAbsolutePath(),
                        iterOutputPredFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
                partPredFiles.add(iterOutputPredFile);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // average predictions
            avgPredictions = PredictionUtils.computeMultipleAverage(partPredFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
        return avgPredictions;
    }
}

class SNLDATestRunner implements Runnable {

    SNLDA sampler;
    int[][] newWords;
    ArrayList<Integer> newDocIndices;
    String stateFile;
    String outputStateFile;
    String outputPredictionFile;

    public SNLDATestRunner(SNLDA sampler,
            int[][] newWords,
            ArrayList<Integer> newDocIndices,
            String stateFile,
            String outputStateFile,
            String outputPredFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newDocIndices = newDocIndices;
        this.stateFile = stateFile;
        this.outputStateFile = outputStateFile;
        this.outputPredictionFile = outputPredFile;
    }

    @Override
    public void run() {
        SNLDA testSampler = new SNLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        try {
            testSampler.test(newWords, newDocIndices,
                    new File(stateFile),
                    new File(outputStateFile),
                    new File(outputPredictionFile));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
