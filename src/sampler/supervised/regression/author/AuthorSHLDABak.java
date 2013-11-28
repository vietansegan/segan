package sampler.supervised.regression.author;

import core.AbstractSampler;
import data.AuthorResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import optimization.GurobiMLRL1Norm;
import optimization.GurobiMLRL2Norm;
import regression.Regressor;
import sampler.RLDA;
import sampler.RecursiveLDA;
import sampler.TwoLevelHierSegLDA;
import sampler.supervised.regression.SHLDA;
import sampling.likelihood.DirMult;
import sampling.util.FullTable;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.TopicTreeNode;
import sampling.util.TopicTreeNode.PathAssumption;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class AuthorSHLDABak extends AbstractSampler implements Regressor<AuthorResponseTextDataset> {

    public static final int STAY = 0;
    public static final int PASS = 1;
    public static final Double WEIGHT_THRESHOLD = 10e-2;
    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    // hyperparameter indices
    public static final int ALPHA = 0;
    public static final int RHO = 1;
    public static final int GEM_MEAN = 2;
    public static final int GEM_SCALE = 3;
    public static final int TAU_MEAN = 4;
    public static final int TAU_SCALE = 5;
    // hyperparameters
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;    // regression parameter means
    protected double[] sigmas; // regression parameter variances
    // input data
    protected int[][][] words;  // [D] x [S_d] x [N_ds]: words
    protected int[] authors; // [D]: author of each document
    protected double[] responses; // [A]: response variables of each author
    protected int[][] authorDocIndices; // [A] x [D_a]
//    protected double[][] authorDocWeights; // [A] x [D_a]
    protected double[] docWeights;
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int A; // number of authors
    protected int C; // number of lexical regression parameters
    protected double T; // regularizer's parameter
    // input statistics
    protected int sentCount;
    protected int tokenCount;
    protected int[] docTokenCounts;
    // pre-computed hyperparameters
    protected double logAlpha;
    protected double sqrtRho;
    protected double[] sqrtSigmas;
    protected double[] logGammas;
    protected PathAssumption pathAssumption;
    // latent variables
    private STable[][] c; // path assigned to sentences
    private int[][][] z; // level assigned to tokens
    // state structure
    private SNode globalTreeRoot; // tree
    private Restaurant<STable, Integer, SNode>[] localRestaurants; // franchise
    // state statistics stored
    protected SparseVector lexicalWeights;
    protected ArrayList<Integer> lexicalList;
    protected int[][][] sentLevelCounts;
    // for regression
    protected double[] docLexicalWeights;
    protected double[] docTopicWeights;
    protected double[][] docLexicalDesignMatrix;
//    protected double[] authorLexicalWeights;
//    protected double[] authorTopicWeights;
//    protected double[][] authorLexicalDesignMatrix;
    // over time
    protected ArrayList<double[]> lexicalWeightsOverTime;
    // auxiliary
    protected double[] uniform;
    protected int numTokenAsgnsChange;
    protected int numSentAsntsChange;
    protected int numTableAsgnsChange;
    protected DirMult emptySwitch;
    protected ArrayList<String> authorVocab;

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }

    public void configure(String folder,
            int V, int L,
            double T,
            double alpha,
            double rho,
            double gem_mean,
            double gem_scale,
            double tau_mean,
            double tau_scale,
            double[] betas,
            double[] gammas,
            double[] mus,
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
        this.L = L;
        this.T = T;

        this.betas = betas;
        this.gammas = gammas;
        this.mus = mus;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(rho);
        this.hyperparams.add(gem_mean);
        this.hyperparams.add(gem_scale);
        this.hyperparams.add(tau_mean);
        this.hyperparams.add(tau_scale);
        for (double beta : betas) {
            this.hyperparams.add(beta);
        }
        for (double gamma : gammas) {
            this.hyperparams.add(gamma);
        }
        for (double mu : mus) {
            this.hyperparams.add(mu);
        }
        for (double sigma : sigmas) {
            this.hyperparams.add(sigma);
        }

        this.updatePrecomputedHyperparameters();

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.pathAssumption = pathAssumption;
        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();

        this.setName();

        // assert dimensions
        if (this.betas.length != this.L) {
            throw new RuntimeException("Vector betas must have length " + this.L
                    + ". Current length = " + this.betas.length);
        }
        if (this.gammas.length != this.L - 1) {
            throw new RuntimeException("Vector gammas must have length " + (this.L - 1)
                    + ". Current length = " + this.gammas.length);
        }
        if (this.mus.length != this.L) {
            throw new RuntimeException("Vector mus must have length " + this.L
                    + ". Current length = " + this.mus.length);
        }
        if (this.sigmas.length != this.L) {
            throw new RuntimeException("Vector sigmas must have length " + this.L
                    + ". Current length = " + this.sigmas.length);
        }

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- V = " + V);
            logln("--- T = " + T);
            logln("--- A = " + A);
            logln("--- L = " + L);
            logln("--- # documents = " + D); // number of groups
            logln("--- # sentences = " + sentCount);
            logln("--- # tokens = " + tokenCount);

            logln("--- folder\t" + folder);
            logln("--- max level:\t" + L);
            logln("--- alpha:\t" + hyperparams.get(ALPHA));
            logln("--- rho:\t" + hyperparams.get(RHO));
            logln("--- GEM mean:\t" + hyperparams.get(GEM_MEAN));
            logln("--- GEM scale:\t" + hyperparams.get(GEM_SCALE));
            logln("--- tau mean:\t" + hyperparams.get(TAU_MEAN));
            logln("--- tau scale:\t" + hyperparams.get(TAU_SCALE));

            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- reg mus:\t" + MiscUtils.arrayToString(mus));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));

            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.pathAssumption);

            logln("--- response distributions:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatisticsUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatisticsUtils.standardDeviation(responses)));
            int[] histogram = StatisticsUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }
        }
    }

    private void updatePrecomputedHyperparameters() {
        logAlpha = Math.log(hyperparams.get(ALPHA));
        sqrtRho = Math.sqrt(hyperparams.get(RHO));
        sqrtSigmas = new double[sigmas.length];
        for (int i = 0; i < sqrtSigmas.length; i++) {
            sqrtSigmas[i] = Math.sqrt(sigmas[i]);
        }
        logGammas = new double[gammas.length];
        for (int i = 0; i < logGammas.length; i++) {
            logGammas[i] = Math.log(gammas[i]);
        }
    }

//    private void updateAuthorTopicWeights() {
//        this.authorTopicWeights = new double[A];
//        for (int a = 0; a < A; a++) {
//            for (int d : authorDocIndices[a]) {
//                for (int s = 0; s < words[d].length; s++) {
//                    this.authorTopicWeights[a] += docWeights[d] *computeTopicWeight(d, s);
//                }
//            }
//        }
//    }
//    private void updateAuthorLexicalWeights() {
//        this.authorLexicalWeights = new double[A];
//        for (int a = 0; a < A; a++) {
//            for (int d : authorDocIndices[a]) {
//                for (int s = 0; s < words[d].length; s++) {
//                    for (int n = 0; n < words[d][s].length; n++) {
//                        Double w = this.lexicalWeights.get(words[d][s][n]);
//                        if (w != null) {
//                            this.authorLexicalWeights[d] += w;
//                        }
//                    }
//                }
//            }
//        }
//    }
    private void updateDocumentTopicWeights() {
        this.docTopicWeights = new double[D];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                this.docTopicWeights[d] += computeTopicWeight(d, s);
            }
        }
    }

    private void updateDocumentLexicalWeights() {
        this.docLexicalWeights = new double[D];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    Double w = this.lexicalWeights.get(words[d][s][n]);
                    if (w != null) {
                        this.docLexicalWeights[d] += w;
                    }
                }
            }
        }
    }

    @Override
    public String getName() {
        return this.name;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_author-SHLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_T-").append(T)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_gm-").append(formatter.format(hyperparams.get(GEM_MEAN)))
                .append("_gs-").append(formatter.format(hyperparams.get(GEM_SCALE)))
                .append("_tm-").append(formatter.format(hyperparams.get(TAU_MEAN)))
                .append("_ts-").append(formatter.format(hyperparams.get(TAU_SCALE)));
        int count = TAU_SCALE + 1;
        str.append("_b");
        for (int i = 0; i < betas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_g");
        for (int i = 0; i < gammas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_m");
        for (int i = 0; i < mus.length; i++) {
            str.append("-").append(formatter.format(mus[i]));
        }
        str.append("_s");
        for (int i = 0; i < sigmas.length; i++) {
            str.append("-").append(formatter.format(sigmas[i]));
        }
        str.append("_opt-").append(this.paramOptimized);
        str.append("_").append(this.paramOptimized);
        this.name = str.toString();
    }

    private void computeDataStatistics() {
        sentCount = 0;
        tokenCount = 0;
        docTokenCounts = new int[D];

//        authorDocCounts = new int[A];
//        authorTokenCounts = new int[A];
        for (int d = 0; d < D; d++) {
            int a = authors[d];
//            authorDocCounts[a]++;
            sentCount += words[d].length;
            for (int s = 0; s < words[d].length; s++) {
                tokenCount += words[d][s].length;
                docTokenCounts[d] += words[d][s].length;
//                authorTokenCounts[a] += words[d][s].length;
            }
        }
    }

    public void train(int[][][] ws, int[] as, double[] rs) {
        this.words = ws;
        this.authors = as;
        this.responses = rs;
        this.A = this.responses.length;
        this.D = this.words.length;

        ArrayList<Integer>[] authorDocList = new ArrayList[A];
        for (int a = 0; a < A; a++) {
            authorDocList[a] = new ArrayList<Integer>();
        }
        for (int d = 0; d < D; d++) {
            authorDocList[authors[d]].add(d);
        }
        this.authorDocIndices = new int[A][];
        for (int a = 0; a < A; a++) {
            this.authorDocIndices[a] = new int[authorDocList[a].size()];
            for (int dd = 0; dd < this.authorDocIndices[a].length; dd++) {
                this.authorDocIndices[a][dd] = authorDocList[a].get(dd);
            }
        }

        this.computeDataStatistics();
    }

    @Override
    public void train(AuthorResponseTextDataset trainData) {
        train(trainData.getSentenceWords(), trainData.getAuthors(), trainData.getAuthorResponses());
    }

    @Override
    public void test(AuthorResponseTextDataset testData) {
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;

        initializeLexicalWeights();
        initializeModelStructure();
        initializeDataStructure();
        initializeAssignments();

        updateDocumentTopicWeights();
        updateDocumentLexicalWeights();

//        updateAuthorTopicWeights();
//        updateAuthorLexicalWeights();

        if (verbose) {
            logln("--- --- Done initializing.\n" + getCurrentState());
            logln(printGlobalTree());
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        if (debug) {
            validate("Initialized");
        }
    }

    /**
     * Initialize lexical weights using LASSO
     */
    private void initializeLexicalWeights() {
        if (verbose) {
            logln("Initializing lexical weights ...");
        }

        this.docWeights = new double[D];
        for(int a=0; a<A; a++) {
            int numAuthorDocs = authorDocIndices[a].length;
            double weight = 1.0 / numAuthorDocs;
            for(int d : authorDocIndices[a]){
                docWeights[d] = weight;
            }
        }
        
        this.lexicalWeights = new SparseVector();
        this.lexicalList = new ArrayList<Integer>();
        if (T > 0) {
            GurobiMLRL1Norm lasso = new GurobiMLRL1Norm(T);
            double[] ws = null;
            try {
                File regFile = new File(this.folder, "author-init-weights-" + T + ".txt");
                if (regFile.exists()) {
                    ws = inputWeights(regFile);
                } else {
                    if (verbose) {
                        logln("--- Initial weights not found. " + regFile);
                        logln("--- Optimizing ...");
                    }
                    double[][] designMatrix = new double[A][V];
                    for (int a = 0; a < A; a++) {

                        for (int d : authorDocIndices[a]) {
                            double[] docWordCounts = new double[V];

                            for (int s = 0; s < words[d].length; s++) {
                                for (int n = 0; n < words[d][s].length; n++) {
                                    docWordCounts[words[d][s][n]]++;
                                }
                            }

                            for (int v = 0; v < V; v++) {
                                designMatrix[a][v] += docWeights[d] * docWordCounts[v] / docTokenCounts[d];
                            }
                        }

                    }
                    lasso.setDesignMatrix(designMatrix);
                    lasso.setResponseVector(responses);
                    ws = lasso.solve();
                    outputWeights(regFile, ws);
                }
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("Exception while initializing lexical weights");
            }

            int count = 0;
            for (int v = 0; v < V; v++) {
                if (Math.abs(ws[v]) >= WEIGHT_THRESHOLD) {
                    this.lexicalWeights.set(v, ws[v]);
                    this.lexicalList.add(v);
                    count++;
                }
            }
            this.C = count;
            if (verbose) {
                logln("--- # non-zero lexical weights: " + this.C);
            }

            // document design matrix for lexical items
            this.docLexicalDesignMatrix = new double[D][C];
            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    for (int n = 0; n < words[d][s].length; n++) {
                        int w = words[d][s][n];
                        if (this.lexicalWeights.containsIndex(w)) {
                            docLexicalDesignMatrix[d][lexicalList.indexOf(w)]++;
                        }
                    }
                }
                for (int ii = 0; ii < count; ii++) {
                    docLexicalDesignMatrix[d][ii] /= docTokenCounts[d];
                }
            }
        }
    }

    protected void outputWeights(File outputFile, double[] ws) throws Exception {
        if (verbose) {
            logln("--- Writing weights to file " + outputFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int ii = 0; ii < ws.length; ii++) {
            writer.write(wordVocab.get(ii) + "\t" + ws[ii] + "\n");
        }
        writer.close();
    }

    protected double[] inputWeights(File inputFile) throws Exception {
        if (verbose) {
            logln("--- Reading weights from file " + inputFile);
        }
        double[] ws = new double[V];
        BufferedReader reader = IOUtils.getBufferedReader(inputFile);
        for (int v = 0; v < V; v++) {
            ws[v] = Double.parseDouble(reader.readLine().split("\t")[1]);
        }
        reader.close();
        return ws;
    }

    /**
     * Initialize model structure.
     */
    protected void initializeModelStructure() {
        double stay = hyperparams.get(GEM_MEAN) * hyperparams.get(GEM_SCALE);
        double pass = (1 - hyperparams.get(GEM_MEAN)) * hyperparams.get(GEM_SCALE);
        double[] switchPrior = new double[]{stay, pass};
        emptySwitch = new DirMult(switchPrior);
        
        DirMult dmModel = new DirMult(V, betas[0] * V, uniform);
        double regParam = 0.0;
        this.globalTreeRoot = new SNode(iter, 0, 0, dmModel, regParam, null);
    }

    /**
     * Initialize data-specific structures.
     */
    protected void initializeDataStructure() {
        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<STable, Integer, SNode>();
        }

        this.sentLevelCounts = new int[D][][];
        for (int d = 0; d < D; d++) {
            this.sentLevelCounts[d] = new int[words[d].length][L];
        }

        this.c = new STable[D][];
        this.z = new int[D][][];
        for (int d = 0; d < D; d++) {
            c[d] = new STable[words[d].length];
            z[d] = new int[words[d].length][];
            for (int s = 0; s < words[d].length; s++) {
                z[d][s] = new int[words[d][s].length];
            }
        }
    }

    /**
     * Initialize assignments.
     */
    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                this.initializeRecursiveLDAAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }
    
    protected void initializeRecursiveLDAAssignments() {
        if (verbose) {
            logln("--- Initializing assignments using hierarchical segmented LDA ...");
        }
        double[] empBackgroundTopic = new double[V];
        int[][] docWords = new int[D][];
        for (int d = 0; d < D; d++) {
            docWords[d] = new int[docTokenCounts[d]];
            int count = 0;
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docWords[d][count++] = words[d][s][n];
                    empBackgroundTopic[words[d][s][n]]++;
                }
            }
        }

        for (int v = 0; v < V; v++) {
            empBackgroundTopic[v] /= tokenCount;
        }

        int init_burnin = 10;
        int init_maxiter = 20;
        int init_samplelag = 5;

        double alpha_1 = 0.1;
        double alpha_2 = 0.01;
        double beta_1 = 0.1;
        double beta_2 = 0.01;
        int numFirstTopics = 20;
        int numSecondTopics = 4;

        TwoLevelHierSegLDA sampler = new TwoLevelHierSegLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(false);
        sampler.setReport(false);

        sampler.configure(
                null,
                docWords, V, numFirstTopics, numSecondTopics,
                alpha_1, alpha_2,
                beta_1, beta_2,
                initState,
                paramOptimized, init_burnin, init_maxiter, init_samplelag, 1);

        try {
            File initFile = new File(this.folder, "hslda-init-"
                    + numFirstTopics + "-"
                    + numSecondTopics + ".zip");
            if (initFile.exists()) {
                sampler.inputState(initFile);
            } else {
                sampler.initialize();
                sampler.iterate();
                sampler.outputState(initFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        setLog(true);

        // initialize structure
        int topicLevel = 1;
        int frameLevel = 2;
        ArrayList<SNode> frameNodes = new ArrayList<SNode>();
        this.globalTreeRoot.setTopic(empBackgroundTopic);
        for (int k = 0; k < numFirstTopics; k++) {
            SNode topicNode = new SNode(iter, k, 1,
                    new DirMult(V, betas[topicLevel] * V, 1.0 / V),
                    SamplerUtils.getGaussian(mus[topicLevel], sigmas[topicLevel]),
                    globalTreeRoot);
            topicNode.setTopic(sampler.getFirstLevelTopics()[k].getDistribution());
            globalTreeRoot.addChild(k, topicNode);

            for (int f = 0; f < numSecondTopics; f++) {
                SNode frameNode = new SNode(iter, f, frameLevel,
                        new DirMult(V, betas[frameLevel] * V, 1.0 / V),
                        SamplerUtils.getGaussian(mus[topicLevel], sigmas[topicLevel]), topicNode);
                frameNode.setTopic(sampler.getSecondLevelTopics()[k][f].getDistribution());
                topicNode.addChild(f, frameNode);
                frameNodes.add(frameNode);
            }
        }

        logln(printGlobalTree());

        // assign
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
//                // create a new table for each sentence
//                TruncatedStickBreaking stick = new TruncatedStickBreaking(L,
//                        hyperparams.get(GEM_MEAN), hyperparams.get(GEM_SCALE));
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // assume all tokens are at the leave node, choose a path
                SparseCount sentObs = new SparseCount();
                for (int n = 0; n < words[d][s].length; n++) {
                    sentObs.increment(words[d][s][n]);
                }
                ArrayList<Double> logprobs = new ArrayList<Double>();
                for (int ii = 0; ii < frameNodes.size(); ii++) {
                    double lp = frameNodes.get(ii).getLogProbability(sentObs);
                    logprobs.add(lp);
                }
                int idx = SamplerUtils.logMaxRescaleSample(logprobs);
                SNode frameNode = frameNodes.get(idx);
                table.setContent(frameNode);
                addTableToPath(frameNode);

                // sample level for token
                for (int n = 0; n < words[d][s].length; n++) {
                    SNode[] path = getPathFromNode(frameNode);
                    logprobs = new ArrayList<Double>();
                    for (int l = 0; l < L; l++) {
                        double lp = path[l].getLogProbability(words[d][s][n]);
                        logprobs.add(lp);
                    }
                    idx = SamplerUtils.logMaxRescaleSample(logprobs);

                    z[d][s][n] = idx;
                    sentLevelCounts[d][s][z[d][s][n]]++;
                    addObservationToNode(path[z[d][s][n]], d, words[d][s][n]);
                }
            }
        }

        if (debug) {
            validate("After initial assignments");
        }

        this.sampleTopics();

        if (verbose) {
            logln("--- --- Start sampling paths for tables\n" + getCurrentState());
        }
        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                samplePathForTable(d, table, REMOVE, ADD, !OBSERVED, EXTEND);
            }
        }
    }

    protected void initializeRecursiveLDAAssignmentsOld() {
        // wait for background lda or background recursive lda
        if (verbose) {
            logln("--- Initializing assignments using recursive LDA ...");
        }
        RecursiveLDA rLDA = new RecursiveLDA();
        rLDA.setVerbose(verbose);
        rLDA.setDebug(debug);
        rLDA.setLog(false);
        rLDA.setReport(false);

        double[] empBackgroundTopic = new double[V];
        int[][] docWords = new int[D][];
        for (int d = 0; d < D; d++) {
            docWords[d] = new int[docTokenCounts[d]];
            int count = 0;
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docWords[d][count++] = words[d][s][n];
                    empBackgroundTopic[words[d][s][n]]++;
                }
            }
        }

        for (int v = 0; v < V; v++) {
            empBackgroundTopic[v] /= tokenCount;
        }

        int init_burnin = 250;
        int init_maxiter = 500;
        int init_samplelag = 25;

        int[] Ks = {16, 3};
        double[] init_alphas = {0.1, 0.1};
        double[] init_betas = {0.1, 0.1};
        double ratio = 1000;

        rLDA.configure(folder, docWords,
                V, Ks, ratio, init_alphas, init_betas, initState,
                paramOptimized, init_burnin, init_maxiter, init_samplelag, 1);

        try {
            File lldaZFile = new File(rLDA.getSamplerFolderPath(), "model.zip");
            if (lldaZFile.exists()) {
                rLDA.inputState(lldaZFile);
            } else {
                rLDA.sample();
                IOUtils.createFolder(rLDA.getSamplerFolderPath());
                rLDA.outputState(lldaZFile);
            }
            rLDA.setWordVocab(wordVocab);
            rLDA.outputTopicTopWords(new File(rLDA.getSamplerFolderPath(), TopWordFile), 20);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while initializing");
        }

        setLog(true);
        this.globalTreeRoot.setTopic(empBackgroundTopic);

        HashMap<RLDA, SNode> nodeMap = new HashMap<RLDA, SNode>();
        nodeMap.put(rLDA.getRoot(), globalTreeRoot);
        ArrayList<SNode> leafNodes = new ArrayList<SNode>();

        Queue<RLDA> queue = new LinkedList<RLDA>();
        queue.add(rLDA.getRoot());
        while (!queue.isEmpty()) {
            RLDA rldaNode = queue.poll();
            for (RLDA rldaChild : rldaNode.getChildren()) {
                queue.add(rldaChild);
            }

            if (rldaNode.getParent() == null) {
                continue;
            }

            int rLDAIndex = rldaNode.getIndex();
            int level = rldaNode.getLevel();

            if (rLDA.hasBackground() && level == 1 && rLDAIndex == RecursiveLDA.BACKGROUND) {
                continue; // skip background node
            }

            DirMult topic = new DirMult(V, betas[level] * V, 1.0 / V);
            double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
            SNode parent = nodeMap.get(rldaNode.getParent());
            int sNodeIndex = parent.getNextChildIndex();
            SNode node = new SNode(iter, sNodeIndex, level, topic, regParam, parent);
            node.setTopic(rldaNode.getParent().getTopics()[rLDAIndex].getDistribution());
            parent.addChild(sNodeIndex, node);

            nodeMap.put(rldaNode, node);

            level++;
            if (level == rLDA.getNumLevels()) {
                for (int ii = 0; ii < rldaNode.getTopics().length; ii++) {
                    DirMult subtopic = new DirMult(V, betas[level] * V, 1.0 / V);
                    double subregParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
                    SNode leaf = new SNode(iter, ii, level, subtopic, subregParam, node);
                    leaf.setTopic(rldaNode.getTopics()[ii].getDistribution());
                    node.addChild(ii, leaf);

                    leafNodes.add(leaf);
                }
            }
        }

        if (verbose) {
            logln(printGlobalTree());
            outputTopicTopWords(new File(getSamplerFolderPath(), "init-" + TopWordFile), 15);
        }

        // sample initial assignments
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // create a new table for each sentence
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // assume all tokens are at the leave node, choose a path
                SparseCount sentObs = new SparseCount();
                for (int n = 0; n < words[d][s].length; n++) {
                    sentObs.increment(words[d][s][n]);
                }

                ArrayList<Double> logprobs = new ArrayList<Double>();
                for (int ii = 0; ii < leafNodes.size(); ii++) {
                    double lp = leafNodes.get(ii).getLogProbability(sentObs);
                    logprobs.add(lp);
                }
                int idx = SamplerUtils.logMaxRescaleSample(logprobs);
                SNode frameNode = leafNodes.get(idx);
                table.setContent(frameNode);
                addTableToPath(frameNode);

                // sample level for token
                for (int n = 0; n < words[d][s].length; n++) {
                    SNode[] path = getPathFromNode(frameNode);
                    logprobs = new ArrayList<Double>();
                    for (int l = 0; l < L; l++) {
                        double lp = path[l].getLogProbability(words[d][s][n]);
                        logprobs.add(lp);
                    }
                    idx = SamplerUtils.logMaxRescaleSample(logprobs);

                    z[d][s][n] = idx;
                    sentLevelCounts[d][s][z[d][s][n]]++;
                    addObservationToNode(path[z[d][s][n]], d, words[d][s][n]);
                }

            }
        }

        if (debug) {
            validate("After initial assignments");
        }

        this.sampleTopics();

        if (verbose) {
            logln("--- --- Start sampling paths for tables\n" + getCurrentState());
        }
        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                samplePathForTable(d, table, REMOVE, ADD, !OBSERVED, EXTEND);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();
        this.lexicalWeightsOverTime = new ArrayList<double[]>();

        File repFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        try {
            if (report && !repFolderPath.exists()) {
                IOUtils.createFolder(repFolderPath);
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
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            double[] storeWeights = new double[V];
            for (int v = 0; v < V; v++) {
                Double w = this.lexicalWeights.get(v);
                if (w != null) {
                    storeWeights[v] = w;
                }
            }
            this.lexicalWeightsOverTime.add(storeWeights);

            if (verbose) {
                String str = "Iter " + iter
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t # tokens change: " + numTokenAsgnsChange
                        + "\t # sents change: " + numSentAsntsChange
                        + "\t # tables change: " + numTableAsgnsChange
                        + "\n" + getCurrentState()
                        + "\n";
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            numTableAsgnsChange = 0;
            numSentAsntsChange = 0;
            numTokenAsgnsChange = 0;

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    sampleTableForSentence(d, s, REMOVE, ADD, OBSERVED, EXTEND);

                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleLevelForToken(d, s, n, REMOVE, ADD, OBSERVED);
                    }
                }

                for (STable table : this.localRestaurants[d].getTables()) {
                    samplePathForTable(d, table, REMOVE, ADD, OBSERVED, EXTEND);
                }
            }

            optimizeTopicRegressionParameters();

            if (this.T > 0) {
                optimizeLexicalRegressionParameters();
            }

            sampleTopics();

            if (verbose) {
                double[] trPredResponses = getRegressionValues();
                RegressionEvaluation eval = new RegressionEvaluation(
                        (responses),
                        (trPredResponses));
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                if (paramOptimized) {
                    if (verbose) {
                        logln("--- --- Slice sampling ...");
                    }

                    sliceSample();
                    this.sampledParams.add(this.cloneHyperparameters());

                    if (verbose) {
                        logln("--- ---- " + MiscUtils.listToString(hyperparams));
                    }
                }
            }

            if (debug) {
                this.validate("Iteration " + iter);
            }
            

            float elapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
            logln("Elapsed time iterating: " + elapsedSeconds + " seconds");
            System.out.println();

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(repFolderPath, "iter-" + iter + ".zip"));
                outputTopicTopWords(new File(repFolderPath,
                        "iter-" + iter + "-top-words.txt"), 15);
            }
        }

        // output final model
        if (report) {
            outputState(new File(repFolderPath, "iter-" + iter + ".zip"));
        }

        if (verbose) {
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        float elapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + elapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                        "hyperparameters.txt"));
            }

            if (report) {
                // weights over time
//                outputLexicalWeightsOverTime(new File(getSamplerFolderPath(), "weights-over-time.txt"));
                // average weights
//                outputAverageLexicalWeights(new File(getSamplerFolderPath(), "weights.txt"));
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Compute the predicted values of the response variable
     */
    public double[] getRegressionValues() {
        double[] regValues = new double[A];
        for (int a = 0; a < A; a++) {
            regValues[a] = 0.0;
            for (int d : this.authorDocIndices[a]) {
                regValues[a] += docWeights[d] * (docTopicWeights[d] + docLexicalWeights[d]) / docTokenCounts[d];
            }
        }
        return regValues;
    }

    /**
     * Add a customer to a path. A path is specified by the pointer to its leaf
     * node. If the given node is not a leaf node, an exception will be thrown.
     * The number of customers at each node on the path will be incremented.
     *
     * @param leafNode The leaf node of the path
     */
    void addTableToPath(SNode leafNode) {
        SNode node = leafNode;
        while (node != null) {
            node.incrementNumCustomers();
            node = node.getParent();
        }
    }

    /**
     * Remove a customer from a path. A path is specified by the pointer to its
     * leaf node. The number of customers at each node on the path will be
     * decremented. If the number of customers at a node is 0, the node will be
     * removed.
     *
     * @param leafNode The leaf node of the path
     * @return Return the node that specifies the path that the leaf node is
     * removed from. If a lower-level node has no customer, it will be removed
     * and the lowest parent node on the path that has non-zero number of
     * customers will be returned.
     */
    SNode removeTableFromPath(SNode leafNode) {
        SNode retNode = leafNode;
        SNode node = leafNode;
        while (node != null) {
            node.decrementNumCustomers();
            if (node.isEmpty()) {
                retNode = node.getParent();
                node.getParent().removeChild(node.getIndex());
            }
            node = node.getParent();
        }
        return retNode;
    }

    /**
     * Add a set of observations (given their level assignments) to a path
     *
     * @param leafNode The leaf node identifying the path
     * @param observations The observations per level
     */
    SNode[] addObservationsToPath(SNode leafNode, int d, SparseCount[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], d, observations[l]);
        }
        return path;
    }

    /**
     * Add a set of observations to a node. This will (1) add the set of
     * observations to the node's topic, and (2) add the token counts for
     * switches from the root to the node.
     *
     * @param node The node
     * @param observations The set of observations
     */
    void addObservationsToNode(SNode node, int d, SparseCount observations) {
        for (int obs : observations.getIndices()) {
            int count = observations.getCount(obs);
            node.getContent().changeCount(obs, count);
        }
        int numObs = observations.getCountSum();
//        node.changeCount(d, STAY, numObs);
        node.changeCount(STAY, numObs);
        SNode temp = node.getParent();
        while (temp != null) {
//            temp.changeCount(d, PASS, numObs);
            temp.changeCount(PASS, numObs);
            temp = temp.getParent();
        }
    }

    /**
     * Remove a set of observations (given their level assignments) from a path
     *
     * @param leafNode The leaf node identifying the path
     * @param observations The observations per level
     */
    SNode[] removeObservationsFromPath(SNode leafNode, int d, SparseCount[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], d, observations[l]);
        }
        return path;
    }

    /**
     * Remove a set of observations from a node. This will (1) remove the set of
     * observations from the node's topic, and (2) remove the token counts from
     * the switches from the root to the node.
     *
     * @param node The node
     * @param observations The set of observations
     */
    void removeObservationsFromNode(SNode node, int d, SparseCount observations) {
        for (int obs : observations.getIndices()) {
            int count = observations.getCount(obs);
            node.getContent().changeCount(obs, -count);
        }
        int numObs = observations.getCountSum();
//        node.changeCount(d, STAY, -numObs);
        node.changeCount(STAY, -numObs);
        SNode temp = node.getParent();
        while (temp != null) {
//            temp.changeCount(d, PASS, -numObs);
            temp.changeCount(PASS, -numObs);
            temp = temp.getParent();
        }

    }

    /**
     * Remove an observation from a node. This will (1) remove the observation
     * from the topic at the given node, and (2) remove the token counts for
     * switches from the root to the given node.
     *
     * @param node The node from which the observation is removed from
     * @param d The document index
     * @param obs The observation
     */
    void removeObseravtionFromNode(SNode node, int d, int obs) {
        node.getContent().decrement(obs);
//        node.decrementCount(d, STAY);
        node.decrementCount(STAY);
        SNode temp = node.getParent();
        while (temp != null) {
//            temp.decrementCount(d, PASS);
            temp.decrementCount(PASS);
            temp = temp.getParent();
        }
    }

    /**
     * Add an observation to a node. This will (1) add the observation to the
     * topic at the given node, and (2) add the token counts for switches from
     * the root to the given node.
     *
     * @param node The node to which the observation is added to
     * @param d The document index
     * @param obs The observation
     */
    void addObservationToNode(SNode node, int d, int obs) {
        node.getContent().increment(obs);
//        node.incrementCount(d, STAY);
        node.incrementCount(STAY);
        SNode temp = node.getParent();
        while (temp != null) {
//            temp.incrementCount(d, PASS);
            temp.incrementCount(PASS);
            temp = temp.getParent();
        }
    }

    /**
     * Create a new path from an internal node.
     *
     * @param internalNode The internal node
     */
    SNode createNewPath(SNode internalNode) {
        SNode node = internalNode;
        for (int l = internalNode.getLevel(); l < L - 1; l++) {
            node = this.createNode(node);
        }
        return node;
    }

    /**
     * Create a node given a parent node
     *
     * @param parent The parent node
     */
    SNode createNode(SNode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmm = new DirMult(V, betas[level] * V, uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SNode child = new SNode(iter, nextChildIndex, level, dmm, regParam, parent);
        return parent.addChild(nextChildIndex, child);
    }

    /**
     * Get the observation counts per level of a sentence given the current
     * token assignments.
     *
     * @param d The document index
     * @param s The sentence index
     */
    protected SparseCount[] getSentObsCountPerLevel(int d, int s) {
        SparseCount[] counts = new SparseCount[L];
        for (int ll = 0; ll < L; ll++) {
            counts[ll] = new SparseCount();
        }
        for (int n = 0; n < words[d][s].length; n++) {
            int type = words[d][s][n];
            int level = z[d][s][n];
            counts[level].increment(type);
        }
        return counts;
    }

    /**
     * Get the observation counts per level of a given table
     *
     * @param d The document index
     * @param table The table
     */
    SparseCount[] getTableObsCountPerLevel(int d, STable table) {
        // observations of sentences currently being assign to this table
        SparseCount[] obsCountPerLevel = new SparseCount[L];
        for (int l = 0; l < L; l++) {
            obsCountPerLevel[l] = new SparseCount();
        }

        for (int s : table.getCustomers()) {
            for (int n = 0; n < words[d][s].length; n++) {
                int level = z[d][s][n];
                int obs = words[d][s][n];
                obsCountPerLevel[level].increment(obs);
            }
        }
        return obsCountPerLevel;
    }

    /**
     * Sample a level for a token
     *
     * @param d The document index
     * @param s The sentence index
     * @param n The token index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param observed Whether the response variable is observed
     */
    protected void sampleLevelForToken(
            int d, int s, int n,
            boolean remove, boolean add,
            boolean observed) {
        STable curTable = c[d][s];
        SNode[] curPath = getPathFromNode(curTable.getContent());

        if (observed) {
            docTopicWeights[d] -= curPath[z[d][s][n]].getRegressionParameter();
        }

        if (remove) {
            sentLevelCounts[d][s][z[d][s][n]]--;
            removeObseravtionFromNode(curPath[z[d][s][n]], d, words[d][s][n]);
        }

        double[] logprobs = new double[L];
        double[] logpriors = getLevelLogProbabilities(curPath, d);

        // debug
//        logln("d = " + d + ". s = " + s + ". n = " + n);

        for (int l = 0; l < L; l++) {
            double wordLlh = curPath[l].getLogProbability(words[d][s][n]);
            double resLlh = 0.0;
            if (observed) {
                double sum = docTopicWeights[d] + docLexicalWeights[d]
                        + curPath[l].getRegressionParameter();
                double mean = docWeights[d] * sum / docTokenCounts[d];

                double authorMean = mean;
                for (int dd : authorDocIndices[authors[d]]) {
                    if (dd != d) {
                        authorMean += docWeights[dd] * (docTopicWeights[dd] + docLexicalWeights[dd]) / docTokenCounts[dd];
                    }
                }

                resLlh = StatisticsUtils.logNormalProbability(responses[authors[d]], authorMean, sqrtRho);
            }
            logprobs[l] = logpriors[l] + wordLlh + resLlh;

            // debug
//            logln("l = " + l
//                    + ". lp = " + MiscUtils.formatDouble(logpriors[l])
//                    + ". word = " + MiscUtils.formatDouble(wordLlh)
//                    + ". res = " + MiscUtils.formatDouble(resLlh)
//                    + ". total = " + MiscUtils.formatDouble(logprobs[l]));
        }

        // debug
//        System.out.println();

        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);

        if (z[d][s][n] != sampledL) {
            numTokenAsgnsChange++;
        }

        // update and increment
        z[d][s][n] = sampledL;

        if (add) {
            sentLevelCounts[d][s][z[d][s][n]]++;
            addObservationToNode(curPath[z[d][s][n]], d, words[d][s][n]);
        }

        if (observed) {
            docTopicWeights[d] += curPath[z[d][s][n]].getRegressionParameter();
        }
    }

    /**
     * Compute the log probabilities of assigning a token of a given document to
     * a level on a given path.
     *
     * @param path The nodes on the given path
     * @param d The document index
     */
    private double[] getLevelLogProbabilities(SNode[] path, int d) {
        double[] logprobs = new double[L];

        double passLogProb = 0.0;
        for (int l = 0; l < L; l++) {
//            double stayLogProb = path[l].getSwitchLogProbability(d, STAY);
            double stayLogProb = path[l].getSwitchLogProbability(STAY);
            logprobs[l] = passLogProb + stayLogProb;

//            passLogProb = path[l].getSwitchLogProbability(d, PASS);
            passLogProb = path[l].getSwitchLogProbability(PASS);
        }

        return logprobs;
    }

    private double getPathLevelLogProbability(SNode[] path, int d, int[] lvlCounts) {
        int[] stayAndPassCounts = new int[lvlCounts.length];
        stayAndPassCounts[lvlCounts.length - 1] = lvlCounts[lvlCounts.length - 1];
        for (int l = lvlCounts.length - 2; l >= 0; l--) {
            stayAndPassCounts[l] = stayAndPassCounts[l + 1] + lvlCounts[l];
        }

        double val = 0.0;
        for (int l = 0; l < lvlCounts.length; l++) {
            HashMap<Integer, Integer> lvlSwitchCounts = new HashMap<Integer, Integer>();
            if (lvlCounts[l] > 0) {
                lvlSwitchCounts.put(STAY, lvlCounts[l]);
            }
            if (stayAndPassCounts[l] - lvlCounts[l] > 0) {
                lvlSwitchCounts.put(PASS, stayAndPassCounts[l] - lvlCounts[l]);
            }

//            val += path[l].getSwitchLogProbability(d, lvlSwitchCounts);
            val += path[l].getSwitchLogProbability(lvlSwitchCounts);
        }
        return val;
    }

    private double getNewPathLevelLogProbability(int[] lvlCounts) {
        int[] stayAndPassCounts = new int[lvlCounts.length];
        stayAndPassCounts[lvlCounts.length - 1] = lvlCounts[lvlCounts.length - 1];
        for (int l = lvlCounts.length - 2; l >= 0; l--) {
            stayAndPassCounts[l] = stayAndPassCounts[l + 1] + lvlCounts[l];
        }

        double val = 0.0;
        for (int l = 0; l < lvlCounts.length; l++) {
            HashMap<Integer, Integer> lvlSwitchCounts = new HashMap<Integer, Integer>();
            lvlSwitchCounts.put(STAY, lvlCounts[l]);
            lvlSwitchCounts.put(PASS, stayAndPassCounts[l] - lvlCounts[l]);
            val += emptySwitch.getLogLikelihood(lvlSwitchCounts);
        }
        return val;
    }

    /**
     * Sample a table assignment for a sentence
     *
     * @param d The document index
     * @param s The sentence index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param observed Whether the response is observed
     * @param extend Whether the structure is extendable
     */
    protected void sampleTableForSentence(int d, int s,
            boolean remove, boolean add,
            boolean observed, boolean extend) {
        STable curTable = c[d][s];

        SparseCount[] sentObsCountPerLevel = getSentObsCountPerLevel(d, s);

        if (observed) {
            this.docTopicWeights[d] -= computeTopicWeight(d, s);
        }

        if (remove) {
            removeObservationsFromPath(c[d][s].getContent(), d, sentObsCountPerLevel);
            localRestaurants[d].removeCustomerFromTable(s, c[d][s].getIndex());
            if (c[d][s].isEmpty()) {
                removeTableFromPath(c[d][s].getContent());
                localRestaurants[d].removeTable(c[d][s].getIndex());
            }
        }

        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logProbs = new ArrayList<Double>();

        // existing tables
        for (STable table : localRestaurants[d].getTables()) {
            double logprior = Math.log(table.getNumCustomers());
            SNode[] path = getPathFromNode(table.getContent());
            double wordLlh = 0.0;
            for (int l = 0; l < L; l++) {
                for (int obs : sentObsCountPerLevel[l].getIndices()) {
                    wordLlh += path[l].getLogProbability(obs) * sentObsCountPerLevel[l].getCount(obs);
                }
            }

            // log prob of the stick breaking at this table
            double stickLp = getPathLevelLogProbability(path, d, sentLevelCounts[d][s]);

            double resLlh = 0.0;
            if (observed) {
                double addTopicWeight = 0.0;
                for (int l = 0; l < L; l++) {
                    addTopicWeight += path[l].getRegressionParameter() * sentLevelCounts[d][s][l];
                }

                double mean = docWeights[d] * (docTopicWeights[d] + docLexicalWeights[d] + addTopicWeight) / docTokenCounts[d];

                double authorMean = mean;
                for (int dd : authorDocIndices[authors[d]]) {
                    if (dd != d) {
                        authorMean += docWeights[dd] * (docTopicWeights[dd] + docLexicalWeights[dd]) / docTokenCounts[dd];
                    }
                }

                resLlh = StatisticsUtils.logNormalProbability(responses[authors[d]], authorMean, sqrtRho);
            }

            double lp = logprior + wordLlh + resLlh + stickLp;
            logProbs.add(lp);
            tableIndices.add(table.getIndex());

            // debug
//            logln("iter = " + iter + ". d = " + d + ". s = " + s
//                    + ". table: " + table.toString()
//                    + ". log prior = " + MiscUtils.formatDouble(logprior)
//                    + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                    + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                    + ". stick llh = " + MiscUtils.formatDouble(stickLp)
//                    + ". lp = " + MiscUtils.formatDouble(lp));
        }

        // new table
        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (extend) {
            // log priors
            computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

            // word log likelihoods
            computePathWordLogLikelihood(pathWordLlhs,
                    globalTreeRoot,
                    sentObsCountPerLevel,
                    0.0);

            // debug
            if (pathLogPriors.size() != pathWordLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }

            // response log likelihoods
            if (observed) {
                pathResLlhs = computePathResponseLogLikelihood(d, s);

                // debug
                if (pathLogPriors.size() != pathResLlhs.size()) {
                    throw new RuntimeException("Numbers of paths mismatch");
                }
            }

            double logPrior = logAlpha;
            double marginals = computeMarginals(pathLogPriors, pathWordLlhs, pathResLlhs, observed);

            double newStickLogProb = getNewPathLevelLogProbability(sentLevelCounts[d][s]);
            double lp = logPrior + marginals + newStickLogProb;
            logProbs.add(lp);
            tableIndices.add(PSEUDO_TABLE_INDEX);

            // debug
//            logln("iter = " + iter + ". d = " + d + ". s = " + s
//                    + ". new table"
//                    + ". log prior = " + MiscUtils.formatDouble(logPrior)
//                    + ". new stick = " + MiscUtils.formatDouble(newStickLogProb)
//                    + ". marginal = " + MiscUtils.formatDouble(marginals)
//                    + ". lp = " + MiscUtils.formatDouble(lp));
        }

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        int tableIdx = tableIndices.get(sampledIndex);

        // debug
//        logln(">>> idx = " + sampledIndex + ". tabIdx = " + tableIdx + "\n\n");

        if (curTable != null && curTable.getIndex() != tableIdx) {
            numSentAsntsChange++;
        }

        STable table;
        if (tableIdx == PSEUDO_NODE_INDEX) {
            int newTableIdx = localRestaurants[d].getNextTableIndex();
            table = new STable(iter, newTableIdx, null, d);
            localRestaurants[d].addTable(table);

            SNode newNode = samplePath(pathLogPriors, pathWordLlhs, pathResLlhs, observed);
            if (!isLeafNode(newNode)) {
                newNode = createNewPath(newNode);
            }
            table.setContent(newNode);
            addTableToPath(table.getContent());
        } else {
            table = localRestaurants[d].getTable(tableIdx);
        }

        // debug
//        logln("---> assigned table: " + table.toString());

        c[d][s] = table;

        if (add) {
            addObservationsToPath(table.getContent(), d, sentObsCountPerLevel);
            localRestaurants[d].addCustomerToTable(s, table.getIndex());
        }

        if (observed) {
            docTopicWeights[d] += computeTopicWeight(d, s);
        }
    }

    /**
     * Sample a path on the global tree for a table
     *
     * @param d The restaurant index
     * @param table The table
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param observed Whether the response variable is observed
     * @param extend Whether the global tree is extendable
     */
    private void samplePathForTable(int d, STable table,
            boolean remove, boolean add,
            boolean observed, boolean extend) {
        SNode curLeaf = table.getContent();

        // observation counts of this table per level
        SparseCount[] tabObsCountPerLevel = getTableObsCountPerLevel(d, table);

        if (observed) {
            for (int s : table.getCustomers()) {
                docTopicWeights[d] -= computeTopicWeight(d, s);
            }
        }

        if (remove) {
            removeObservationsFromPath(table.getContent(), d, tabObsCountPerLevel);
            removeTableFromPath(table.getContent());
        }

        // log priors
        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

        // word log likelihoods
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        computePathWordLogLikelihood(pathWordLlhs, globalTreeRoot, tabObsCountPerLevel, 0.0);

        // debug
        if (pathLogPriors.size() != pathWordLlhs.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        // response log likelihoods
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (observed) {
            pathResLlhs = computePathResponseLogLikelihood(d, table);

            if (pathLogPriors.size() != pathResLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }
        }

        // sample
        ArrayList<SNode> pathList = new ArrayList<SNode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SNode path : pathLogPriors.keySet()) {
            if (!extend && !isLeafNode(path)) {
                continue;
            }

            double lp = pathLogPriors.get(path) + pathWordLlhs.get(path);
            if (observed) {
                lp += pathResLlhs.get(path);
            }

            logProbs.add(lp);
            pathList.add(path);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode newLeaf = pathList.get(sampledIndex);

        // debug
        if (curLeaf == null || curLeaf.equals(newLeaf)) {
            numTableAsgnsChange++;
        }

        // if pick an internal node, create the path from the internal node to leave
        if (newLeaf.getLevel() < L - 1) {
            newLeaf = this.createNewPath(newLeaf);
        }

        // update
        table.setContent(newLeaf);

        if (add) {
            addTableToPath(newLeaf);
            addObservationsToPath(newLeaf, d, tabObsCountPerLevel);
        }

        if (observed) {
            for (int s : table.getCustomers()) {
                docTopicWeights[d] += computeTopicWeight(d, s);
            }
        }
    }

    /**
     * Sample topics of each tree node
     */
    protected void sampleTopics() {
        // get all leaves of the tree
        ArrayList<SNode> leaves = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (node.getChildren().isEmpty()) {
                leaves.add(node);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        // bottom-up smoothing to compute pseudo-counts from children
        Queue<SNode> queue = new LinkedList<SNode>();
        for (SNode leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            SNode node = queue.poll();
            if (node.equals(globalTreeRoot)) {
                break;
            }

            SNode parent = node.getParent();
            if (!queue.contains(parent)) {
                queue.add(parent);
            }

            if (node.isLeaf()) {
                continue;
            }

            if (this.pathAssumption == PathAssumption.MINIMAL) {
                node.getPseudoCountsFromChildrenMin();
            } else if (this.pathAssumption == PathAssumption.MAXIMAL) {
                node.getPseudoCountsFromChildrenMax();
            } else {
                throw new RuntimeException("Path assumption " + this.pathAssumption
                        + " is not supported.");
            }
        }

        // top-down sampling to get topics
        queue = new LinkedList<SNode>();
        queue.add(globalTreeRoot);
        while (!queue.isEmpty()) {
            SNode node = queue.poll();
            for (SNode child : node.getChildren()) {
                queue.add(child);
            }

            node.sampleTopic(betas[node.getLevel()], betas[node.getLevel()]);
        }
    }

    protected void optimizeLexicalRegressionParameters() {
        double[] responseVector = new double[A];
        for (int a = 0; a < A; a++) {
            double topicVal = 0.0;
            for (int d : authorDocIndices[a]) {
                topicVal += docWeights[d] * docTopicWeights[d] / docTokenCounts[d];
            }
            responseVector[a] = responses[a] - topicVal;
        }

        double[][] authorLexDesignMatrix = new double[A][C];
        for (int a = 0; a < A; a++) {
            for (int d : authorDocIndices[a]) {
                for (int ii = 0; ii < C; ii++) {
                    authorLexDesignMatrix[a][ii] += docWeights[d] * docLexicalDesignMatrix[d][ii];
                }
            }
        }

        double ratio = hyperparams.get(RHO) / hyperparams.get(TAU_SCALE);
        GurobiMLRL2Norm mlr =
                new GurobiMLRL2Norm(authorLexDesignMatrix, responseVector, ratio);
        double[] weights = mlr.solve();
        for (int ii = 0; ii < weights.length; ii++) {
            int v = this.lexicalList.get(ii);
            this.lexicalWeights.set(v, weights[ii]);
        }
        this.updateDocumentLexicalWeights();
    }

    /**
     * Optimize lexical regression parameters.
     */
    protected void optimizeTopicRegressionParameters() {
        ArrayList<SNode> flattenTree = flattenTreeWithoutRoot();
        int numNodes = flattenTree.size();

        double[] lambdas = new double[numNodes];
        HashMap<SNode, Integer> nodeIndices = new HashMap<SNode, Integer>();
        for (int i = 0; i < flattenTree.size(); i++) {
            SNode node = flattenTree.get(i);
            nodeIndices.put(node, i);
            lambdas[i] = hyperparams.get(RHO) / sigmas[node.getLevel()];
        }

        double[] responseVector = new double[A];
        for (int a = 0; a < A; a++) {
            double lexVal = 0.0;
            for (int d : authorDocIndices[a]) {
                lexVal += docWeights[d] * docLexicalWeights[d] / docTokenCounts[d];
            }
            responseVector[a] = responses[a] - lexVal;
        }

        double[][] authorTopicDesginMatrix = new double[A][numNodes];
        for (int a = 0; a < A; a++) {
            for (int d : authorDocIndices[a]) {
                double[] docTopicCounts = new double[numNodes];
                for (int s = 0; s < words[d].length; s++) {
                    SNode[] path = getPathFromNode(c[d][s].getContent());
                    for (int l = 1; l < L; l++) {
                        int nodeIdx = nodeIndices.get(path[l]);
                        int count = sentLevelCounts[d][s][l];
                        docTopicCounts[nodeIdx] += count;
                    }
                }
                for (int nn = 0; nn < numNodes; nn++) {
                    authorTopicDesginMatrix[a][nn] += docWeights[d] * docTopicCounts[nn] / docTokenCounts[d];
                }
            }
        }

        GurobiMLRL2Norm mlr =
                new GurobiMLRL2Norm(authorTopicDesginMatrix, responseVector, lambdas);
        double[] weights = mlr.solve();

        // update
        for (int i = 0; i < numNodes; i++) {
            flattenTree.get(i).setRegressionParameter(weights[i]);
        }
        this.updateDocumentTopicWeights();
    }

    ArrayList<SNode> flattenTreeWithoutRoot() {
        ArrayList<SNode> flattenTree = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (!node.isRoot()) {
                flattenTree.add(node);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return flattenTree;
    }

    /**
     * Sample a path
     *
     * @param logPriors Path log priors
     * @param wordLlhs Path word log likelihoods
     * @param resLlhs Path response variable log likelihoods
     */
    SNode samplePath(
            HashMap<SNode, Double> logPriors,
            HashMap<SNode, Double> wordLlhs,
            HashMap<SNode, Double> resLlhs,
            boolean observed) {
        ArrayList<SNode> pathList = new ArrayList<SNode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SNode node : logPriors.keySet()) {
            double lp = logPriors.get(node) + wordLlhs.get(node);
            if (observed) {
                lp += resLlhs.get(node);
            }
            pathList.add(node);
            logProbs.add(lp);

            // debug
//            if (observed) {
//                logln("--- " + (pathList.size() - 1)
//                        + ". " + node.toString()
//                        + ". logprior: " + MiscUtils.formatDouble(logPriors.get(node))
//                        + ". wordllh: " + MiscUtils.formatDouble(wordLlhs.get(node))
//                        + ". resllh: " + MiscUtils.formatDouble(resLlhs.get(node))
//                        + ". lp: " + MiscUtils.formatDouble(lp));
//            } else {
//                logln("--- " + (pathList.size() - 1)
//                        + ". " + node.toString()
//                        + ". logprior: " + MiscUtils.formatDouble(logPriors.get(node))
//                        + ". wordllh: " + MiscUtils.formatDouble(wordLlhs.get(node))
//                        + ". lp: " + MiscUtils.formatDouble(lp));
//            }
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode path = pathList.get(sampledIndex);

        // debug
//        logln("--- >>> sampler idx: " + sampledIndex
//                + ". " + path.toString()
//                + "\n\n");

        return path;
    }

    /**
     * Compute the log probability of a new table, marginalized over all
     * possible paths
     *
     * @param pathLogPriors The log priors of each path
     * @param pathWordLogLikelihoods The word likelihoods
     * @param pathResLogLikelihoods The response variable likelihoods
     */
    double computeMarginals(
            HashMap<SNode, Double> pathLogPriors,
            HashMap<SNode, Double> pathWordLogLikelihoods,
            HashMap<SNode, Double> pathResLogLikelihoods,
            boolean resObserved) {
        double marginal = 0.0;
        for (SNode node : pathLogPriors.keySet()) {
            double logprior = pathLogPriors.get(node);
            double loglikelihood = pathWordLogLikelihoods.get(node);

            double lp = logprior + loglikelihood;
            if (resObserved) {
                lp += pathResLogLikelihoods.get(node);
            }

            if (marginal == 0.0) {
                marginal = lp;
            } else {
                marginal = SamplerUtils.logAdd(marginal, lp);
            }
        }
        return marginal;
    }

    /**
     * Compute the log probability of the response variable when the given table
     * is assigned to each path
     *
     * @param d The document index
     * @param table The table
     */
    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            STable table) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            SNode[] path = getPathFromNode(node);
            double addSum = 0.0;
            double var = hyperparams.get(RHO);
            int level;
            for (level = 0; level < path.length; level++) {
                for (int s : table.getCustomers()) {
                    addSum += path[level].getRegressionParameter() * sentLevelCounts[d][s][level];
                }
            }
            while (level < L) {
                int totalLevelCount = 0;
                for (int s : table.getCustomers()) {
                    int levelCount = sentLevelCounts[d][s][level];
                    addSum += levelCount * mus[level];
                    totalLevelCount += levelCount;
                }
                var += Math.pow((double) totalLevelCount / docTokenCounts[d], 2) * sigmas[level];
                level++;
            }

            double mean = docWeights[d] * (docTopicWeights[d] + docLexicalWeights[d] + addSum) / docTokenCounts[d];

            double authorMean = mean;
            for (int dd : authorDocIndices[authors[d]]) {
                if (dd != d) {
                    authorMean += docWeights[dd] * (docTopicWeights[dd] + docLexicalWeights[dd]) / docTokenCounts[dd];
                }
            }

            double resLlh = StatisticsUtils.logNormalProbability(responses[authors[d]], authorMean, Math.sqrt(var));
            resLlhs.put(node, resLlh);

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return resLlhs;
    }

    /**
     * Compute the log probability of the response variable when the given
     * sentence is assigned to each path
     *
     * @param d The document index
     * @param s The sentence index
     */
    private HashMap<SNode, Double> computePathResponseLogLikelihood(int d, int s) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            SNode[] path = getPathFromNode(node);
            double addTopicWeight = 0.0;
            double var = hyperparams.get(RHO);
            int level;
            for (level = 0; level < path.length; level++) {
                addTopicWeight += path[level].getRegressionParameter() * sentLevelCounts[d][s][level];
            }

            while (level < L) {
                int levelCount = sentLevelCounts[d][s][level];
                addTopicWeight += levelCount * mus[level];
                var += Math.pow((double) levelCount / docTokenCounts[d], 2) * sigmas[level];
                level++;
            }

            // note: the topic weight of the current sentence s has been excluded
            // from docTopicWeights[d]
            double mean = docWeights[d] * (docTopicWeights[d] + docLexicalWeights[d] + addTopicWeight) / docTokenCounts[d];

            double authorMean = mean;
            for (int dd : authorDocIndices[authors[d]]) {
                if (dd != d) {
                    authorMean += docWeights[dd] * (docTopicWeights[dd] + docLexicalWeights[dd]) / docTokenCounts[dd];
                }
            }

            double resLlh = StatisticsUtils.logNormalProbability(responses[authors[d]], authorMean, Math.sqrt(var));
            resLlhs.put(node, resLlh);

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        return resLlhs;
    }

    /**
     * Compute the log probability of assigning a set of words to each path in
     * the tree
     *
     * @param nodeDataLlhs HashMap to store the result
     * @param curNode The current node in recursive calls
     * @param tokenCountPerLevel Token counts per level
     * @param parentDataLlh The value passed from the parent node
     */
    void computePathWordLogLikelihood(
            HashMap<SNode, Double> nodeDataLlhs,
            SNode curNode,
            SparseCount[] tokenCountPerLevel,
            double parentDataLlh) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getLogProbability(tokenCountPerLevel[level]);

        // populate to child nodes
        for (SNode child : curNode.getChildren()) {
            computePathWordLogLikelihood(nodeDataLlhs, child, tokenCountPerLevel,
                    parentDataLlh + nodeDataLlh);
        }

        // store the data llh from the root to this current node
        double storeDataLlh = parentDataLlh + nodeDataLlh;
        level++;
        while (level < L) { // if this is an internal node, add llh of new child node
            DirMult dirMult;
            if (curNode.getTopic() == null) {
                dirMult = new DirMult(V, betas[level] * V, 1.0 / V);
            } else {
                dirMult = new DirMult(V, betas[level] * V, curNode.getTopic());
            }
            storeDataLlh += dirMult.getLogLikelihood(tokenCountPerLevel[level].getObservations());
            level++;
        }
        nodeDataLlhs.put(curNode, storeDataLlh);
    }

    /**
     * Recursively compute the log probability of each path in the global tree
     *
     * @param nodeLogProbs HashMap to store the results
     * @param curNode Current node in the recursive call
     * @param parentLogProb The log probability passed from the parent node
     */
    void computePathLogPrior(
            HashMap<SNode, Double> nodeLogProbs,
            SNode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);
            newWeight += logGammas[curNode.getLevel()] - logNorm;

            for (SNode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight);
            }
        }
        nodeLogProbs.put(curNode, newWeight);
    }

    /**
     * Compute the regression sum from the topic tree for a sentence
     *
     * @param d The document index
     * @param s The sentence index
     * @return The regression sum of the sentence
     */
    protected double computeTopicWeight(int d, int s) {
        double regSum = 0.0;
        SNode[] path = getPathFromNode(c[d][s].getContent());
        for (int l = 0; l < path.length; l++) {
            regSum += path[l].getRegressionParameter() * sentLevelCounts[d][s][l];
        }
        return regSum;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    SNode[] getPathFromNode(SNode node) {
        SNode[] path = new SNode[node.getLevel() + 1];
        SNode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
    }

    /**
     * Check whether a given node is a leaf node.
     *
     * @param node The node
     */
    boolean isLeafNode(SNode node) {
        return node.getLevel() == L - 1;
    }

    /**
     * Parse the node path string.
     *
     * @param nodePath The node path string
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
    private SNode getNode(int[] parsedPath) {
        SNode node = globalTreeRoot;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg);
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        double switchLogprob = 0.0;
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood();
            switchLogprob += node.getSwitchLogLikelihood();

            regParamLgprob += StatisticsUtils.logNormalProbability(node.getRegressionParameter(),
                    mus[node.getLevel()], Math.sqrt(sigmas[node.getLevel()]));

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(gammas[node.getLevel()]);
            }

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double resLlh = 0.0;
        double restLgprob = 0.0;

        for (int d = 0; d < D; d++) {
            restLgprob += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));
        }

        double[] regValues = getRegressionValues();
        for (int a = 0; a < A; a++) {
            resLlh += StatisticsUtils.logNormalProbability(responses[a],
                    regValues[a], sqrtRho);
        }

        logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                + ". tree = " + MiscUtils.formatDouble(treeLogProb)
                + ". rest = " + MiscUtils.formatDouble(restLgprob)
                + ". switch = " + MiscUtils.formatDouble(switchLogprob)
                + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                + ". response = " + MiscUtils.formatDouble(resLlh));

        double llh = wordLlh + treeLogProb + switchLogprob + regParamLgprob + resLlh + restLgprob;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(printGlobalTreeSummary()).append("\n");
        str.append(printLocalRestaurantSummary()).append("\n");
        return str.toString();
    }

    @Override
    public void output(File samplerFile) {
        this.outputState(samplerFile.getAbsolutePath());
    }

    @Override
    public void input(File samplerFile) {
        this.inputModel(samplerFile.getAbsolutePath());
    }

    public void inputFinalModel() {
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        this.inputModel(new File(reportFolder, "iter-" + MAX_ITER + ".zip").getAbsolutePath());
    }

    public void inputFinalState() {
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        this.inputState(new File(reportFolder, "iter-" + MAX_ITER + ".zip"));
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath + "\n");
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            modelStr.append(SparseVector.output(lexicalWeights)).append("\n");

            Stack<SNode> stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getIterationCreated()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");
                modelStr.append(DirMult.outputDistribution(node.getContent().getSamplingDistribution())).append("\n");

                for (SNode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d)
                        .append("\t").append(localRestaurants[d].getNumTables())
                        .append("\n");
                for (STable table : localRestaurants[d].getTables()) {
                    assignStr.append(table.getIndex()).append("\n");
                    assignStr.append(table.getIterationCreated()).append("\n");
                    assignStr.append(table.getContent().getPathString()).append("\n");
                }
            }
            
            stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(DirMult.output(node.getSwitch())).append("\n");
//                modelStr.append(node.switches.size()).append("\n");
//                for(int dd : node.switches.keySet()) {
//                    modelStr.append(dd).append("\n");
//                    modelStr.append(DirMult.output(node.switches.get(dd))).append("\n");
//                }

                for (SNode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    assignStr.append(d)
                            .append(":").append(s)
                            .append("\t").append(c[d][s].getIndex())
                            .append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        assignStr.append(d)
                                .append(":").append(t)
                                .append(":").append(n)
                                .append("\t").append(z[d][t][n])
                                .append("\n");
                    }
                }
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath + "\n");
        }
    }

    void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath + "\n");
        }
    }

    void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath + "\n");
        }
    }

    public String printGlobalTreeSummary() {
        StringBuilder str = new StringBuilder();
        int[] nodeCountPerLevel = new int[L];
        int[] obsCountPerLevel = new int[L];

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        int totalObs = 0;
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            nodeCountPerLevel[node.getLevel()]++;
            obsCountPerLevel[node.getLevel()] += node.getContent().getCountSum();

            totalObs += node.getContent().getCountSum();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("global tree:\n\t>>> node count per level: ");
        for (int l = 0; l < L; l++) {
            str.append(l).append("(")
                    .append(nodeCountPerLevel[l])
                    .append(", ").append(obsCountPerLevel[l])
                    .append(");\t");
        }
        str.append("\n");
        str.append("\t>>> # observations = ").append(totalObs)
                .append("\n\t>>> # customers = ").append(globalTreeRoot.getNumCustomers());
        return str.toString();
    }

    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        str.append("global tree\n");

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        int totalObs = 0;

        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    .append("\n");

            totalObs += node.getContent().getCountSum();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # customers = ").append(globalTreeRoot.getNumCustomers())
                .append("\n");
        return str.toString();
    }

    public String printLocalRestaurantSummary() {
        StringBuilder str = new StringBuilder();
        str.append("local restaurants:\n");
        int[] numTables = new int[D];
        int totalTableCusts = 0;
        for (int d = 0; d < D; d++) {
            numTables[d] = localRestaurants[d].getNumTables();
            for (STable table : localRestaurants[d].getTables()) {
                totalTableCusts += table.getNumCustomers();
            }
        }
        str.append("\t>>> # tables:")
                .append(". min: ").append(MiscUtils.formatDouble(StatisticsUtils.min(numTables)))
                .append(". max: ").append(MiscUtils.formatDouble(StatisticsUtils.max(numTables)))
                .append(". avg: ").append(MiscUtils.formatDouble(StatisticsUtils.mean(numTables)))
                .append(". total: ").append(MiscUtils.formatDouble(StatisticsUtils.sum(numTables)))
                .append("\n");
        str.append("\t>>> # customers: ").append(totalTableCusts);
        return str.toString();
    }

    public String printLocalRestaurants() {
        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            logln("restaurant d = " + d
                    + ". # tables: " + localRestaurants[d].getNumTables()
                    + ". # total customers: " + localRestaurants[d].getTotalNumCustomers());
            for (STable table : localRestaurants[d].getTables()) {
                logln("--- table: " + table.toString());
            }
            System.out.println();
        }
        return str.toString();
    }

    public String printLocalRestaurant(int d) {
        StringBuilder str = new StringBuilder();
        str.append("restaurant d = ").append(d)
                .append(". # tables: ").append(localRestaurants[d].getNumTables())
                .append(". # total customers: ").append(localRestaurants[d].getTotalNumCustomers()).append("\n");
        for (STable table : localRestaurants[d].getTables()) {
            str.append("--- table: ").append(table.toString()).append("\n");
        }
        return str.toString();
    }

    public void outputTopicTopWords(File outputFile, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            ArrayList<RankingItem<SNode>> rankChildren = new ArrayList<RankingItem<SNode>>();
            for (SNode child : node.getChildren()) {
                rankChildren.add(new RankingItem<SNode>(child, child.getRegressionParameter()));
            }
            Collections.sort(rankChildren);
            for (RankingItem<SNode> item : rankChildren) {
                stack.add(item.getObject());
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }
            if (node.getIterationCreated() >= MAX_ITER - LAG) {
                continue;
            }

            double[] nodeTopic = node.getTopic();
            String[] topWords = getTopWords(nodeTopic, numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.getIterationCreated())
                    .append("; ").append(node.getNumCustomers())
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.getRegressionParameter()))
                    .append(")");
            for (String topWord : topWords) {
                str.append(" ").append(topWord);
            }
            str.append("\n\n");
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(str.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topics "
                    + outputFile);
        }
    }
    
    public void outputHTML(
            File htmlFile,
            String[] docIds,
            String[][] rawSentences,
            int numSents,
            int numWords) throws Exception {
        if (verbose) {
            logln("--- Outputing result to HTML file " + htmlFile);
        }

        // rank sentences for each path
        HashMap<SNode, ArrayList<RankingItem<String>>> pathRankSentMap = getRankingSentences();

        StringBuilder str = new StringBuilder();
        str.append("<!DOCTYPE html>\n<html>\n");

        // header containing styles and javascript functions
        str.append("<head>\n");
        str.append("<link type=\"text/css\" rel=\"stylesheet\" href=\"http://argviz.umiacs.umd.edu/teaparty/framing.css\">\n"); // style
        str.append("<script type=\"text/javascript\" src=\"http://argviz.umiacs.umd.edu/teaparty/framing.js\"></script>\n"); // script
        str.append("</head>\n"); // end head

        // start body
        str.append("<body>\n");
        str.append("<table>\n");
        str.append("<tbody>\n");

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            ArrayList<RankingItem<SNode>> rankChildren = new ArrayList<RankingItem<SNode>>();
            for (SNode child : node.getChildren()) {
                rankChildren.add(new RankingItem<SNode>(child, child.getRegressionParameter()));
            }
            Collections.sort(rankChildren);
            for (RankingItem<SNode> rankChild : rankChildren) {
                stack.add(rankChild.getObject());
            }

            double[] nodeTopic = node.getTopic();
            String[] topWords = getTopWords(nodeTopic, numWords);

            if (node.getLevel() == 1) {
                str.append("<tr class=\"level").append(node.getLevel()).append("\">\n");
                str.append("<td>\n")
                        .append("[Topic ").append(node.getPathString())
                        .append("] ")
                        .append(" (")
                        .append(node.getNumChildren())
                        .append("; ").append(node.getNumCustomers())
                        .append("; ").append(node.getContent().getCountSum())
                        .append("; ").append(formatter.format(node.getRegressionParameter()))
                        .append(")");
                for (String topWord : topWords) {
                    str.append(" ").append(topWord);
                }
                str.append("</td>\n");
                str.append("</tr>\n");
            } else if (node.getLevel() == 2) {
                ArrayList<RankingItem<String>> rankSents = pathRankSentMap.get(node);
                if (rankSents == null || rankSents.size() < 10) {
                    continue;
                }
                Collections.sort(rankSents);

                // node info
                str.append("<tr class=\"level").append(node.getLevel()).append("\">\n");
                str.append("<td>\n")
                        .append("<a style=\"text-decoration:underline;color:blue;\" onclick=\"showme('")
                        .append(node.getPathString())
                        .append("');\" id=\"toggleDisplay\">")
                        .append("[Frame candidate ")
                        .append(node.getPathString())
                        .append("]</a>")
                        .append(" (").append(node.getNumCustomers())
                        .append("; ").append(node.getContent().getCountSum())
                        .append("; ").append(formatter.format(node.getRegressionParameter()))
                        .append(")");
                for (String topWord : topWords) {
                    str.append(" ").append(topWord);
                }
                str.append("</td>\n");
                str.append("</tr>\n");

                // sentences
                str.append("<tr class=\"level").append(L).append("\"")
                        .append(" id=\"").append(node.getPathString()).append("\"")
                        .append(" style=\"display:none;\"")
                        .append(">\n");
                str.append("<td>\n");

                for (int ii = 0; ii < Math.min(numSents, rankSents.size()); ii++) {
                    RankingItem<String> sent = rankSents.get(ii);
                    int d = Integer.parseInt(sent.getObject().split("-")[0]);
                    int s = Integer.parseInt(sent.getObject().split("-")[1]);

                    String debateId = docIds[d].substring(0, docIds[d].indexOf("_"));
                    if(debateId.startsWith("112-") || debateId.startsWith("111-")) {
                        debateId = debateId.substring(4);
                    }
                    str.append("<a href=\"")
                            .append("https://www.govtrack.us/data/us/112/cr/")
                            .append(debateId).append(".xml")
                            .append("\" ")
                            .append("target=\"_blank\">")
                            .append(docIds[d]).append("_").append(s)
                            .append("</a> ")
                            .append(rawSentences[d][s])
                            .append("<br/>\n");
                    str.append("</br>");
                }
                str.append("</td>\n</tr>\n");
            }
        }

        str.append("</tbody>\n");
        str.append("</table>\n");
        str.append("</body>\n");
        str.append("</html>");

        // output to file
        BufferedWriter writer = IOUtils.getBufferedWriter(htmlFile);
        writer.write(str.toString());
        writer.close();
    }
    
    private HashMap<SNode, ArrayList<RankingItem<String>>> getRankingSentences() {
        HashMap<SNode, ArrayList<RankingItem<String>>> pathRankSentMap =
                new HashMap<SNode, ArrayList<RankingItem<String>>>();
        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                SNode pathNode = table.getContent();
                ArrayList<RankingItem<String>> rankSents = pathRankSentMap.get(pathNode);
                if (rankSents == null) {
                    rankSents = new ArrayList<RankingItem<String>>();
                }
                for (int s : table.getCustomers()) {
                    if (words[d][s].length < 10) { // filter out too short sentences
                        continue;
                    }

                    double logprob = 0.0;
                    for (int n = 0; n < words[d][s].length; n++) {
                        if (z[d][s][n] == L - 1) {
                            logprob += pathNode.getLogProbability(words[d][s][n]);
                        }
                    }
                    if (logprob != 0) {
                        rankSents.add(new RankingItem<String>(d + "-" + s, logprob / words[d][s].length));
                    }
                }
                pathRankSentMap.put(pathNode, rankSents);
            }
        }
        return pathRankSentMap;
    }

//    private void testSampler(int[][][] newWords, int[] newAuthors) {
//        if (verbose) {
//            logln("Test sampling ...");
//        }
//        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
//        if (!reportFolder.exists()) {
//            throw new RuntimeException("Report folder does not exist");
//        }
//        String[] filenames = reportFolder.list();
//
//        File iterPredFolder = new File(getSamplerFolderPath(), IterPredictionFolder);
//        IOUtils.createFolder(iterPredFolder);
//
//        try {
//            for (int i = 0; i < filenames.length; i++) {
//                String filename = filenames[i];
//                if (!filename.contains("zip")) {
//                    continue;
//                }
//
//                File partialResultFile = new File(iterPredFolder, IOUtils.removeExtension(filename) + ".txt");
//                sampleNewDocuments(
//                        new File(reportFolder, filename),
//                        newWords, newAuthors,
//                        partialResultFile.getAbsolutePath());
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while sampling during test time.");
//        }
//    }
//
//    private void sampleNewDocuments(
//            File stateFile,
//            int[][][] newWords,
//            int[] newAuthors,
//            String outputResultFile) throws Exception {
//        if (verbose) {
//            logln("\nPerform regression using model from " + stateFile);
//        }
//
//        try {
//            inputModel(stateFile.getAbsolutePath());
//        } catch (Exception e) {
//            e.printStackTrace();
//            System.exit(1);
//        }
//
//
//        words = newWords;
//        responses = null; // for evaluation
//        D = words.length;
//
//        sentCount = 0;
//        tokenCount = 0;
//        docTokenCounts = new int[D];
//        for (int d = 0; d < D; d++) {
//            sentCount += words[d].length;
//            for (int s = 0; s < words[d].length; s++) {
//                tokenCount += words[d][s].length;
//                docTokenCounts[d] += words[d][s].length;
//            }
//        }
//
//        logln("--- V = " + V);
//        logln("--- # documents = " + D); // number of groups
//        logln("--- # sentences = " + sentCount);
//        logln("--- # tokens = " + tokenCount);
//
//        // initialize structure for test data
//        initializeDataStructure();
//
//        if (verbose) {
//            logln("Initialized data structure");
//            logln(printGlobalTreeSummary());
//            logln(printLocalRestaurantSummary());
//        }
//
//        // initialize random assignments
//        initializeRandomAssignmentsNewDocuments();
//
//        updateDocumentTopicWeights();
//        updateDocumentLexicalWeights();
//
//        if (verbose) {
//            logln("Initialized random assignments");
//            logln(printGlobalTreeSummary());
//            logln(printLocalRestaurantSummary());
//        }
//
//        if (debug) {
//            validateAssignments("Initialized");
//        }
//
//        // iterate
//        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
//        for (iter = 0; iter < MAX_ITER; iter++) {
//            for (int d = 0; d < D; d++) {
//                for (int s = 0; s < words[d].length; s++) {
//                    if (words[d].length > 1) // if this document has only 1 sentence, no sampling is needed
//                    {
//                        sampleTableForSentence(d, s, REMOVE, ADD, !OBSERVED, !EXTEND);
//                    }
//
//                    for (int n = 0; n < words[d][s].length; n++) {
//                        sampleLevelForToken(d, s, n, REMOVE, ADD, !OBSERVED);
//                    }
//                }
//
//                for (STable table : localRestaurants[d].getTables()) {
//                    samplePathForTable(d, table, REMOVE, ADD, !OBSERVED, !EXTEND);
//                }
//            }
//
//            if (verbose && iter % LAG == 0) {
//                logln("--- iter = " + iter + " / " + MAX_ITER);
//            }
//
//            if (iter >= BURN_IN && iter % LAG == 0) {
//                this.updateDocumentLexicalWeights();
//                this.updateDocumentTopicWeights();
//
//                double[] predResponses = getRegressionValues();
//                predResponsesList.add(predResponses);
//            }
//        }
//
//        // output result during test time 
//        BufferedWriter writer = IOUtils.getBufferedWriter(outputResultFile);
//        for (int d = 0; d < D; d++) {
//            writer.write(Integer.toString(d));
//
//            for (int ii = 0; ii < predResponsesList.size(); ii++) {
//                writer.write("\t" + predResponsesList.get(ii)[d]);
//            }
//            writer.write("\n");
//        }
//        writer.close();
//    }
//
//    private void initializeRandomAssignmentsNewDocuments() {
//        if (verbose) {
//            logln("--- Initializing random assignments ...");
//        }
//
//        for (int d = 0; d < D; d++) {
//            for (int s = 0; s < words[d].length; s++) {
//                // create a new table for each sentence
//                STable table = new STable(iter, s, null, d,
//                        new TruncatedStickBreaking(L, hyperparams.get(GEM_MEAN),
//                        hyperparams.get(GEM_SCALE)));
//                localRestaurants[d].addTable(table);
//                localRestaurants[d].addCustomerToTable(s, table.getIndex());
//                c[d][s] = table;
//
//                // initialize all tokens at the leaf node first
//                for (int n = 0; n < words[d][s].length; n++) {
//                    z[d][s][n] = L - 1;
//                    table.incrementLevelCount(z[d][s][n]);
//                    sentLevelCounts[d][s][z[d][s][n]]++;
//                }
//            }
//        }
//
//        for (int d = 0; d < D; d++) {
//            for (STable table : localRestaurants[d].getTables()) {
//                samplePathForTable(d, table, !REMOVE, ADD, !OBSERVED, !EXTEND);
//            }
//        }
//    }

    class SNode extends TopicTreeNode<SNode, DirMult> {

        private final int born;
        private int numCustomers;
        private double regression;
//        private HashMap<Integer, DirMult> switches;
        private DirMult gate;

        SNode(int iter, int index, int level,
                DirMult content,
                double regParam,
                SNode parent) {
            super(index, level, content, parent);
            this.born = iter;
            this.numCustomers = 0;
            this.regression = regParam;
//            this.switches = new HashMap<Integer, DirMult>();
            this.gate = emptySwitch.clone();
        }

        public int getIterationCreated() {
            return this.born;
        }

        public double getSwitchLogLikelihood() {
            double llh = 0.0;
//            for (int d : this.switches.keySet()) {
//                llh += this.switches.get(d).getLogLikelihood();
//            }
            llh = this.gate.getLogLikelihood();
            return llh;
        }

        public double getSwitchLogProbability(int type) {
//            DirMult swch = this.switches.get(d);
//            if (gate != null) {
                return gate.getLogLikelihood(type);
//            } else {
//                return emptySwitch.getLogLikelihood(type);
//            }
        }

        public double getSwitchLogProbability(HashMap<Integer, Integer> typeCounts) {
//            if (gate != null) {
                return gate.getLogLikelihood(typeCounts);
//            } else {
//                return emptySwitch.getLogLikelihood(typeCounts);
//            }
        }

        public DirMult getSwitch() {
            return this.gate;
        }

//        public void setSwitch(int d, DirMult s) {
//            this.switches.put(d, s);
//        }

        public int getCount(int type) {
            return this.gate.getCount(type);
        }

        public void decrementCount(int type) {
            this.changeCount(type, -1);
        }

        public void incrementCount(int type) {
            this.changeCount(type, 1);
        }

        public void changeCount(int type, int delta) {
//            DirMult swch = this.switches.get(d);
//            if (swch == null) {
//                swch = emptySwitch.clone();
//            }
            gate.changeCount(type, delta);

            if (gate.getCount(type) < 0) {
                throw new RuntimeException("Negative count " + gate.getCount(type));
            }

//            if (swch.isEmpty()) {
//                this.switches.remove(d);
//            } else {
//                this.switches.put(d, swch);
//            }
        }

        /**
         * Get the log probability of a set of observations given the topic at
         * this node.
         *
         * @param obs The set of observations
         */
        public double getLogProbability(SparseCount obs) {
            if (this.getTopic() == null) {
                return this.content.getLogLikelihood(obs.getObservations());
            } else {
                double val = 0.0;
                for (int o : obs.getIndices()) {
                    val += obs.getCount(o) * this.getLogProbability(o);
                }
                return val;
            }
        }

        double getLogJointProbability(double gamma) {
            ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
            for (SNode child : this.getChildren()) {
                numChildrenCusts.add(child.getNumCustomers());
            }
            return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gamma);
        }

        public double getRegressionParameter() {
            return this.regression;
        }

        public void setRegressionParameter(double reg) {
            this.regression = reg;
        }

        public int getNumCustomers() {
            return this.numCustomers;
        }

        public void decrementNumCustomers() {
            this.numCustomers--;
        }

        public void incrementNumCustomers() {
            this.numCustomers++;
        }

        public void changeNumCustomers(int delta) {
            this.numCustomers += delta;
        }

        public boolean isEmpty() {
            return this.numCustomers == 0;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(getPathString())
                    .append(" (").append(born).append(")")
                    .append(" #ch = ").append(getNumChildren())
                    .append(", #c = ").append(getNumCustomers())
                    .append(", #o = ").append(getContent().getCountSum())
                    .append(", reg = ").append(MiscUtils.formatDouble(regression))
                    .append("]");
            return str.toString();
        }

        void validate(String msg) {
            int maxChildIndex = SHLDA.PSEUDO_NODE_INDEX;
            for (SNode child : this.getChildren()) {
                if (maxChildIndex < child.getIndex()) {
                    maxChildIndex = child.getIndex();
                }
            }

            for (int i = 0; i < maxChildIndex; i++) {
                if (!inactiveChildren.contains(i) && !isChild(i)) {
                    throw new RuntimeException(msg + ". Child inactive indices"
                            + " have not been updated. Node: " + this.toString()
                            + ". Index " + i + " is neither active nor inactive");
                }
            }
        }
    }

    class STable extends FullTable<Integer, SNode> {

        private final int born;
        private final int restIndex;

        public STable(int iter, int index,
                SNode content, int restId) {
            super(index, content);
            this.born = iter;
            this.restIndex = restId;
        }

        public int getRestaurantIndex() {
            return this.restIndex;
        }

        public boolean containsCustomer(int c) {
            return this.customers.contains(c);
        }

        public int getIterationCreated() {
            return this.born;
        }

        public String getTableId() {
            return restIndex + ":" + index;
        }

        @Override
        public int hashCode() {
            String hashCodeStr = getTableId();
            return hashCodeStr.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if ((obj == null) || (this.getClass() != obj.getClass())) {
                return false;
            }
            STable r = (STable) (obj);

            return r.index == this.index
                    && r.restIndex == this.restIndex;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(getTableId())
                    .append(", ").append(born)
                    .append(", ").append(getNumCustomers())
                    .append("]")
                    .append(" >> ").append(getContent() == null ? "null" : getContent().toString());
            return str.toString();
        }
    }
}
