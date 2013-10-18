package sampler.supervised.regression;

import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import optimization.GurobiMultipleLinearRegression;
import sampling.likelihood.DirichletMultinomialModel;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.Table;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class LabeledSHLDASampler extends AbstractSampler {

    public static final String IterPredictionFolder = "iter-predictions/";
    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    public static final int ALPHA = 0;
    public static final int RHO = 1;
    public static final int GEM_MEAN = 2;
    public static final int GEM_SCALE = 3;
    public static final int TAU_MEAN = 4;
    public static final int TAU_SCALE = 5;
    // options
    protected boolean supervised = true;
    protected boolean labeled;
    protected boolean lexicalRegression;
    protected boolean optimizeLexicalWeights = false;
    // parameters
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int K; // number of labels
    // hyperparameters
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;    // regression parameter means
    protected double[] sigmas; // regression parameter variances
    private double logAlpha;
    private double sqrtRho;
    private double[] sqrtSigmas;
    private double[] logGammas;
    // input data
    protected int[][][] words;  // [D] x [S_d] x [N_ds]: words
    protected double[] responses; // [D]
    protected int[][] labels; // [D] x [T_d]
    private ArrayList<String> labelVocab;
    // input statistics
    private int sentCount;
    private int tokenCount;
    private int[] docTokenCounts;
    // assignments
    private STable[][] c; // path assigned to sentences
    private int[][][] z; // level assigned to tokens
    // state structure
    private SNode globalTreeRoot; // tree
    private SNode[] firstLevelNodes; // first level nodes
    private Restaurant<STable, Integer, SNode>[] localRestaurants; // franchise
    private TruncatedStickBreaking[] docLevelDists; // doc sticks
    // lexical 
    private double[] lexicalWeights; // background lexical
    private int numLexicalItems;
    private ArrayList<Integer> lexicalIndices;
    private double[][] docLexicalDesignMatrix;
    // state statistics stored
    private SparseCount[][] sentLevelCounts;
    private double[] docLexicalWeights;
    private double[] docTopicWeights;
    // over time
    private ArrayList<double[]> lexicalWeightsOverTime;
    // auxiliary
    private double[] uniform;
    private DirichletMultinomialModel[] emptyModels;
    private int numTokenAssignmentsChange;
    private int numSentAssignmentsChange;
    private int numTableAssignmentsChange;

    public void setLabelVocab(ArrayList<String> labelVoc) {
        this.labelVocab = labelVoc;
        this.K = this.labelVocab.size();
    }

    /**
     * Set number of lexical items considered.
     *
     * @param Number of lexical items having non-zero regression parameters
     */
    public void setNumLexicalItems(int numLexItems) {
        this.numLexicalItems = numLexItems;
    }

    /**
     * Weights for all lexical items (i.e., words)
     *
     * @param lexWeights V-dim vector containing lexical weights
     */
    public void setLexicalWeights(double[] lexWeights) {
        this.lexicalWeights = lexWeights;
    }

    public void configure(String folder,
            int[][][] words,
            double[] responses,
            int[][] labels,
            int V, int L,
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
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.words = words;
        this.responses = responses;
        this.labels = labels;

        this.V = V;
        this.L = L;
        this.D = this.words.length;

        // statistics
        sentCount = 0;
        tokenCount = 0;
        docTokenCounts = new int[D];
        for (int d = 0; d < D; d++) {
            sentCount += words[d].length;
            for (int s = 0; s < words[d].length; s++) {
                tokenCount += words[d][s].length;
                docTokenCounts[d] += words[d][s].length;
            }
        }

        // hyperparameters
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

        if (lexicalWeights != null) { // if the lexical weights are given
            this.filterLexicalItems();
        } else {
            this.initializeLexicalWeights();
        }

        this.docLexicalDesignMatrix = new double[D][this.numLexicalItems];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    int idx = this.lexicalIndices.indexOf(words[d][s][n]);
                    if (idx != -1) {
                        docLexicalDesignMatrix[d][idx]++;
                    }
                }
            }

            for (int ii = 0; ii < this.lexicalIndices.size(); ii++) {
                docLexicalDesignMatrix[d][ii] /= docTokenCounts[d];
            }
        }

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

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

        int numLabeledDocs = 0;
        int[] labelCounts = new int[K];
        for (int d = 0; d < D; d++) {
            if (labels[d].length > 0) {
                numLabeledDocs++;

                for (int ii = 0; ii < labels[d].length; ii++) {
                    labelCounts[labels[d][ii]]++;
                }
            }
        }

        if (verbose) {
            logln("--- V = " + V);
            logln("--- # documents = " + D); // number of groups
            logln("--- # sentences = " + sentCount);
            logln("--- # tokens = " + tokenCount);
            logln("--- # labeled docs = " + numLabeledDocs);
            logln("--- label histogram:");
            for (int kk = 0; kk < K; kk++) {
                logln("--- --- " + kk
                        + "\t" + labelCounts[kk]
                        + " (" + labelVocab.get(kk) + ")");
            }

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
            logln("--- initialize:\t" + initState);
        }
    }

    public void priorTopics(int[][] docs, int[][] tags) {
    }

    private void filterLexicalItems() {
        this.lexicalIndices = new ArrayList<Integer>();
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < V; v++) {
            rankItems.add(new RankingItem<Integer>(v, lexicalWeights[v]));
        }
        Collections.sort(rankItems);

        for (int i = 0; i < numLexicalItems / 2; i++) {
            this.lexicalIndices.add(rankItems.get(i).getObject());
        }
        for (int i = 0; i < numLexicalItems / 2; i++) {
            this.lexicalIndices.add(rankItems.get(V - 1 - i).getObject());
        }

        for (int v = 0; v < V; v++) {
            int idx = this.lexicalIndices.indexOf(v);
            if (idx == -1) {
                this.lexicalWeights[v] = 0.0;
            }
        }
    }

    private void initializeLexicalWeights() {
        if (verbose) {
            logln("Initializing lexical weights ...");
        }

        // flatten the input documents' words
        ArrayList<Integer>[] docWords = new ArrayList[D];
        for (int d = 0; d < D; d++) {
            docWords[d] = new ArrayList<Integer>();
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docWords[d].add(words[d][s][n]);
                }
            }
        }

        // compute tf-idf's
        HashMap<Integer, Integer> tfs = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> dfs = new HashMap<Integer, Integer>();
        for (int v = 0; v < V; v++) {
            tfs.put(v, 0);
            dfs.put(v, 0);
        }
        for (int d = 0; d < D; d++) {
            Set<Integer> docUniqueTerms = new HashSet<Integer>();
            for (int n = 0; n < docWords[d].size(); n++) {
                int token = docWords[d].get(n);
                docUniqueTerms.add(token);

                Integer tf = tfs.get(token);
                tfs.put(token, tf + 1);
            }

            for (int token : docUniqueTerms) {
                Integer df = dfs.get(token);
                dfs.put(token, df + 1);
            }
        }

        int maxTf = 0;
        for (int type : tfs.keySet()) {
            if (maxTf < tfs.get(type)) {
                maxTf = tfs.get(type);
            }
        }

        ArrayList<RankingItem<Integer>> rankWords = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < V; v++) {
            double tf = 0.5 + 0.5 * tfs.get(v) / maxTf;
            double idf = Math.log(D) - Math.log(dfs.get(v) + 1);
            double tf_idf = tf * idf;

            rankWords.add(new RankingItem<Integer>(v, tf_idf));
        }
        Collections.sort(rankWords);

        // only keep low tf-idf lexical items
        lexicalIndices = new ArrayList<Integer>();
        for (int i = 0; i < this.numLexicalItems; i++) {
            lexicalIndices.add(rankWords.get(rankWords.size() - 1 - i).getObject());
        }

        // optimize
        double[][] designMatrix = new double[D][numLexicalItems];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < docWords[d].size(); n++) {
                int featIndex = lexicalIndices.indexOf(docWords[d].get(n));
                if (featIndex == -1) {
                    continue;
                }
                designMatrix[d][featIndex]++;
            }
        }

        this.lexicalWeights = new double[V];
        double lambda = 1.0 / hyperparams.get(TAU_SCALE);

        if (verbose) {
            logln("--- Start running gurobi ...");
        }
        GurobiMultipleLinearRegression lasso = new GurobiMultipleLinearRegression(
                designMatrix, responses, lambda);
        double[] weights = lasso.solve();
        for (int ii = 0; ii < weights.length; ii++) {
            int v = lexicalIndices.get(ii);
            this.lexicalWeights[v] = weights[ii];
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
                    this.docLexicalWeights[d] += this.lexicalWeights[words[d][s][n]];
                }
            }
        }
    }

    public boolean isOptimizingLexicalWeights() {
        return this.optimizeLexicalWeights;
    }

    public void setOptimizingLexicalWeights(boolean opt) {
        this.optimizeLexicalWeights = opt;
    }

    public void setSupervised(boolean s) {
        this.supervised = s;
    }

    public boolean isSupervised() {
        return this.supervised;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_labeled-SHLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
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
        this.name = str.toString();
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
        updateDocumentTopicWeights();
        updateDocumentLexicalWeights();

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

    private void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }
        int rootLevel = 0;
        int rootIndex = 0;
        DirichletMultinomialModel dmModel = new DirichletMultinomialModel(V, betas[rootLevel], uniform);
        double regParam = 0.0;
        this.globalTreeRoot = new SNode(iter, rootIndex, rootLevel, dmModel, regParam, null);

        this.firstLevelNodes = new SNode[this.K];
        for (int kk = 0; kk < K; kk++) {
            this.firstLevelNodes[kk] = this.createNode(globalTreeRoot);
            if (this.firstLevelNodes[kk].getIndex() != kk) {
                throw new RuntimeException("Mismatched indices");
            }
        }

        this.emptyModels = new DirichletMultinomialModel[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirichletMultinomialModel(V, betas[l + 1], uniform);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<STable, Integer, SNode>();
        }

        this.docLevelDists = new TruncatedStickBreaking[D];
        for (int d = 0; d < D; d++) {
            this.docLevelDists[d] = new TruncatedStickBreaking(L,
                    hyperparams.get(GEM_MEAN), hyperparams.get(GEM_SCALE));
        }

        this.sentLevelCounts = new SparseCount[D][];
        for (int d = 0; d < D; d++) {
            this.sentLevelCounts[d] = new SparseCount[words[d].length];
            for (int s = 0; s < words[d].length; s++) {
                this.sentLevelCounts[d][s] = new SparseCount();
            }
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

    private void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- --- Initializing random assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // create a new table for each sentence
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // randomly choose a label
                int label;
                if (labels[d].length > 0) {
                    label = labels[d][rand.nextInt(labels[d].length)];
                } else {
                    label = rand.nextInt(K);
                }

                // create a new path for each table
                SNode node = globalTreeRoot.getChild(label);
                while (node.getLevel() != L - 1) {
                    node = createNode(node);
                }
                addTableToPath(node);
                table.setContent(node);

                // sample level
                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelForToken(d, s, n, !REMOVE, ADD, !OBSERVED);
                }

//                System.out.println("d = " + d
//                        + "\ts=" + s
//                        + "\t" + MiscUtils.arrayToString(labels[d])
//                        + "\t" + printGlobalTreeSummary());

                if (d > 0 || s > 0) {
                    sampleTableForSentence(d, s, REMOVE, ADD, !OBSERVED, EXTEND);
                }

                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelForToken(d, s, n, REMOVE, ADD, !OBSERVED);
                }
            }
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
            if (!repFolderPath.exists()) {
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
            System.arraycopy(this.lexicalWeights, 0, storeWeights, 0, V);
            this.lexicalWeightsOverTime.add(storeWeights);

            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens change: " + numTokenAssignmentsChange
                            + "\t # sents change: " + numSentAssignmentsChange
                            + "\t # tables change: " + numTableAssignmentsChange
                            + "\n" + getCurrentState()
                            + "\n");
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens change: " + numTokenAssignmentsChange
                            + "\t # sents change: " + numSentAssignmentsChange
                            + "\t # tables change: " + numTableAssignmentsChange
                            + "\n" + getCurrentState()
                            + "\n");
                }
            }

            numTableAssignmentsChange = 0;
            numSentAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;

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

            if (isSupervised()) {
                optimizeTopicRegressionParameters();
            }

            if (isOptimizingLexicalWeights()) {
                // not perform 
            }

            if (verbose && isSupervised()) {
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

            System.out.println();

            // store model
            if (iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(repFolderPath, "iter-" + iter + ".zip"));
                try {
                    outputTopicTopWords(new File(repFolderPath,
                            "iter-" + iter + "-top-words.txt"), 15);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            }
        }

        // output final model
        outputState(new File(repFolderPath, "iter-" + iter + ".zip"));

        if (verbose) {
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                        "hyperparameters.txt").getAbsolutePath());
            }

            BufferedWriter writer = IOUtils.getBufferedWriter(
                    new File(getSamplerFolderPath(), "weights.txt"));
            for (int v = 0; v < V; v++) {
                writer.write(v + "\t" + wordVocab.get(v));
                for (int i = 0; i < this.lexicalWeightsOverTime.size(); i++) {
                    writer.write("\t" + this.lexicalWeightsOverTime.get(i)[v]);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public double[] getRegressionValues() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = docTopicWeights[d] + docLexicalWeights[d];
            regValues[d] = sum / docTokenCounts[d];
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
    private void addTableToPath(SNode leafNode) {
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
    private SNode removeTableFromPath(SNode leafNode) {
        SNode retNode = leafNode;
        SNode node = leafNode;
        while (node != null) {
            node.decrementNumCustomers();
            if (node.isEmpty() && node.getLevel() > 1) { // delete non-first-level empty nodes
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
    private SNode[] addObservationsToPath(SNode leafNode, HashMap<Integer, Integer>[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], observations[l]);
        }
        return path;
    }

    /**
     * Remove a set of observations (given their level assignments) from a path
     *
     * @param leafNode The leaf node identifying the path
     * @param observations The observations per level
     */
    private SNode[] removeObservationsFromPath(SNode leafNode, HashMap<Integer, Integer>[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], observations[l]);
        }
        return path;
    }

    /**
     * Remove a set of observations from a node
     *
     * @param node The node
     * @param observations The set of observations
     */
    private void removeObservationsFromNode(SNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, -count);
        }
    }

    /**
     * Add a set of observations to a node
     *
     * @param node The node
     * @param observations The set of observations
     */
    private void addObservationsToNode(SNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, count);
        }
    }

    /**
     * Create new path from an existing internal node
     *
     * @param internalNode The internal node
     */
    private SNode createNewPath(SNode internalNode) {
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
    private SNode createNode(SNode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirichletMultinomialModel dmm = new DirichletMultinomialModel(V, betas[level], uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SNode child = new SNode(iter, nextChildIndex, level, dmm, regParam, parent);
        return parent.addChild(nextChildIndex, child);
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
    private void sampleTableForSentence(int d, int s,
            boolean remove, boolean add,
            boolean observed, boolean extend) {
        STable curTable = c[d][s];

        // empirical level assignments of the current sentence
        HashMap<Integer, Integer>[] sentObsCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            sentObsCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[d][s].length; n++) {
            int type = words[d][s][n];
            int level = z[d][s][n];
            Integer count = sentObsCountPerLevel[level].get(type);
            if (count == null) {
                sentObsCountPerLevel[level].put(type, 1);
            } else {
                sentObsCountPerLevel[level].put(type, count + 1);
            }
        }

        // debug
//        boolean condition = d == 19 && s == 0;
//        if (condition) {
//            double topicWeight = 0.0;
//            for (int ss = 0; ss < words[d].length; ss++) {
//                topicWeight += computeTopicWeight(d, ss);
//            }
//
//            logln("1. iter = " + iter
//                    + " d = " + d
//                    + " s = " + s
//                    + ". docTopicWeight = " + docTopicWeights[d]
//                    + ". true = " + topicWeight);
//        }

        if (observed) {
            this.docTopicWeights[d] -= computeTopicWeight(d, s);
        }

        if (remove) {
            removeObservationsFromPath(c[d][s].getContent(), sentObsCountPerLevel);
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
                wordLlh += path[l].getContent().getLogLikelihood(sentObsCountPerLevel[l]);
            }

            double resLlh = 0.0;
            if (observed) {
                double addTopicWeight = 0.0;
                for (int l = 0; l < L; l++) {
                    addTopicWeight += path[l].getRegressionParameter() * sentLevelCounts[d][s].getCount(l);
                }

                double mean = (docTopicWeights[d] + docLexicalWeights[d] + addTopicWeight) / docTokenCounts[d];
                resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, sqrtRho);
            }

            double lp = logprior + wordLlh + resLlh;
            logProbs.add(lp);
            tableIndices.add(table.getIndex());

            // debug
//            if (condition) {
//                logln("iter = " + iter + ". d = " + d + ". s = " + s
//                        + ". table: " + table.toString()
//                        + ". log prior = " + MiscUtils.formatDouble(logprior)
//                        + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                        + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                        + ". lp = " + MiscUtils.formatDouble(lp));
//            }
        }

        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (extend) {
            // log priors
            computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0, labels[d]);

            // word log likelihoods
            double[] dataLlhNewTopic = new double[L];
            for (int l = 1; l < L; l++) // skip the root
            {
                dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(sentObsCountPerLevel[l]);
            }
            computePathWordLogLikelihood(pathWordLlhs,
                    globalTreeRoot,
                    sentObsCountPerLevel,
                    dataLlhNewTopic,
                    0.0,
                    labels[d]);

            // debug
            if (pathLogPriors.size() != pathWordLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch. "
                        + pathLogPriors.size() + " vs. "
                        + pathWordLlhs.size());
            }

//            if (condition) {
//                logln("--- logprior size = " + pathLogPriors.size()
//                        + ". logwordlh size = " + pathWordLlhs.size());
//            }

            // response log likelihoods
            if (supervised && observed) {
                pathResLlhs = computePathResponseLogLikelihood(d, s, labels[d]);

                if (pathLogPriors.size() != pathResLlhs.size()) {
                    for (SNode node1 : pathLogPriors.keySet()) {
                        System.out.println("+++ " + node1.toString());
                    }
                    for (SNode node2 : pathResLlhs.keySet()) {
                        System.out.println("--- " + node2.toString());
                    }
                    throw new RuntimeException("Numbers of paths mismatch. "
                            + pathLogPriors.size() + " vs. "
                            + pathResLlhs.size());
                }
            }

//            if (condition) {
//                logln("--- logreslh size = " + pathResLlhs.size());
//            }

            double logPrior = logAlpha;
            double marginals = computeMarginals(pathLogPriors, pathWordLlhs, pathResLlhs, observed);

            double lp = logPrior + marginals;
            logProbs.add(lp);
            tableIndices.add(PSEUDO_TABLE_INDEX);

            // debug
//            logln("iter = " + iter + ". d = " + d + ". s = " + s
//                    + ". new table"
//                    + ". log prior = " + MiscUtils.formatDouble(logPrior)
//                    + ". marginal = " + MiscUtils.formatDouble(marginals)
//                    + ". lp = " + MiscUtils.formatDouble(lp));
        }

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        int tableIdx = tableIndices.get(sampledIndex);

        // debug
//        logln(">>> idx = " + sampledIndex + ". tabIdx = " + tableIdx + "\n");

        if (curTable != null && curTable.getIndex() != tableIdx) {
            numSentAssignmentsChange++;
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

        c[d][s] = table;

        if (add) {
            addObservationsToPath(table.getContent(), sentObsCountPerLevel);
            localRestaurants[d].addCustomerToTable(s, table.getIndex());
        }

        if (observed) {
            docTopicWeights[d] += computeTopicWeight(d, s);
        }

        // debug
//        if(iter > 0){
//            double topicWeight = 0.0;
//            for(int ss=0; ss<words[d].length; ss++)
//                topicWeight += computeTopicWeight(d, ss);
//            
//            if(Math.abs(docTopicWeights[d] - topicWeight) > 0.001)
//                logln("2. iter = " + iter
//                    + " d = " + d
//                    + " s = " + s
//                    + ". docTopicWeight = " + docTopicWeights[d]
//                    + ". true = " + topicWeight);
//        }
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
    private void sampleLevelForToken(int d, int s, int n,
            boolean remove, boolean add,
            boolean observed) {
        STable curTable = c[d][s];
        SNode[] curPath = getPathFromNode(curTable.getContent());

        if (observed) {
            docTopicWeights[d] -= curPath[z[d][s][n]].getRegressionParameter();
        }

        if (remove) {
            docLevelDists[d].decrement(z[d][s][n]);
            sentLevelCounts[d][s].decrement(z[d][s][n]);
            curPath[z[d][s][n]].getContent().decrement(words[d][s][n]);
        }

        double[] logprobs = new double[L];
        for (int l = 0; l < L; l++) {
            double logPrior = docLevelDists[d].getLogProbability(l);
            double wordLlh = curPath[l].getContent().getLogLikelihood(words[d][s][n]);
            double resLlh = 0.0;
            if (observed) {
                double sum = docTopicWeights[d] + docLexicalWeights[d]
                        + curPath[l].getRegressionParameter();
                double mean = sum / docTokenCounts[d];
                resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, sqrtRho);
            }
            logprobs[l] = logPrior + wordLlh + resLlh;

            // debug
//            logln("iter = " + iter + ". " + d + ":" + s + ":" + n
//                    + ". l = " + l + ". count = " + docLevelDists[d].getCount(l)
//                    + ". log prior = " + MiscUtils.formatDouble(logPrior)
//                    + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                    + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                    + ". lp = " + MiscUtils.formatDouble(logprobs[l]));
        }

        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);

        // debug
//        logln("--->>> sampled level = " + sampledL + "\n");

        if (z[d][s][n] != sampledL) {
            numTokenAssignmentsChange++;
        }

        // update and increment
        z[d][s][n] = sampledL;

        if (add) {
            docLevelDists[d].increment(z[d][s][n]);
            sentLevelCounts[d][s].increment(z[d][s][n]);
            curPath[z[d][s][n]].getContent().increment(words[d][s][n]);
        }

        if (observed) {
            docTopicWeights[d] += curPath[z[d][s][n]].getRegressionParameter();
        }

        // debug
//        if(iter > 0){
//            double topicWeight = 0.0;
//            for(int ss=0; ss<words[d].length; ss++)
//                topicWeight += computeTopicWeight(d, ss);
//            
//            if(Math.abs(docTopicWeights[d] - topicWeight) > 0.001)
//                logln("2. iter = " + iter
//                        + " d = " + d
//                        + " s = " + s
//                        + " n = " + n
//                        + ". docTopicWeight = " + docTopicWeights[d]
//                        + ". true = " + topicWeight);
//        }
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
    private void samplePathForTable(
            int d,
            STable table,
            boolean remove, boolean add,
            boolean observed, boolean extend) {
        SNode curLeaf = table.getContent();

        // observations of sentences currently being assign to this table
        HashMap<Integer, Integer>[] obsCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            obsCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int s : table.getCustomers()) {
            for (int n = 0; n < words[d][s].length; n++) {
                int level = z[d][s][n];
                int obs = words[d][s][n];

                Integer count = obsCountPerLevel[level].get(obs);
                if (count == null) {
                    obsCountPerLevel[level].put(obs, 1);
                } else {
                    obsCountPerLevel[level].put(obs, count + 1);
                }
            }
        }

        // data likelihood for new nodes at each level
        double[] dataLlhNewTopic = new double[L];
        for (int l = 1; l < L; l++) // skip the root
        {
            dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(obsCountPerLevel[l]);
        }

//        boolean condition = false;
//        if(condition){
//            logln("iter = " + iter + ". d = " + d + ". tabIdx = " + table.getTableId());
//            logln(printGlobalTree());
//            logln(printLocalRestaurant(d));
//        }

        if (observed) {
            for (int s : table.getCustomers()) {
                docTopicWeights[d] -= computeTopicWeight(d, s);
            }
        }

        if (remove) {
            removeObservationsFromPath(table.getContent(), obsCountPerLevel);
            removeTableFromPath(table.getContent());
        }

//        if(condition){
//            logln("After remove. iter = " + iter + ". d = " + d + ". tabIdx = " + table.getTableId());
//            logln(printGlobalTree());
//            logln(printLocalRestaurant(d));
//        }

        // log priors
        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0, labels[d]);

        // word log likelihoods
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        computePathWordLogLikelihood(pathWordLlhs,
                globalTreeRoot,
                obsCountPerLevel,
                dataLlhNewTopic,
                0.0,
                labels[d]);

        // debug
        if (pathLogPriors.size() != pathWordLlhs.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        // response log likelihoods
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (supervised && observed) {
            pathResLlhs = computePathResponseLogLikelihood(d, table, labels[d]);

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
            if (supervised && observed) {
                lp += pathResLlhs.get(path);
            }

            logProbs.add(lp);
            pathList.add(path);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode newLeaf = pathList.get(sampledIndex);

        // debug
        if (curLeaf == null || curLeaf.equals(newLeaf)) {
            numTableAssignmentsChange++;
        }

        // if pick an internal node, create the path from the internal node to leave
        if (newLeaf.getLevel() < L - 1) {
            newLeaf = this.createNewPath(newLeaf);
        }

        // update
        table.setContent(newLeaf);

        if (add) {
            addTableToPath(newLeaf);
            addObservationsToPath(newLeaf, obsCountPerLevel);
        }

        if (observed) {
            for (int s : table.getCustomers()) {
                docTopicWeights[d] += computeTopicWeight(d, s);
            }
        }

        // debug
//        if(iter > 0){
//            double topicWeight = 0.0;
//            for(int ss=0; ss<words[d].length; ss++)
//                topicWeight += computeTopicWeight(d, ss);
//            
//            if(Math.abs(docTopicWeights[d] - topicWeight) > 0.001)
//                logln("2. iter = " + iter
//                        + " d = " + d
//                        + " table = " + table.toString()
//                        + ". docTopicWeight = " + docTopicWeights[d]
//                        + ". true = " + topicWeight);
//        }
    }

    private void optimizeTopicRegressionParameters() {
        ArrayList<SNode> flattenTree = flattenTreeWithoutRoot();
        int numNodes = flattenTree.size();

        double[] lambdas = new double[numNodes];
        HashMap<SNode, Integer> nodeIndices = new HashMap<SNode, Integer>();
        for (int i = 0; i < flattenTree.size(); i++) {
            SNode node = flattenTree.get(i);
            nodeIndices.put(node, i);
            lambdas[i] = 1.0 / sigmas[node.getLevel()];
        }

        // design matrix
        double[][] designMatrix = new double[D][numNodes];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 1; l < L; l++) {
                    int nodeIdx = nodeIndices.get(path[l]);
                    int count = sentLevelCounts[d][s].getCount(l);
                    designMatrix[d][nodeIdx] += count;
                }
            }

            for (int i = 0; i < numNodes; i++) {
                designMatrix[d][i] /= docTokenCounts[d];
            }
        }

        // adjusted response vector
        double[] responseVector = new double[D];
        for (int d = 0; d < D; d++) {
            responseVector[d] = responses[d] - docLexicalWeights[d] / docTokenCounts[d];
        }

        GurobiMultipleLinearRegression mlr =
                new GurobiMultipleLinearRegression(designMatrix, responseVector, lambdas);
        double[] weights = mlr.solve();

        // update
        for (int i = 0; i < numNodes; i++) {
            flattenTree.get(i).setRegressionParameter(weights[i]);
        }
        updateDocumentTopicWeights();
    }

    private ArrayList<SNode> flattenTreeWithoutRoot() {
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

    private SNode samplePath(
            HashMap<SNode, Double> logPriors,
            HashMap<SNode, Double> wordLlhs,
            HashMap<SNode, Double> resLlhs,
            boolean observed) {
        ArrayList<SNode> pathList = new ArrayList<SNode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SNode node : logPriors.keySet()) {
            double lp = logPriors.get(node) + wordLlhs.get(node);
            if (supervised && observed) {
                lp += resLlhs.get(node);
            }

            pathList.add(node);
            logProbs.add(lp);
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode path = pathList.get(sampledIndex);
        return path;
    }

    private double computeMarginals(
            HashMap<SNode, Double> pathLogPriors,
            HashMap<SNode, Double> pathWordLogLikelihoods,
            HashMap<SNode, Double> pathResLogLikelihoods,
            boolean resObserved) {
        double marginal = 0.0;
        for (SNode node : pathLogPriors.keySet()) {
            double logprior = pathLogPriors.get(node);
            double loglikelihood = pathWordLogLikelihoods.get(node);

            double lp = logprior + loglikelihood;
            if (isSupervised() && resObserved) {
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

    private void computePathLogPrior(
            HashMap<SNode, Double> nodeLogProbs,
            SNode curNode,
            double parentLogProb,
            int[] firstLevelNodeIndices) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            if (curNode.isRoot()) {
                int totalNumValidCusts = 0;
                ArrayList<SNode> validChildren = new ArrayList<SNode>();
                for (SNode child : curNode.getChildren()) {
                    int childIndex = child.getIndex();
                    if (!isValid(firstLevelNodeIndices, childIndex)) {
                        continue;
                    }
                    totalNumValidCusts += child.getNumCustomers();
                    validChildren.add(child);
                }

                double logNorm = Math.log(totalNumValidCusts
                        + validChildren.size() * gammas[curNode.getLevel()]);
                for (SNode validChild : validChildren) {
                    double childWeight = parentLogProb
                            + Math.log(validChild.getNumCustomers() + gammas[curNode.getLevel()])
                            - logNorm;
                    computePathLogPrior(nodeLogProbs, validChild, childWeight, firstLevelNodeIndices);
                }
            } else {
                double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);
                newWeight += logGammas[curNode.getLevel()] - logNorm;

                for (SNode child : curNode.getChildren()) {
                    double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                    computePathLogPrior(nodeLogProbs, child, childWeight, firstLevelNodeIndices);
                }
            }
        }

        if (!curNode.isRoot()) // do not extend the root node
        {
            nodeLogProbs.put(curNode, newWeight);
        }
    }

    private boolean isValid(int[] arr, int val) {
        if (arr.length == 0) {
            return true;
        }
        for (int ii = 0; ii < arr.length; ii++) {
            if (arr[ii] == val) {
                return true;
            }
        }
        return false;
    }

    /**
     * Compute the probability of a given set of observed words (with their
     * current assignments) having been generated from each valid path in the
     * tree.
     *
     * @param nodeDataLlhs Result hash map
     * @param curNode The current node in the recursive call
     * @param docTokenCountPerLevel The token count per level (given the current
     * level assignments)
     * @param dataLlhNewTopic The log probability of new node at each level
     * @param parentDataLlh Value passed from the parent
     * @param firstLevelNodeIndices Set of valid first-level nodes
     */
    private void computePathWordLogLikelihood(
            HashMap<SNode, Double> nodeDataLlhs,
            SNode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh,
            int[] firstLevelNodeIndices) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getContent().getLogLikelihood(docTokenCountPerLevel[level]);

        // populate to child nodes
        for (SNode child : curNode.getChildren()) {
            if (curNode.isRoot() && !isValid(firstLevelNodeIndices, child.getIndex())) {
                continue;
            }

            computePathWordLogLikelihood(nodeDataLlhs, child, docTokenCountPerLevel,
                    dataLlhNewTopic, parentDataLlh + nodeDataLlh, firstLevelNodeIndices);
        }

        // store the data llh from the root to this current node
        if (!curNode.isRoot()) {
            double storeDataLlh = parentDataLlh + nodeDataLlh;
            level++;
            while (level < L) // if this is an internal node, add llh of new child node
            {
                storeDataLlh += dataLlhNewTopic[level++];
            }
            nodeDataLlhs.put(curNode, storeDataLlh);
        }
    }

    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            int s,
            int[] firstLevelNodeIndices) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            if (!node.isRoot()) {
                SNode[] path = getPathFromNode(node);
                double addTopicWeight = 0.0;
                double var = hyperparams.get(RHO);
                int level;
                for (level = 0; level < path.length; level++) {
                    addTopicWeight += path[level].getRegressionParameter()
                            * sentLevelCounts[d][s].getCount(level);
                }

                while (level < L) {
                    int levelCount = sentLevelCounts[d][s].getCount(level);
                    addTopicWeight += levelCount * mus[level];
                    var += Math.pow((double) levelCount / docTokenCounts[d], 2) * sigmas[level];
                    level++;
                }

                // note: the topic weight of the current sentence s has been excluded
                // from docTopicWeights[d]
                double mean = (docTopicWeights[d] + docLexicalWeights[d] + addTopicWeight) / docTokenCounts[d];
                double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
                resLlhs.put(node, resLlh);
            }

            for (SNode child : node.getChildren()) {
                if (node.isRoot() && !isValid(firstLevelNodeIndices, child.getIndex())) {
                    continue;
                }
                stack.add(child);
            }
        }

        return resLlhs;
    }

    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            STable table,
            int[] firstLevelNodeIndices) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            if (!node.isRoot()) {
                SNode[] path = getPathFromNode(node);
                double addSum = 0.0;
                double var = hyperparams.get(RHO);
                int level;
                for (level = 0; level < path.length; level++) {
                    for (int s : table.getCustomers()) {
                        addSum += path[level].getRegressionParameter() * sentLevelCounts[d][s].getCount(level);
                    }
                }
                while (level < L) {
                    int totalLevelCount = 0;
                    for (int s : table.getCustomers()) {
                        int levelCount = sentLevelCounts[d][s].getCount(level);
                        addSum += levelCount * mus[level];
                        totalLevelCount += levelCount;
                    }
                    var += Math.pow((double) totalLevelCount / docTokenCounts[d], 2) * sigmas[level];
                    level++;
                }

                double mean = (docTopicWeights[d] + docLexicalWeights[d] + addSum) / docTokenCounts[d];
                double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
                resLlhs.put(node, resLlh);
            }

            for (SNode child : node.getChildren()) {
                if (node.isRoot() && !isValid(firstLevelNodeIndices, child.getIndex())) {
                    continue;
                }

                stack.add(child);
            }
        }
        return resLlhs;
    }

    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    private boolean isLeafNode(SNode node) {
        return node.getLevel() == L - 1;
    }

    private SNode getNode(int[] parsedPath) {
        SNode node = globalTreeRoot;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    /**
     * Compute the regression sum from the topic tree for a sentence
     *
     * @param d The document index
     * @param s The sentence index
     * @return The regression sum of the sentence
     */
    private double computeTopicWeight(int d, int s) {
        double regSum = 0.0;
        if (c[d][s] == null) {
            System.out.println("---> c. d = " + d + ". s = " + s);
        } else if (c[d][s].getContent() == null) {
            System.out.println("---> content");
        }

        SNode[] path = getPathFromNode(c[d][s].getContent());
        for (int l = 0; l < path.length; l++) {
            regSum += path[l].getRegressionParameter() * sentLevelCounts[d][s].getCount(l);
        }
        return regSum;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    private SNode[] getPathFromNode(SNode node) {
        SNode[] path = new SNode[node.getLevel() + 1];
        SNode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
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
                .append(". # total customers: ").append(localRestaurants[d].getTotalNumCustomers())
                .append("\n");
        for (STable table : localRestaurants[d].getTables()) {
            str.append("--- table: ").append(table.toString()).append("\n");
        }
        return str.toString();
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg);

        validateModel(msg);

        validateAssignments(msg);
    }

    private void validateModel(String msg) {
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            if (!isLeafNode(node)) {
                int childNumCusts = 0;

                for (SNode child : node.getChildren()) {
                    childNumCusts += child.getNumCustomers();
                    stack.add(child);
                }

                if (childNumCusts != node.getNumCustomers()) {
                    throw new RuntimeException(msg + ". Numbers of customers mismatch. "
                            + node.toString());
                }
            }

            if (this.isLeafNode(node) && node.isEmpty()) {
                throw new RuntimeException(msg + ". Leaf node " + node.toString()
                        + " is empty");
            }
        }
    }

    private void validateAssignments(String msg) {
        for (int d = 0; d < D; d++) {
            docLevelDists[d].validate(msg);
        }

        for (int d = 0; d < D; d++) {
            int totalCusts = 0;
            for (STable table : localRestaurants[d].getTables()) {
                totalCusts += table.getNumCustomers();
            }
            if (totalCusts != words[d].length) {
                for (STable table : localRestaurants[d].getTables()) {
                    System.out.println(table.toString() + ". customers: " + table.getCustomers().toString());
                }
                throw new RuntimeException(msg + ". Numbers of customers in restaurant " + d
                        + " mismatch. " + totalCusts + " vs. " + words[d].length);
            }

            HashMap<STable, Integer> tableCustCounts = new HashMap<STable, Integer>();
            for (int s = 0; s < words[d].length; s++) {
                Integer count = tableCustCounts.get(c[d][s]);

                if (count == null) {
                    tableCustCounts.put(c[d][s], 1);
                } else {
                    tableCustCounts.put(c[d][s], count + 1);
                }
            }

            if (tableCustCounts.size() != localRestaurants[d].getNumTables()) {
                throw new RuntimeException(msg + ". Numbers of tables mismatch in"
                        + " restaurant " + d);
            }

            for (STable table : localRestaurants[d].getTables()) {
                if (table.getNumCustomers() != tableCustCounts.get(table)) {
                    System.out.println("Table: " + table.toString());

                    for (int s : table.getCustomers()) {
                        System.out.println("--- s = " + s + ". " + c[d][s].toString());
                    }
                    System.out.println(tableCustCounts.get(table));


                    throw new RuntimeException(msg + ". Number of customers "
                            + "mismatch. Table " + table.toString()
                            + ". " + table.getNumCustomers() + " vs. " + tableCustCounts.get(table));
                }
            }
        }

        for (int d = 0; d < D; d++) {
            double topicWeight = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                topicWeight += computeTopicWeight(d, s);
            }
            if (Math.abs(topicWeight - docTopicWeights[d]) > 0.01) {
                throw new RuntimeException(msg + ". Topic weights of document " + d
                        + " mismatch. " + topicWeight + " vs. " + docTokenCounts[d]);
            }
        }
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood();

            if (isSupervised()) {
                regParamLgprob += StatisticsUtils.logNormalProbability(node.getRegressionParameter(),
                        mus[node.getLevel()], Math.sqrt(sigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(gammas[node.getLevel()]);
            }

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double restLgprob = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += docLevelDists[d].getLogLikelihood();

            restLgprob += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));

            if (supervised) {
                resLlh += StatisticsUtils.logNormalProbability(responses[d],
                        regValues[d], sqrtRho);
            }
        }

        logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                + ". tree = " + MiscUtils.formatDouble(treeLogProb)
                + ". rest = " + MiscUtils.formatDouble(restLgprob)
                + ". stick = " + MiscUtils.formatDouble(stickLgprob)
                + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                + ". response = " + MiscUtils.formatDouble(resLlh));

        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh + restLgprob;
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

    public void outputTopicTopWords(File outputFile, int numWords)
            throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (this.labelVocab == null) {
            throw new RuntimeException("The label vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }
            if (node.getIterationCreated() >= MAX_ITER - LAG) {
                continue;
            }

            String[] topWords = getTopWords(node.getContent().getDistribution(), numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            if(node.getLevel() == 1) {
                str.append(labelVocab.get(node.getIndex()))
                        .append("\n");
                for (int i = 0; i < node.getLevel(); i++) {
                    str.append("   ");
                }
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

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write(str.toString());
        writer.close();
    }

    public void outputTopicCoherence(
            File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            logln("Outputing topic coherence to file " + file);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(node.getPathString()
                    + "\t" + node.getIterationCreated()
                    + "\t" + node.getNumCustomers()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }

        writer.close();
    }

    public void outputLexicalWeights(File file) throws Exception {
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < V; v++) {
            if (lexicalWeights[v] != 0) {
                rankItems.add(new RankingItem<Integer>(v, lexicalWeights[v]));
            }
        }
        Collections.sort(rankItems);

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int ii = 0; ii < rankItems.size(); ii++) {
            RankingItem<Integer> item = rankItems.get(ii);
            writer.write(item.getObject()
                    + "\t" + item.getPrimaryValue()
                    + "\t" + wordVocab.get(item.getObject())
                    + "\n");
        }
        writer.close();
    }

    public void outputTopicWordDistributions(File file) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            writer.write(node.getPathString());
            for (int v = 0; v < distribution.length; v++) {
                writer.write("\t" + distribution[v]);
            }
            writer.write("\n");
        }

        writer.close();
    }

    public void outputDocPathAssignments(File file) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int d = 0; d < D; d++) {
            writer.append(Integer.toString(d));
            for (int s = 0; s < this.c[d].length; s++) {
                writer.append("\t" + c[d][s].getContent().getPathString());
            }
            writer.write("\n");
        }
        writer.close();
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath + "\n");
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int v = 0; v < V - 1; v++) {
                modelStr.append(lexicalWeights[v]).append("\t");
            }
            modelStr.append(lexicalWeights[V - 1]).append("\n");

            Stack<SNode> stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getIterationCreated()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirichletMultinomialModel.output(node.getContent())).append("\n");

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
            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            data = assignStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath + "\n");
        }

        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Load the model from a compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath + "\n");
        }

        // initialize
        this.initializeModelStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
        BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
        String line = reader.readLine();
        String[] sline = line.split("\t");
        this.lexicalWeights = new double[sline.length];
        for (int v = 0; v < V; v++) {
            this.lexicalWeights[v] = Double.parseDouble(sline[v]);
        }

        HashMap<String, SNode> nodeMap = new HashMap<String, SNode>();
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int iterCreated = Integer.parseInt(reader.readLine());
            int numCustomers = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirichletMultinomialModel dmm = DirichletMultinomialModel.input(reader.readLine());

            // create node
            int lastColonIndex = pathStr.lastIndexOf(":");
            SNode parent = null;
            if (lastColonIndex != -1) {
                parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
            }

            String[] pathIndices = pathStr.split(":");
            int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
            int nodeLevel = pathIndices.length - 1;
            SNode node = new SNode(iterCreated, nodeIndex,
                    nodeLevel, dmm, regParam, parent);

            node.changeNumCustomers(numCustomers);

            if (node.getLevel() == 0) {
                globalTreeRoot = node;
            }

            if (parent != null) {
                parent.addChild(node.getIndex(), node);
            }

            nodeMap.put(pathStr, node);
        }
        reader.close();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (!isLeafNode(node)) {
                node.fillInactiveChildIndices();
                for (SNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        validateModel("Loading model " + filename);
    }

    /**
     * Load the assignments of the training data from the compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath + "\n");
        }

        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
        BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
        String[] sline;

        for (int d = 0; d < D; d++) {
            sline = reader.readLine().split("\t");
            if (d != Integer.parseInt(sline[0])) {
                throw new RuntimeException("Mismatch");
            }
            int numTables = Integer.parseInt(sline[1]);

            for (int i = 0; i < numTables; i++) {
                int tabIndex = Integer.parseInt(reader.readLine());
                int iterCreated = Integer.parseInt(reader.readLine());
                String leafPathStr = reader.readLine();

                SNode leafNode = getNode(parseNodePath(leafPathStr));
                STable table = new STable(iterCreated, tabIndex, leafNode, d);
                localRestaurants[d].addTable(table);
            }
        }

        for (int d = 0; d < D; d++) {
            localRestaurants[d].fillInactiveTableIndices();
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                sline = reader.readLine().split("\t");
                if (!sline[0].equals(d + ":" + s)) {
                    throw new RuntimeException("Mismatch");
                }
                int tableIndex = Integer.parseInt(sline[1]);
                c[d][s] = localRestaurants[d].getTable(tableIndex);
            }
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + s + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    z[d][s][n] = Integer.parseInt(sline[1]);
                }
            }
        }

        reader.close();
    }

    class SNode extends TreeNode<SNode, DirichletMultinomialModel> {

        private final int born;
        private int numCustomers;
        private double regression;

        SNode(int iter, int index, int level,
                DirichletMultinomialModel content,
                double regParam,
                SNode parent) {
            super(index, level, content, parent);
            this.born = iter;
            this.numCustomers = 0;
            this.regression = regParam;
        }

        public int getIterationCreated() {
            return this.born;
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
            int maxChildIndex = PSEUDO_NODE_INDEX;
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

    class STable extends Table<Integer, SNode> {

        private final int born;
        private final int restIndex;

        public STable(int iter, int index, SNode content, int restId) {
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
                    .append(" >> ").append(getContent() == null ? "null"
                    : getContent().toString());
            return str.toString();
        }
    }
}
