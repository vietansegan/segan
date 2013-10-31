package sampler.supervised.regression;

import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import core.crossvalidation.RegressionDocumentInstance;
import data.RegressionTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import optimization.GurobiMultipleLinearRegression;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.FullTable;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class SHLDA extends AbstractSampler {

    public static final String IterPredictionFolder = "iter-predictions/";
    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    public static final int ALPHA = 0;
    public static final int RHO = 1;
    public static final int GEM_MEAN = 2;
    public static final int GEM_SCALE = 3;
    public static final int TAU_MEAN = 4;
    public static final int TAU_SCALE = 5;
    protected boolean supervised = true;
    protected boolean optimizeLexicalWeights = false;
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;    // regression parameter means
    protected double[] sigmas; // regression parameter variances
    protected int[][][] words;  // [D] x [S_d] x [N_ds]: words
    protected double[] responses; // [D]
    // input statistics
    private int sentCount;
    private int tokenCount;
    private int[] docTokenCounts;
    private double logAlpha;
    private double sqrtRho;
    private double[] sqrtSigmas;
    private double[] logGammas;
    private STable[][] c; // path assigned to sentences
    private int[][][] z; // level assigned to tokens
    // state structure
    private SNode globalTreeRoot; // tree
    private Restaurant<STable, Integer, SNode>[] localRestaurants; // franchise
    private TruncatedStickBreaking[] docLevelDists; // doc sticks
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
    private DirMult[] emptyModels;
    private int numTokenAssignmentsChange;
    private int numSentAssignmentsChange;
    private int numTableAssignmentsChange;

    public void configure(String folder,
            int[][][] words,
            double[] responses,
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
            double[] lexicalWeights, // weights for all lexical items (i.e., words)
            int numLexicalItems, // number of lexical items considered
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.words = words;
        this.responses = responses;

        this.V = V;
        this.L = L;
        this.D = this.words.length;

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

        this.numLexicalItems = numLexicalItems;
        if (lexicalWeights != null) { // if the lexical weights are given
            this.lexicalWeights = lexicalWeights;
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

        if (verbose) {
            logln("--- V = " + V);
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
            logln("--- initialize:\t" + initState);

            logln("--- responses:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatisticsUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatisticsUtils.standardDeviation(responses)));
            int[] histogram = StatisticsUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }
        }
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
                .append("_SHLDA")
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
        int rootLevel = 0;
        int rootIndex = 0;
        DirMult dmModel = new DirMult(V, betas[rootLevel], uniform);
        double regParam = 0.0;
        this.globalTreeRoot = new SNode(iter, rootIndex, rootLevel, dmModel, regParam, null);

        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }
    }

    private void initializeDataStructure() {
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
            logln("--- Initializing random assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // create a new table for each sentence
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // create a new path for each table
                SNode node = globalTreeRoot;
                for (int l = 0; l < L - 1; l++) {
                    node = createNode(node);
                }
                addTableToPath(node);
                table.setContent(node);

                // sample level
                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelForToken(d, s, n, !REMOVE, ADD, !OBSERVED);
                }

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

            BufferedWriter writer = IOUtils.getBufferedWriter(new File(getSamplerFolderPath(), "weights.txt"));
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
        DirMult dmm = new DirMult(V, betas[level], uniform);
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
    private void sampleTableForSentence(int d, int s, boolean remove, boolean add,
            boolean observed, boolean extend) {
        STable curTable = c[d][s];

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
        boolean condition = false;
        if (condition) {
            double topicWeight = 0.0;
            for (int ss = 0; ss < words[d].length; ss++) {
                topicWeight += computeTopicWeight(d, ss);
            }

            logln("1. iter = " + iter
                    + " d = " + d
                    + " s = " + s
                    + ". docTopicWeight = " + docTopicWeights[d]
                    + ". true = " + topicWeight);
        }

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
//            logln("iter = " + iter + ". d = " + d + ". s = " + s
//                    + ". table: " + table.toString()
//                    + ". log prior = " + MiscUtils.formatDouble(logprior)
//                    + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                    + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                    + ". lp = " + MiscUtils.formatDouble(lp));
        }

        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (extend) {
            // log priors
            computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

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
                    0.0);

            // debug
            if (pathLogPriors.size() != pathWordLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }

            // response log likelihoods
            if (supervised && observed) {
                pathResLlhs = computePathResponseLogLikelihood(d, s);

                if (pathLogPriors.size() != pathResLlhs.size()) {
                    throw new RuntimeException("Numbers of paths mismatch");
                }
            }

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
    private void samplePathForTable(int d, STable table,
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
        computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

        // word log likelihoods
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        computePathWordLogLikelihood(pathWordLlhs, globalTreeRoot, obsCountPerLevel, dataLlhNewTopic, 0.0);

        // debug
        if (pathLogPriors.size() != pathWordLlhs.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        // response log likelihoods
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (supervised && observed) {
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

    private void optimizeAllRegressionParameters() {
        ArrayList<SNode> flattenTree = flattenTreeWithoutRoot();
        int numTopicParams = flattenTree.size();
        int numLexParams = lexicalIndices.size();

        double[] lambdas = new double[numTopicParams + numLexParams];
        HashMap<SNode, Integer> nodeIndices = new HashMap<SNode, Integer>();
        for (int i = 0; i < flattenTree.size(); i++) {
            SNode node = flattenTree.get(i);
            nodeIndices.put(node, i);
            lambdas[i] = 1.0 / sigmas[node.getLevel()];
        }
        for (int ii = 0; ii < numLexParams; ii++) {
            lambdas[numTopicParams + ii] = 200;
        }
//            lambdas[numTopicParams + ii] = 1.0 / hyperparams.get(TAU_SCALE);

        // design matrix
        double[][] designMatrix = new double[D][numTopicParams + numLexParams];
        for (int d = 0; d < D; d++) {
            // topic
            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 1; l < L; l++) {
                    int nodeIdx = nodeIndices.get(path[l]);
                    int count = sentLevelCounts[d][s].getCount(l);
                    designMatrix[d][nodeIdx] += count;
                }
            }
            for (int i = 0; i < numTopicParams; i++) {
                designMatrix[d][i] /= docTokenCounts[d];
            }
            System.arraycopy(docLexicalDesignMatrix[d], 0, designMatrix[d], numTopicParams, numLexParams);
        }

        GurobiMultipleLinearRegression mlr =
                new GurobiMultipleLinearRegression(designMatrix, responses, lambdas);
        double[] weights = mlr.solve();

        // update
        for (int i = 0; i < numTopicParams; i++) {
            flattenTree.get(i).setRegressionParameter(weights[i]);
        }
        for (int i = 0; i < numLexParams; i++) {
            int vocIdx = lexicalIndices.get(i);
            lexicalWeights[vocIdx] = weights[numTopicParams + i];
        }

        updateDocumentTopicWeights();
        updateDocumentLexicalWeights();
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

    private ArrayList<SNode> flattenTree() {
        ArrayList<SNode> flattenTree = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            flattenTree.add(node);
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

    private void computePathWordLogLikelihood(
            HashMap<SNode, Double> nodeDataLlhs,
            SNode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getContent().getLogLikelihood(docTokenCountPerLevel[level]);

        // populate to child nodes
        for (SNode child : curNode.getChildren()) {
            computePathWordLogLikelihood(nodeDataLlhs, child, docTokenCountPerLevel,
                    dataLlhNewTopic, parentDataLlh + nodeDataLlh);
        }

        // store the data llh from the root to this current node
        double storeDataLlh = parentDataLlh + nodeDataLlh;
        level++;
        while (level < L) // if this is an internal node, add llh of new child node
        {
            storeDataLlh += dataLlhNewTopic[level++];
        }
        nodeDataLlhs.put(curNode, storeDataLlh);
    }

    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            int s) {
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
                addTopicWeight += path[level].getRegressionParameter() * sentLevelCounts[d][s].getCount(level);
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

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        return resLlhs;
    }

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

            for (SNode child : node.getChildren()) {
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
                .append(". # total customers: ").append(localRestaurants[d].getTotalNumCustomers()).append("\n");
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
            writer.write(item.getObject() + "\t" + item.getPrimaryValue() + "\n");
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
                modelStr.append(DirMult.output(node.getContent())).append("\n");

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
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
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

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + ModelFileExt);
        HashMap<String, SNode> nodeMap = new HashMap<String, SNode>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
        String line = reader.readLine();
        String[] sline = line.split("\t");
        this.lexicalWeights = new double[sline.length];
        for (int v = 0; v < V; v++) {
            this.lexicalWeights[v] = Double.parseDouble(sline[v]);
        }

        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int iterCreated = Integer.parseInt(reader.readLine());
            int numCustomers = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

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

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + AssignmentFileExt);
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
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

    public double[] outputRegressionResults(
            double[] trueResponses,
            String predFilepath,
            String outputFile) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(predFilepath);
        String line = reader.readLine();
        String[] modelNames = line.split("\t");
        int numModels = modelNames.length;

        double[][] predResponses = new double[numModels][trueResponses.length];

        int idx = 0;
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            for (int j = 0; j < numModels; j++) {
                predResponses[j][idx] = Double.parseDouble(sline[j]);
            }
            idx++;
        }
        reader.close();

        double[] finalPredResponses = new double[trueResponses.length];
        for (int d = 0; d < trueResponses.length; d++) {
            double sum = 0.0;
            for (int i = 0; i < numModels; i++) {
                sum += predResponses[i][d];
            }
            finalPredResponses[d] = sum / numModels;
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int i = 0; i < numModels; i++) {
            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, predResponses[i]);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();

            if (i == 0) {
                writer.write("Model");
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getName());
                }
                writer.write("\n");
            }
            writer.write(modelNames[i]);
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
        }
        writer.close();

        return finalPredResponses;
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }

    public void testSampler(int[][][] newWords) {
        if (verbose) {
            logln("Test sampling ...");
        }
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        File iterPredFolder = new File(getSamplerFolderPath(), IterPredictionFolder);
        IOUtils.createFolder(iterPredFolder);

        try {
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File partialResultFile = new File(iterPredFolder, IOUtils.removeExtension(filename) + ".txt");
                sampleNewDocuments(
                        new File(reportFolder, filename),
                        newWords,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    private void sampleNewDocuments(
            File stateFile,
            int[][][] newWords,
            String outputResultFile) throws Exception {
        if (verbose) {
            logln("\nPerform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        words = newWords;
        responses = null; // for evaluation
        D = words.length;

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

        logln("--- V = " + V);
        logln("--- # documents = " + D); // number of groups
        logln("--- # sentences = " + sentCount);
        logln("--- # tokens = " + tokenCount);

        // initialize structure for test data
        initializeDataStructure();

        if (verbose) {
            logln("Initialized data structure");
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        // initialize random assignments
        initializeRandomAssignmentsNewDocuments();

        if (verbose) {
            logln("Initialized random assignments");
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    if (words[d].length > 1) // if this document has only 1 sentence, no sampling is needed
                    {
                        sampleTableForSentence(d, s, REMOVE, ADD, !OBSERVED, !EXTEND);
                    }

                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleLevelForToken(d, s, n, REMOVE, ADD, !OBSERVED);
                    }
                }

                for (STable table : localRestaurants[d].getTables()) {
                    samplePathForTable(d, table, REMOVE, ADD, !OBSERVED, !EXTEND);
                }
            }

            if (verbose && iter % LAG == 0) {
                logln("--- iter = " + iter + " / " + MAX_ITER);
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                this.updateDocumentLexicalWeights();
                this.updateDocumentTopicWeights();

                double[] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);
            }
        }

        // output result during test time 
        BufferedWriter writer = IOUtils.getBufferedWriter(outputResultFile);
        for (int d = 0; d < D; d++) {
            writer.write(Integer.toString(d));

            for (int ii = 0; ii < predResponsesList.size(); ii++) {
                writer.write("\t" + predResponsesList.get(ii)[d]);
            }
            writer.write("\n");
        }
        writer.close();
    }

    private void initializeRandomAssignmentsNewDocuments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // create a new table for each sentence
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // sample level
                for (int n = 0; n < words[d][s].length; n++) {
                    // sample from prior
//                    double[] levelDist = docLevelDists[d].getDistribution();
//                    int randLevel = SamplerUtils.scaleSample(levelDist);
                    int randLevel = rand.nextInt(L);

                    // update and increment
                    z[d][s][n] = randLevel;
                    docLevelDists[d].increment(z[d][s][n]);
                    sentLevelCounts[d][s].increment(z[d][s][n]);
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                samplePathForTable(d, table, !REMOVE, ADD, !OBSERVED, !EXTEND);
            }
        }
    }

    class SNode extends TreeNode<SNode, DirMult> {

        private final int born;
        private int numCustomers;
        private double regression;

        SNode(int iter, int index, int level, DirMult content, double regParam,
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

    class STable extends FullTable<Integer, SNode> {

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
                    .append(" >> ").append(getContent() == null ? "null" : getContent().toString());
            return str.toString();
        }
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

            addOption("output", "Output folder");
            addOption("dataset", "Dataset");
            addOption("data-folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("burnIn", "Burn-in");
            addOption("maxIter", "Maximum number of iterations");
            addOption("sampleLag", "Sample lag");
            addOption("report", "Report interval.");
            addOption("gem-mean", "GEM mean. [0.5]");
            addOption("gem-scale", "GEM scale. [50]");
            addOption("betas", "Dirichlet hyperparameter for topic distributions."
                    + " [1, 0.5, 0.25] for a 3-level tree.");
            addOption("gammas", "DP hyperparameters. [1.0, 1.0] for a 3-level tree");
            addOption("mus", "Prior means for topic regression parameters."
                    + " [0.0, 0.0, 0.0] for a 3-level tree and standardized"
                    + " response variable.");
            addOption("sigmas", "Prior variances for topic regression parameters."
                    + " [0.0001, 0.5, 1.0] for a 3-level tree and stadardized"
                    + " response variable.");
            addOption("rho", "Prior variance for response variable. [1.0]");
            addOption("tau-mean", "Prior mean of lexical regression parameters. [0.0]");
            addOption("tau-scale", "Prior scale of lexical regression parameters. [1.0]");
            addOption("num-lex-items", "Number of non-zero lexical regression parameters."
                    + " Defaule: vocabulary size.");

            addOption("cv-folder", "Cross validation folder");
            addOption("num-folds", "Number of folds");
            addOption("run-mode", "Running mode");
            addOption("fold", "The cross-validation fold to run");

            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("z", false, "whether standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' "
                        + "main.RunLexicalSHLDA -help", options);
                return;
            }

            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
//                runModel();
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar sampler.supervised.regression.SHLDA -help", options);
            System.exit(1);
        }
    }

    public static void runCrossValidation() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String resultFolder = cmd.getOptionValue("output");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");
        InitialState initState = InitialState.RANDOM;

        String cvFolder = cmd.getOptionValue("cv-folder");
        int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));
        String runMode = cmd.getOptionValue("run-mode");

        if (resultFolder == null) {
            throw new RuntimeException("Result folder (--output) is not set.");
        }

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        RegressionTextDataset data = new RegressionTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        data.prepareTopicCoherence(numTopWords);

        int V = data.getWordVocab().size();
        int L = CLIUtils.getIntegerArgument(cmd, "tree-height", 3);
        double gem_mean = CLIUtils.getDoubleArgument(cmd, "gem-mean", 0.3);
        double gem_scale = CLIUtils.getDoubleArgument(cmd, "gem-scale", 50);

        double[] defaultBetas = new double[L];
        defaultBetas[0] = 1;
        for (int i = 1; i < L; i++) {
            defaultBetas[i] = 1.0 / (i + 1);
        }
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", defaultBetas, ",");
        for (int i = 0; i < betas.length; i++) {
            betas[i] = betas[i] * V;
        }

        double[] defaultGammas = new double[L - 1];
        for (int i = 0; i < defaultGammas.length; i++) {
            defaultGammas[i] = 1.0;
        }

        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", defaultGammas, ",");

        double[] responses = data.getResponses();
        if (cmd.hasOption("z")) {
            ZNormalizer zNorm = new ZNormalizer(responses);
            for (int i = 0; i < responses.length; i++) {
                responses[i] = zNorm.normalize(responses[i]);
            }
        }

        double meanResponse = StatisticsUtils.mean(responses);
        double[] defaultMus = new double[L];
        for (int i = 0; i < L; i++) {
            defaultMus[i] = meanResponse;
        }
        double[] mus = CLIUtils.getDoubleArrayArgument(cmd, "mus", defaultMus, ",");

        double[] defaultSigmas = new double[L];
        defaultSigmas[0] = 0.0001; // root node
        for (int l = 1; l < L; l++) {
            defaultSigmas[l] = 0.5 * l;
        }
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", defaultSigmas, ",");

        double tau_mean = CLIUtils.getDoubleArgument(cmd, "tau-mean", 0.0);
        double tau_scale = CLIUtils.getDoubleArgument(cmd, "tau-scale", 1.0);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 1.0);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        int numLexicalItems = CLIUtils.getIntegerArgument(cmd, "num-lex-items", V);

        if (verbose) {
            System.out.println("\nLoading cross validation info from " + cvFolder);
        }
        ArrayList<RegressionDocumentInstance> instanceList = new ArrayList<RegressionDocumentInstance>();
        for (int i = 0; i < data.getDocIds().length; i++) {
            instanceList.add(new RegressionDocumentInstance(
                    data.getDocIds()[i],
                    data.getWords()[i],
                    data.getResponses()[i]));
        }
        String cvName = "";
        CrossValidation<String, RegressionDocumentInstance> crossValidation =
                new CrossValidation<String, RegressionDocumentInstance>(
                cvFolder,
                cvName,
                instanceList);
        crossValidation.inputFolds(numFolds);
        int foldIndex = -1;
        if (cmd.hasOption("fold")) {
            foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
        }

        for (Fold<String, ? extends Instance<String>> fold : crossValidation.getFolds()) {
            if (foldIndex != -1 && fold.getIndex() != foldIndex) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + foldIndex);
            }

            File foldFolder = new File(resultFolder, fold.getFoldName());

            SHLDA sampler = new SHLDA();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setLog(true);
            sampler.setReport(true);
            sampler.setWordVocab(data.getWordVocab());

            // training data
            ArrayList<Integer> trInstIndices = fold.getTrainingInstances();
            int[][][] trRevWords = data.getDocSentWords(trInstIndices);
            double[] trResponses = data.getResponses(trInstIndices);

            // test data
            ArrayList<Integer> teInstIndices = fold.getTestingInstances();
            int[][][] teRevWords = data.getDocSentWords(teInstIndices);
            double[] teResponses = data.getResponses(teInstIndices);

            sampler.configure(foldFolder.getAbsolutePath(),
                    trRevWords, trResponses,
                    V, L,
                    alpha,
                    rho,
                    gem_mean, gem_scale,
                    tau_mean, tau_scale,
                    betas, gammas,
                    mus, sigmas,
                    null, numLexicalItems,
                    initState, paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);

            File samplerFolder = new File(foldFolder, sampler.getSamplerFolder());
            File iterPredFolder = sampler.getIterationPredictionFolder();
            IOUtils.createFolder(samplerFolder);

            if (runMode.equals("train")) {
                sampler.initialize();
                sampler.iterate();
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile), data.getTopicCoherence());

                sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                sampler.outputLexicalWeights(new File(samplerFolder, "lexical-reg-params.txt"));
                sampler.outputDocPathAssignments(new File(samplerFolder, "doc-topic.txt"));
            } else if (runMode.equals("test")) {
                sampler.testSampler(teRevWords);
                File teResultFolder = new File(samplerFolder, "te-results");
                IOUtils.createFolder(teResultFolder);
                GibbsRegressorUtils.evaluate(iterPredFolder, teResultFolder, teResponses);
            } else if (runMode.equals("train-test")) {
                // train
                sampler.initialize();
                sampler.iterate();
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile), data.getTopicCoherence());
                sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                sampler.outputLexicalWeights(new File(samplerFolder, "lexical-reg-params.txt"));
                sampler.outputDocPathAssignments(new File(samplerFolder, "doc-topic.txt"));

                // test
                sampler.testSampler(teRevWords);
                File teResultFolder = new File(samplerFolder, "te-results");
                IOUtils.createFolder(teResultFolder);
                GibbsRegressorUtils.evaluate(iterPredFolder, teResultFolder, teResponses);
            } else {
                throw new RuntimeException("Run mode " + runMode + " not supported");
            }
        }
    }
}
