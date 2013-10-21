package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import core.crossvalidation.RegressionDocumentInstance;
import data.SingleResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirichletMultinomialModel;
import sampling.util.Restaurant;
import sampling.util.Table;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
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
public class SHDP extends AbstractSampler {

    public static final String IterPredictionFolder = "iter-predictions/";
    public static final int PSEUDO_INDEX = -1;
    public static final int ALPHA_GLOBAL = 0;
    public static final int ALPHA_LOCAL = 1;
    public static final int BETA = 2;
    public static final int MU = 3;
    public static final int SIGMA = 4;
    public static final int RHO = 5;
    protected boolean supervised = true;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int K; // initial number of tables
    protected int[][] words;
    protected double[] responses;
    private SHDPTable[][] z; // local table index
    private Restaurant<SHDPDish, SHDPTable, DirichletMultinomialModel> globalRestaurant;
    private Restaurant<SHDPTable, Integer, SHDPDish>[] localRestaurants;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private int numTokens = 0;
    private double[] uniform;
    private DirichletMultinomialModel emptyModel;
    private int numTokenAsgnsChange;
    private int numTableAsgnsChange;
    private int numConverged;

    public void configure(String folder,
            int[][] words, double[] responses,
            int V,
            double alpha_global, double alpha_local, double beta,
            double mu, double sigma, double rho,
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
        this.D = this.words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha_global);
        this.hyperparams.add(alpha_local);
        this.hyperparams.add(beta);
        this.hyperparams.add(mu);
        this.hyperparams.add(sigma);
        this.hyperparams.add(rho);

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

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        for (int d = 0; d < D; d++) {
            numTokens += this.words[d].length;
        }
        logln("--- D = " + D);
        logln("--- V = " + V);
        logln("--- # observations = " + numTokens);

        if (!debug) {
            System.err.close();
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_SHDP")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_ag-").append(formatter.format(hyperparams.get(ALPHA_GLOBAL)))
                .append("_al-").append(formatter.format(hyperparams.get(ALPHA_LOCAL)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void setK(int K) {
        this.K = K;
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

        if (verbose) {
            logln("--- --- Done initializing. \n" + getCurrentState());
        }
    }

    protected void initializeModelStructure() {
        this.globalRestaurant = new Restaurant<SHDPDish, SHDPTable, DirichletMultinomialModel>();

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<SHDPTable, Integer, SHDPDish>();
        }

        this.emptyModel = new DirichletMultinomialModel(V, hyperparams.get(BETA), uniform);
    }

    protected void initializeDataStructure() {
        z = new SHDPTable[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new SHDPTable[words[d].length];
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                this.initializePresetAssignments();
                break;

            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        // run LDA
        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        if (K == 0) {// this is not set
            K = 50;
        }
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        lda.configure(null, words, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            String ldaFile = this.folder + "lda-init-" + K + ".txt";
            File ldaZFile = new File(ldaFile);
            if (ldaZFile.exists()) {
                ldaZ = inputLDAInitialization(ldaFile);
            } else {
                lda.sample();
                ldaZ = lda.getZ();
                outputLDAInitialization(ldaFile, ldaZ);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(this.folder, "lda-topwords.txt"), 15);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        setLog(true);

        // initialize assignments
        if (verbose) {
            logln("--- LDA loaded. Start initializing assingments ...");
        }

        for (int d = 0; d < D; d++) {
            // create tables
            for (int k = 0; k < K; k++) {
                SHDPTable table = new SHDPTable(iter, k, null, d);
                this.localRestaurants[d].addTable(table);
            }

            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = localRestaurants[d].getTable(ldaZ[d][n]);
                localRestaurants[d].addCustomerToTable(n, z[d][n].getIndex());
            }

            // assign tables with global nodes
            ArrayList<Integer> emptyTables = new ArrayList<Integer>();
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    emptyTables.add(table.getIndex());
                    continue;
                }
                this.sampleDishForTable(d, table.getIndex(), !REMOVE, ADD, !OBSERVED, EXTEND);
            }

            // remove empty table
            for (int tIndex : emptyTables) {
                this.localRestaurants[d].removeTable(tIndex);
            }
        }

        // debug
        if (verbose) {
            logln("--- After assignment initialization\n"
                    + getCurrentState() + "\n");
        }

        // optimize
        optimize();
    }

    private int[][] inputLDAInitialization(String filepath) {
        if (verbose) {
            logln("--- --- LDA init file found. Loading from " + filepath);
        }

        int[][] ldaZ = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            int numDocs = Integer.parseInt(reader.readLine());
            ldaZ = new int[numDocs][];
            for (int d = 0; d < numDocs; d++) {
                String[] sline = reader.readLine().split("\t")[1].split(" ");
                ldaZ[d] = new int[sline.length];
                for (int n = 0; n < ldaZ[d].length; n++) {
                    ldaZ[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return ldaZ;
    }

    private void outputLDAInitialization(String filepath, int[][] z) {
        if (verbose) {
            logln("--- --- Outputing LDA init state to file " + filepath);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            writer.write(z.length + "\n");
            for (int d = 0; d < z.length; d++) {
                writer.write(z[d].length + "\t");
                for (int n = 0; n < z[d].length; n++) {
                    writer.write(z[d][n] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        try {
            if (report) {
                IOUtils.createFolder(reportFolderPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while creating report folder."
                    + " " + reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                String str = iter
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t # tokens change: " + numTokenAsgnsChange
                        + "\t # tables change: " + numTableAsgnsChange
                        + "\t # converge: " + numConverged
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + str);
                } else {
                    logln("--- Sampling. Iter " + str);
                }
            }

            numTableAsgnsChange = 0;
            numTokenAsgnsChange = 0;
            numConverged = 0;

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    this.sampleTableForToken(d, n, REMOVE, ADD, OBSERVED, EXTEND);
                }

                for (SHDPTable table : localRestaurants[d].getTables()) {
                    if (table.getContent() == null) {
                        throw new RuntimeException("Null dish. d = " + d + ". table. " + table.getTableId());
                    }
                }

                for (SHDPTable table : this.localRestaurants[d].getTables()) {
                    this.sampleDishForTable(d, table.getIndex(), REMOVE, ADD, OBSERVED, EXTEND);
                }
            }

            // optimize regression parameters of local restaurants
            this.optimize();

            if (verbose && iter % REP_INTERVAL == 0) {
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
            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            }
        }

        if (report) {
            outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
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
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Create a brand new dish
     */
    private SHDPDish createDish() {
        int newDishIndex = globalRestaurant.getNextTableIndex();
        DirichletMultinomialModel dm = new DirichletMultinomialModel(V, hyperparams.get(BETA), uniform);
        double dishEta = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
        int baseNumCusts = 0;
        SHDPDish newDish = new SHDPDish(iter, newDishIndex, dm, dishEta, baseNumCusts);
        globalRestaurant.addTable(newDish);
        return newDish;
    }

    /**
     * Remove a customer from a table. This also removes the corresponding
     * observation from the dish. If the table is empty after the removal, the
     * table will be removed as well.
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param n The customer
     */
    private void removeCustomerFromTable(int d, int tableIndex, int n) {
        SHDPTable table = this.localRestaurants[d].getTable(tableIndex);

        this.localRestaurants[d].removeCustomerFromTable(n, tableIndex);
        table.getContent().getContent().decrement(words[d][n]);

        if (table.isEmpty()) {
            this.removeTableFromDish(d, tableIndex, null);
            this.localRestaurants[d].removeTable(tableIndex);
        }
    }

    /**
     * Add a customer to a table. This also adds the corresponding observation
     * to the dish
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param n The customer
     */
    private void addCustomerToTable(int d, int tableIndex, int n) {
        SHDPTable table = this.localRestaurants[d].getTable(tableIndex);

        this.localRestaurants[d].addCustomerToTable(n, tableIndex);
        table.getContent().getContent().increment(words[d][n]);
    }

    /**
     * Remove a table from a dish
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param observations The set of observations currently being assigned to
     * this table
     */
    private void removeTableFromDish(int d, int tableIndex, HashMap<Integer, Integer> observations) {
        SHDPTable table = this.localRestaurants[d].getTable(tableIndex);
        SHDPDish dish = table.getContent();

        if (dish == null) {
            throw new RuntimeException("Removing table from dish. d = " + d
                    + ". tableIndex = " + tableIndex);
        }

        // remove observations from dish
        if (observations != null) {
            removeObservations(dish, observations);
        }

        // remove table from dish
        this.globalRestaurant.removeCustomerFromTable(table, dish.getIndex());

        // if the dish is empty, remove it
        if (dish.isEmpty()) {
            this.globalRestaurant.removeTable(dish.getIndex());
        }
    }

    /**
     * Remove observations form a global dish
     *
     * @param dish The dish
     * @param observations The set of observations to be removed
     */
    private void removeObservations(SHDPDish dish, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            if (dish == null) {
                System.out.println("dish null");
            } else if (dish.getContent() == null) {
                System.out.println("dish content null");
            }

            dish.getContent().changeCount(obs, -observations.get(obs));
        }
    }

    /**
     * Add observations to a global dish
     *
     * @param dish The dish
     * @param observations The set of observations to be added
     */
    private void addObservations(SHDPDish dish, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            dish.getContent().changeCount(obs, observations.get(obs));
        }
    }

    /**
     * Sample a table assignment for a token
     *
     * @param d Document index
     * @param n Token index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param resObserved Whether the response variable is observed
     * @param extend Whether the token should be added to the topic structure
     */
    private void sampleTableForToken(
            int d, int n,
            boolean remove, boolean add,
            boolean resObserved, boolean extend) {
        int curObs = words[d][n];
        SHDPTable curTable = z[d][n];

        if (remove) {
            removeCustomerFromTable(d, curTable.getIndex(), n);
        }

        if (this.localRestaurants[d].isEmpty()) {
            SHDPTable table = new SHDPTable(iter, 0, null, d);
            this.localRestaurants[d].addTable(table);
            z[d][n] = table;
            localRestaurants[d].addCustomerToTable(n, z[d][n].getIndex());
            return;
        }

        double preSum = 0.0;
        if (supervised && resObserved) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                preSum += table.getContent().getRegressionParameter()
                        * table.getNumCustomers();
            }
        }

        // for existing tables
        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();
        for (SHDPTable table : this.localRestaurants[d].getTables()) {
            double logPrior = Math.log(table.getNumCustomers());
            double wordLlh = table.getContent().getContent().getLogLikelihood(curObs);
            double lp = logPrior + wordLlh;

            if (supervised && resObserved) {
                double mean = (preSum
                        + table.getContent().getRegressionParameter()) / words[d].length;
                double resLlh = StatisticsUtils.logNormalProbability(responses[d],
                        mean, Math.sqrt(hyperparams.get(RHO)));
                lp += resLlh;
            }

            tableIndices.add(table.getIndex());
            logprobs.add(lp);
        }

        // for new table
        HashMap<Integer, Double> dishLogPriors = new HashMap<Integer, Double>();
        HashMap<Integer, Double> dishWordLlhs = new HashMap<Integer, Double>();
        HashMap<Integer, Double> dishResLlhs = new HashMap<Integer, Double>();
        if (extend) {
            dishLogPriors = getDishLogPriors();
            dishWordLlhs = getDishWordLogLikelihoods(curObs);

            if (dishLogPriors.size() != dishWordLlhs.size()) {
                throw new RuntimeException("Number of dishes mismatch");
            }

            if (supervised && resObserved) {
                dishResLlhs = getDishResponseLogLikelihoods(d, preSum);

                if (dishLogPriors.size() != dishResLlhs.size()) {
                    throw new RuntimeException("Number of dishes mismatch");
                }
            }

            double marginal = 0.0;
            for (int dishIndex : dishLogPriors.keySet()) {
                double lp = dishLogPriors.get(dishIndex) + dishWordLlhs.get(dishIndex);

                if (supervised && resObserved) {
                    lp += dishResLlhs.get(dishIndex);
                }

                if (marginal == 0.0) {
                    marginal = lp;
                } else {
                    marginal = SamplerUtils.logAdd(marginal, lp);
                }
            }

            double logPrior = Math.log(hyperparams.get(ALPHA_LOCAL));
            double lp = logPrior + marginal;

            tableIndices.add(PSEUDO_INDEX);
            logprobs.add(lp);
        }

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        int tableIndex = tableIndices.get(sampledIndex);

        if (curTable.getIndex() != tableIndex) {
            numTokenAsgnsChange++;
        }

        SHDPTable table;
        if (tableIndex == PSEUDO_INDEX) {
            int newTableIndex = this.localRestaurants[d].getNextTableIndex();
            table = new SHDPTable(iter, newTableIndex, null, d);
            localRestaurants[d].addTable(table);

            // sample dish
            SHDPDish dish;
            int dishIdx = sampleDish(dishLogPriors, dishWordLlhs, dishResLlhs, resObserved, extend);
            if (dishIdx == PSEUDO_INDEX) {
                dish = createDish();
            } else {
                dish = globalRestaurant.getTable(dishIdx);
            }

            if (dish == null) {
                throw new RuntimeException("Creating new table. d = " + d
                        + ". n = " + n
                        + ". dishIdx = " + dishIdx);
            }

            table.setContent(dish);

            globalRestaurant.addCustomerToTable(table, dish.getIndex());
        } else {
            table = this.localRestaurants[d].getTable(tableIndex);
        }

        z[d][n] = table;

        if (add) {
            addCustomerToTable(d, z[d][n].getIndex(), n);
        }

        if (table.getContent() == null) {
            throw new RuntimeException("Dish is null. d = " + d
                    + ". n = " + n
                    + ". table = " + table.getTableId());
        }
    }

    /**
     * Sample a dish assignment for a table
     *
     * @param d Document index
     * @param tableIndex Table index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param resObserved Whether the response variable is observed
     * @param extend Whether the table should be added to the topic structure
     */
    private void sampleDishForTable(
            int d, int tableIndex,
            boolean remove, boolean add,
            boolean resObserved, boolean extend) {
        SHDPTable curTable = localRestaurants[d].getTable(tableIndex);

        // current observations assigned to this table
        HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
        for (int n : curTable.getCustomers()) {
            int type = words[d][n];
            Integer count = observations.get(type);
            if (count == null) {
                observations.put(type, 1);
            } else {
                observations.put(type, count + 1);
            }
        }

        if (globalRestaurant.isEmpty()) {
            SHDPDish dish = createDish();
            curTable.setContent(dish);
            addObservations(dish, observations);
            globalRestaurant.addCustomerToTable(curTable, dish.getIndex());
            return;
        }

        int curDishIndex = PSEUDO_INDEX;
        if (curTable.getContent() != null) {
            curDishIndex = curTable.getContent().getIndex();
        }

        if (remove) {
            removeTableFromDish(d, tableIndex, observations);
        }

        double preSum = 0.0;
        if (supervised && resObserved) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                if (table.getIndex() == tableIndex) {
                    continue;
                }
                preSum += table.getContent().getRegressionParameter()
                        * table.getNumCustomers();
            }
        }

        HashMap<Integer, Double> dishLogPriors = getDishLogPriors();
        HashMap<Integer, Double> dishWordLlhs = getDishWordLogLikelihoods(observations);

        if (dishLogPriors.size() != dishWordLlhs.size()) {
            throw new RuntimeException("Numbers of dishes mismatch");
        }

        HashMap<Integer, Double> dishResLlhs = new HashMap<Integer, Double>();
        if (supervised && resObserved) {
            dishResLlhs = getDishResponseLogLikelihoods(d, preSum);

            if (dishLogPriors.size() != dishResLlhs.size()) {
                throw new RuntimeException("Numbers of dishes mismatch");
            }
        }

        int sampledDishIndex = sampleDish(dishLogPriors, dishWordLlhs, dishResLlhs, resObserved, extend);
        if (curDishIndex != sampledDishIndex) {
            numTableAsgnsChange++;
        }

        SHDPDish dish;
        if (sampledDishIndex == PSEUDO_INDEX) {
            dish = createDish();
        } else {
            dish = globalRestaurant.getTable(sampledDishIndex);
        }

        // update
        curTable.setContent(dish);

        if (add) {
            globalRestaurant.addCustomerToTable(curTable, dish.getIndex());
            addObservations(dish, observations);
        }
    }

    /**
     * Optimize the regression parameters using L-BFGS
     */
    private void optimize() {
        int numDishes = globalRestaurant.getNumTables();
        ArrayList<SHDPDish> dishes = new ArrayList<SHDPDish>();
        for (SHDPDish dish : globalRestaurant.getTables()) {
            dishes.add(dish);
        }

        // current regression parameters and priors
        double[] regParams = new double[numDishes];
        double[] priorMeans = new double[numDishes];
        double[] priorStdvs = new double[numDishes];

        for (int idx = 0; idx < dishes.size(); idx++) {
            SHDPDish dish = dishes.get(idx);
            regParams[idx] = dish.getRegressionParameter();
            priorMeans[idx] = hyperparams.get(MU);
            priorStdvs[idx] = Math.sqrt(hyperparams.get(SIGMA));
        }

        double[][] designMatrix = new double[D][numDishes];
        for (int d = 0; d < D; d++) {
            int[] dishCount = new int[numDishes];
            for (SHDPTable table : localRestaurants[d].getTables()) {
                int dishIdx = dishes.indexOf(table.getContent());
                dishCount[dishIdx] += table.getNumCustomers();
            }

            for (int i = 0; i < dishes.size(); i++) {
                designMatrix[d][i] = (double) dishCount[i] / words[d].length;
            }
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses,
                Math.sqrt(hyperparams.get(RHO)), priorMeans, priorStdvs);
        this.optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            // This exception may be thrown if L-BFGS
            //  cannot step in the current direction.
            // This condition does not necessarily mean that
            //  the optimizer has failed, but it doesn't want
            //  to claim to have succeeded... 
            // do nothing
        }

        if (converged) // if the optimization converges
        {
            numConverged++;
        }

        for (int i = 0; i < dishes.size(); i++) {
            dishes.get(i).setRegressionParameter(optimizable.getParameter(i));
        }
    }

    private int sampleDish(
            HashMap<Integer, Double> dishLogPriors,
            HashMap<Integer, Double> dishWordLlhs,
            HashMap<Integer, Double> dishResLlhs,
            boolean resObserved,
            boolean extend) {
        ArrayList<Integer> indices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();
        for (int idx : dishLogPriors.keySet()) {
            if (!extend && idx == PSEUDO_INDEX) {
                continue;
            }

            double lp = dishLogPriors.get(idx) + dishWordLlhs.get(idx);
            if (supervised && resObserved) {
                lp += dishResLlhs.get(idx);
            }

            indices.add(idx);
            logprobs.add(lp);
        }
        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
        return indices.get(sampledIdx);
    }

    private HashMap<Integer, Double> getDishLogPriors() {
        HashMap<Integer, Double> dishLogPriors = new HashMap<Integer, Double>();
        double normalizer = Math.log(this.globalRestaurant.getTotalNumCustomers()
                + hyperparams.get(ALPHA_GLOBAL));
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishLogPriors.put(dish.getIndex(), Math.log(dish.getNumCustomers()) - normalizer);
        }
        dishLogPriors.put(PSEUDO_INDEX, Math.log(hyperparams.get(ALPHA_GLOBAL)) - normalizer);
        return dishLogPriors;
    }

    private HashMap<Integer, Double> getDishWordLogLikelihoods(int observation) {
        HashMap<Integer, Double> dishLogLikelihoods = new HashMap<Integer, Double>();
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishLogLikelihoods.put(dish.getIndex(), dish.getContent().getLogLikelihood(observation));
        }
        dishLogLikelihoods.put(PSEUDO_INDEX, emptyModel.getLogLikelihood(observation));
        return dishLogLikelihoods;
    }

    private HashMap<Integer, Double> getDishWordLogLikelihoods(HashMap<Integer, Integer> observations) {
        HashMap<Integer, Double> dishLogLikelihoods = new HashMap<Integer, Double>();
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishLogLikelihoods.put(dish.getIndex(), dish.getContent().getLogLikelihood(observations));
        }
        dishLogLikelihoods.put(PSEUDO_INDEX, emptyModel.getLogLikelihood(observations));
        return dishLogLikelihoods;
    }

    private HashMap<Integer, Double> getDishResponseLogLikelihoods(int d, double preSum) {
        HashMap<Integer, Double> resLlhs = new HashMap<Integer, Double>();
        int tokenCount = words[d].length;

        // for existing dishes
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            double mean = (preSum + dish.getRegressionParameter()) / tokenCount;
            double var = hyperparams.get(RHO);
            double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
            resLlhs.put(dish.getIndex(), resLlh);
        }

        // for new dish
        double mean = (preSum + hyperparams.get(MU)) / tokenCount;
        double var = hyperparams.get(SIGMA) / (tokenCount * tokenCount) + hyperparams.get(RHO);
        double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
        resLlhs.put(PSEUDO_INDEX, resLlh);

        return resLlhs;
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(">>> >>> # dishes: ").append(globalRestaurant.getNumTables()).append("\n");

        int[] numTables = new int[D];
        for (int d = 0; d < D; d++) {
            numTables[d] = this.localRestaurants[d].getNumTables();
        }
        str.append(">>> >>> # tables")
                .append(". avg: ").append(MiscUtils.formatDouble(StatisticsUtils.mean(numTables)))
                .append(". min: ").append(StatisticsUtils.min(numTables))
                .append(". max: ").append(StatisticsUtils.max(numTables))
                .append(". sum: ").append(StatisticsUtils.sum(numTables));
        str.append("\n");
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double obsLlh = 0.0;
        for (SHDPDish dish : globalRestaurant.getTables()) {
            obsLlh += dish.getContent().getLogLikelihood();
        }

        double assignLp = globalRestaurant.getJointProbabilityAssignments(hyperparams.get(ALPHA_GLOBAL));
        for (int d = 0; d < D; d++) {
            assignLp += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA_LOCAL));
        }

        double dishRegLlh = 0.0;
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishRegLlh += StatisticsUtils.logNormalProbability(dish.getRegressionParameter(),
                    hyperparams.get(MU), Math.sqrt(hyperparams.get(SIGMA)));
        }

        double resLlh = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            resLlh += StatisticsUtils.logNormalProbability(responses[d], regValues[d], Math.sqrt(hyperparams.get(RHO)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** obs llh: " + MiscUtils.formatDouble(obsLlh)
                    + ". res llh: " + MiscUtils.formatDouble(resLlh)
                    + ". assignments: " + MiscUtils.formatDouble(assignLp)
                    + ". global reg: " + MiscUtils.formatDouble(dishRegLlh));
        }

        return obsLlh + assignLp + dishRegLlh + resLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        double obsLlh = 0.0;
        for (SHDPDish dish : globalRestaurant.getTables()) {
            obsLlh += dish.getContent().getLogLikelihood(newParams.get(BETA), uniform);
        }

        double assignLp = globalRestaurant.getJointProbabilityAssignments(newParams.get(ALPHA_GLOBAL));
        for (int d = 0; d < D; d++) {
            assignLp += localRestaurants[d].getJointProbabilityAssignments(newParams.get(ALPHA_LOCAL));
        }

        double dishRegLlh = 0.0;
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishRegLlh += StatisticsUtils.logNormalProbability(dish.getRegressionParameter(),
                    newParams.get(MU), Math.sqrt(newParams.get(SIGMA)));
        }

        double resLlh = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            resLlh += StatisticsUtils.logNormalProbability(responses[d], regValues[d], Math.sqrt(newParams.get(RHO)));
        }

        return obsLlh + assignLp + dishRegLlh + resLlh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        for (SHDPDish dish : globalRestaurant.getTables()) {
            dish.getContent().setConcentration(newParams.get(BETA));
        }

        this.hyperparams = new ArrayList<Double>();
        for (double param : newParams) {
            this.hyperparams.add(param);
        }
    }

    @Override
    public void validate(String msg) {
        if (verbose) {
            logln(">>> >>> Validating " + msg);
        }

        globalRestaurant.validate(msg);
        for (int d = 0; d < D; d++) {
            localRestaurants[d].validate(msg);
        }

        for (int d = 0; d < D; d++) {
            for (SHDPTable table : localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    throw new RuntimeException(msg
                            + ". Empty table. " + table.toString());
                }

                if (table.getContent() == null) {
                    throw new RuntimeException(msg
                            + ". Null dish on table " + table.getIndex()
                            + " born at " + table.getIterationCreated());
                }
            }
        }

        for (SHDPDish dish : globalRestaurant.getTables()) {
            if (dish.isEmpty() || dish.getContent().getCountSum() == 0) {
                throw new RuntimeException(msg + ". Empty dish. " + dish.toString()
                        + ". tables: " + dish.getCustomers().toString());
            }
        }

        int totalObs = 0;
        for (SHDPDish dish : globalRestaurant.getTables()) {
            int dishNumObs = dish.getContent().getCountSum();
            int tableNumObs = 0;
            for (SHDPTable table : dish.getCustomers()) {
                tableNumObs += table.getNumCustomers();
            }

            if (dishNumObs != tableNumObs) {
                throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                        + dishNumObs + " vs. " + tableNumObs);
            }

            totalObs += dishNumObs;
        }

        if (totalObs != numTokens) {
            throw new RuntimeException(msg + ". Total numbers of observations mismatch. "
                    + totalObs + " vs. " + numTokens);
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
            modelStr.append(globalRestaurant.getNumTables()).append("\n");
            for (SHDPDish dish : globalRestaurant.getTables()) {
                modelStr.append(dish.getIndex()).append("\n");
                modelStr.append(dish.getIterationCreated()).append("\n");
                modelStr.append(dish.getNumCustomers()).append("\n");
                modelStr.append(dish.getRegressionParameter()).append("\n");
                modelStr.append(DirichletMultinomialModel.output(dish.getContent())).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d)
                        .append("\t").append(localRestaurants[d].getNumTables())
                        .append("\n");
                for (SHDPTable table : localRestaurants[d].getTables()) {
                    assignStr.append(table.getIndex()).append("\n");
                    assignStr.append(table.getIterationCreated()).append("\n");
                    assignStr.append(table.getContent().getIndex()).append("\n");

                    assignStr.append(Integer.toString(table.getNumCustomers()));
                    for (int n : table.getCustomers()) {
                        assignStr.append("\t").append(n);
                    }
                    assignStr.append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(d).append(":").append(n)
                            .append("\t").append(z[d][n].getIndex())
                            .append("\n");
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

    /**
     * Load the model from a compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        // initialize
        this.initializeModelStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
        BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
        int numDishes = Integer.parseInt(reader.readLine());
        for (int i = 0; i < numDishes; i++) {
            int dishIdx = Integer.parseInt(reader.readLine());
            int iterCreated = Integer.parseInt(reader.readLine());
            int numCusts = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirichletMultinomialModel dmm = DirichletMultinomialModel.input(reader.readLine());

            SHDPDish dish = new SHDPDish(iterCreated, dishIdx, dmm, regParam, numCusts);
            globalRestaurant.addTable(dish);
        }
        reader.close();

        globalRestaurant.fillInactiveTableIndices();
    }

    /**
     * Load the assignments of the training data from the compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
        BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
        for (int d = 0; d < D; d++) {
            String[] sline = reader.readLine().split("\t");
            if (d != Integer.parseInt(sline[0])) {
                throw new RuntimeException("Mismatch");
            }
            int numTables = Integer.parseInt(sline[1]);
            for (int t = 0; t < numTables; t++) {
                int tableIdx = Integer.parseInt(reader.readLine());
                int iterCreated = Integer.parseInt(reader.readLine());
                int dishIdx = Integer.parseInt(reader.readLine());

                SHDPDish dish = globalRestaurant.getTable(dishIdx);
                SHDPTable table = new SHDPTable(iterCreated, tableIdx, dish, d);
                globalRestaurant.addCustomerToTable(table, dishIdx);
                localRestaurants[d].addTable(table);

                sline = reader.readLine().split("\t");
                int numCusts = Integer.parseInt(sline[0]);
                for (int i = 1; i < sline.length; i++) {
                    localRestaurants[d].addCustomerToTable(Integer.parseInt(sline[i]), tableIdx);
                }

                if (table.getNumCustomers() != numCusts) {
                    throw new RuntimeException("Numbers of customers mismatch");
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                String[] sline = reader.readLine().split("\t");
                if (!sline[0].equals(d + ":" + n)) {
                    throw new RuntimeException("Mismatch");
                }
                z[d][n] = localRestaurants[d].getTable(Integer.parseInt(sline[1]));
            }
        }
        reader.close();

        for (int d = 0; d < D; d++) {
            localRestaurants[d].fillInactiveTableIndices();
        }
    }

    public void outputTopicTopWords(String outputFile, int numWords)
            throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing top words to file " + outputFile);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (SHDPDish dish : globalRestaurant.getTables()) {
            String[] topWords = getTopWords(dish.getContent().getDistribution(), numWords);
            writer.write("[" + dish.getIndex()
                    + ", " + dish.getIterationCreated()
                    + ", " + dish.getNumCustomers()
                    + ", " + MiscUtils.formatDouble(dish.getRegressionParameter())
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");
        }
        writer.close();
    }

    public void outputTopicCoherence(
            String filepath,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            System.out.println("Outputing topic coherence to file " + filepath);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (SHDPDish dish : globalRestaurant.getTables()) {
            double[] distribution = dish.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(dish.getIndex()
                    + "\t" + dish.getNumCustomers()
                    + "\t" + dish.getContent().getCountSum()
                    + "\t" + MiscUtils.formatDouble(score));
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    private double computeRegressionSum(int d) {
        double regSum = 0.0;
        for (SHDPTable table : localRestaurants[d].getTables()) {
            regSum += table.getNumCustomers() * table.getContent().getRegressionParameter();
        }
        return regSum;
    }

    /**
     * Predict the response values using the current model
     */
    public double[] getRegressionValues() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            regValues[d] = computeRegressionSum(d) / words[d].length;
        }
        return regValues;
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
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

        double[] finalPredResponses = new double[D];
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

    public void regressNewDocuments(
            int[][] newWords) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        String iterPredFolderPath = this.folder + this.getSamplerFolder() + IterPredictionFolder;
        IOUtils.createFolder(iterPredFolderPath);

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            regressNewDocuments(
                    reportFolderpath + filename,
                    newWords,
                    iterPredFolderPath + IOUtils.removeExtension(filename) + ".txt");
        }
    }

    private double[] regressNewDocuments(
            String stateFile,
            int[][] newWords,
            String outputResultFile) throws Exception {
        if (verbose) {
            logln("Perform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        words = newWords;
        responses = null; // for evaluation
        D = words.length;
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
        }

        logln("--- V = " + V);
        logln("--- # documents = " + D); // number of groups
        logln("--- # tokens = " + numTokens);

        // initialize structure for test data
        initializeDataStructure();

        if (verbose) {
            logln("Initialized structure\n" + getCurrentState());
        }

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                SHDPTable table = new SHDPTable(iter, n, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(n, table.getIndex());
                z[d][n] = table;

                sampleDishForTable(d, table.getIndex(), !REMOVE, ADD, !OBSERVED, !EXTEND);
            }
        }

        if (verbose) {
            logln("Initialized assignments\n" + getCurrentState());
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleTableForToken(d, n, REMOVE, ADD, !OBSERVED, !EXTEND);
                }

                for (SHDPTable table : localRestaurants[d].getTables()) {
                    sampleDishForTable(d, table.getIndex(), REMOVE, ADD, !OBSERVED, !EXTEND);
                }
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);
            }
        }

        // averaging prediction responses over time
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }

        // output result during test time
        GibbsRegressorUtils.outputSingleModelPredictions(new File(outputResultFile), predResponsesList);
        return finalPredResponses;
    }

    class SHDPDish extends Table<SHDPTable, DirichletMultinomialModel> {

        private final int born;
        private final int baseNumCustomers; // number of customers from training
        private double regParam;

        public SHDPDish(int born, int index, DirichletMultinomialModel content, double mean,
                int baseNumCusts) {
            super(index, content);
            this.regParam = mean;
            this.born = born;
            this.baseNumCustomers = baseNumCusts;
        }

        @Override
        public int getNumCustomers() {
            return super.getNumCustomers() + this.baseNumCustomers;
        }

        public int getIterationCreated() {
            return this.born;
        }

        public double getRegressionParameter() {
            return regParam;
        }

        public void setRegressionParameter(double mean) {
            this.regParam = mean;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append(index)
                    .append(". #c: ").append(getNumCustomers())
                    .append(". #o: ").append(content.getCountSum())
                    .append(". mean: ").append(MiscUtils.formatDouble(regParam));
            return str.toString();
        }
    }

    class SHDPTable extends Table<Integer, SHDPDish> {

        private final int born;
        private int restIndex;

        public SHDPTable(int born, int index, SHDPDish dish, int restIndex) {
            super(index, dish);
            this.born = born;
            this.restIndex = restIndex;
        }

        public int getIterationCreated() {
            return this.born;
        }

        public String getTableId() {
            return restIndex + ":" + index;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append(restIndex).append("-").append(index)
                    .append(". #c = ").append(getNumCustomers())
                    .append(". -> ").append(content.getIndex());
            return str.toString();
        }
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

            // model hyperparameters
            addOption("alpha-global", "Hyperparameter of global DP");
            addOption("alpha-local", "Hyperparameter of local DP");
            addOption("beta", "Hyperparameter of the symmetric Dirichlet prior "
                    + "for word distributions");
            addOption("mu", "Prior mean of regression parameters");
            addOption("sigma", "Prior variance of regression parameters");
            addOption("rho", "Variance of the response variable");

            // running configurations
            addOption("cv-folder", "Cross validation folder");
            addOption("num-folds", "Number of folds");
            addOption("fold", "The cross-validation fold to run");
            addOption("run-mode", "Running mode");

            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("z", false, "standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp dist/segan.jar sampler.supervised.regression.SHDP -help", options);
                return;
            }

            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
            }

        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar sampler.supervised.regression.SLDA -help", options);
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

        SingleResponseTextDataset data = new SingleResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        data.prepareTopicCoherence(numTopWords);


        // hyperparameters
        double alpha_global = CLIUtils.getDoubleArgument(cmd, "alpha-global", 0.1);
        double alpha_local = CLIUtils.getDoubleArgument(cmd, "alpha-local", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double[] responses = data.getResponses();
        if (cmd.hasOption("z")) {
            ZNormalizer zNorm = new ZNormalizer(responses);
            for (int i = 0; i < responses.length; i++) {
                responses[i] = zNorm.normalize(responses[i]);
            }
        }

        double meanResponse = StatisticsUtils.mean(responses);
        double stddevResponse = StatisticsUtils.standardDeviation(responses);

        double mu = CLIUtils.getDoubleArgument(cmd, "mu", meanResponse);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", stddevResponse);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);

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

            File foldFolder = new File(resultFolder, fold.getFoldFolder());

            SHDP sampler = new SHDP();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setLog(true);
            sampler.setReport(true);
            sampler.setWordVocab(data.getWordVocab());

            // training data
            ArrayList<Integer> trInstIndices = fold.getTrainingInstances();
            int[][] trRevWords = data.getDocWords(trInstIndices);
            double[] trResponses = data.getResponses(trInstIndices);

            // test data
            ArrayList<Integer> teInstIndices = fold.getTestingInstances();
            int[][] teRevWords = data.getDocWords(teInstIndices);
            double[] teResponses = data.getResponses(teInstIndices);

            sampler.configure(foldFolder.getAbsolutePath(), trRevWords, trResponses,
                    data.getWordVocab().size(), alpha_global, alpha_local, beta,
                    mu, sigma, rho,
                    initState, paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);

            String samplerFolder = new File(foldFolder, sampler.getSamplerFolder()).getAbsolutePath();
            File iterPredFolder = sampler.getIterationPredictionFolder();
            IOUtils.createFolder(samplerFolder);

            if (runMode.equals("train")) {
                sampler.initialize();
                sampler.iterate();
                sampler.outputTopicTopWords(samplerFolder + TopWordFile, numTopWords);
                sampler.outputTopicCoherence(samplerFolder + TopicCoherenceFile, data.getTopicCoherence());
            } else if (runMode.equals("test")) {
                sampler.regressNewDocuments(teRevWords);

                File teResultFolder = new File(samplerFolder, "te-results");
                IOUtils.createFolder(teResultFolder);
                GibbsRegressorUtils.evaluate(iterPredFolder, teResultFolder, teResponses);
            } else if (runMode.equals("train-test")) {
                // train
                sampler.initialize();
                sampler.iterate();
                sampler.outputTopicTopWords(samplerFolder + TopWordFile, numTopWords);
                sampler.outputTopicCoherence(samplerFolder + TopicCoherenceFile, data.getTopicCoherence());

                // test
                sampler.regressNewDocuments(teRevWords);
                File teResultFolder = new File(samplerFolder, "te-results");
                IOUtils.createFolder(teResultFolder);
                GibbsRegressorUtils.evaluate(iterPredFolder, teResultFolder, teResponses);
            } else {
                throw new RuntimeException("Run mode " + runMode + " not supported");
            }
        }
    }
}
