/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.LDASampler;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirichletMultinomialModel;
import sampling.util.Restaurant;
import sampling.util.Table;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class SHDPSampler extends AbstractSampler {

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
    private int numTokenAssignmentsChange;
    private int numTableAssignmentsChange;
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
        LDASampler lda = new LDASampler();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        if (K == 0) // this is not set
        {
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
                lda.outputTopicTopWords(this.folder + "lda-topwords.txt", 15);
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

        try {
            if (report) {
                IOUtils.createFolder(this.folder + this.getSamplerFolder() + ReportFolder);
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

            if (verbose && iter % REP_INTERVAL == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens change: " + numTokenAssignmentsChange
                            + "\t # tables change: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens change: " + numTokenAssignmentsChange
                            + "\t # tables change: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState());
                }
            }

            numTableAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;
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
                outputState(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + ".zip");
            }
        }

        outputState(this.folder + this.getSamplerFolder() + "final.zip");

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(this.folder + this.getSamplerFolder() + "hyperparameters.txt");
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
                double mean = (preSum + table.getContent().getRegressionParameter()) / words[d].length;
                double resLlh = StatisticsUtils.logNormalProbability(responses[d], mean, Math.sqrt(hyperparams.get(RHO)));
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
            numTokenAssignmentsChange++;
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
            numTableAssignmentsChange++;
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

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + ModelFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));

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

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + AssignmentFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));

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

    /**
     * Perform regression on test documents in the same groups as in the
     * training data.
     *
     * @param newWords New documents
     * @param newResponses The true new responses. This is used to evaluate the
     * predicted values.
     */
//    public double[] regressNewDocuments(int[][] newWords, String filepath) throws Exception{
//        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
//        File reportFolder = new File(reportFolderpath);
//        if(!reportFolder.exists())
//            throw new RuntimeException("Report folder does not exist");
//        String[] filenames = reportFolder.list();
//        
//        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
//        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
//        for(int i=0; i<filenames.length; i++){
//            String filename = filenames[i];
//            if(!filename.contains("zip"))
//                continue;
//            
//            double[] predResponses = regressNewDocuments(reportFolderpath 
//                    + filename, newWords);
//            predResponsesList.add(predResponses);
//            
//            RegressionEvaluation eval = new RegressionEvaluation(
//                        responses, predResponses);
//            eval.computeCorrelationCoefficient();
//            eval.computeMeanSquareError();
//            eval.computeRSquared();
//            ArrayList<Measurement> measurements = eval.getMeasurements();
//            
//            // output results
//            if(i == 0){
//                writer.write("Model");
//                for(Measurement measurement : measurements)
//                    writer.write("\t" + measurement.getName());
//                writer.write("\n");
//            }
//            writer.write(filename);
//            for(Measurement measurement : measurements)
//                writer.write("\t" + measurement.getValue());
//            writer.write("\n");
//            
//            if(verbose){
//                logln("Model from " + reportFolderpath + filename);
//                for(Measurement measurement : measurements)
//                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
//                System.out.println();
//            }
//        }
//        writer.close();
//        
//        // average predicted response over different models
//        double[] finalPredResponses = new double[D];
//        for(int d=0; d<D; d++){
//            double sum = 0.0;
//            for(int i=0; i<predResponsesList.size(); i++)
//                sum += predResponsesList.get(i)[d];
//            finalPredResponses[d] = sum / predResponsesList.size();
//        }
//        return finalPredResponses;
//    }
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

    /**
     * Perform regression on test documents in the same groups as in the
     * training data.
     */
//    public double[] regressNewDocuments(
//            int[][] newWords, 
//            String predFilepath
//            ) throws Exception{
//        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
//        File reportFolder = new File(reportFolderpath);
//        if(!reportFolder.exists())
//            throw new RuntimeException("Report folder does not exist");
//        String[] filenames = reportFolder.list();
//        
//        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
//        ArrayList<String> modelList = new ArrayList<String>();
//        
//        for(int i=0; i<filenames.length; i++){
//            String filename = filenames[i];
//            if(!filename.contains("zip"))
//                continue;
//            
//            double[] predResponses = regressNewDocuments(
//                    reportFolderpath + filename, newWords);
//            predResponsesList.add(predResponses);
//            modelList.add(filename);
//        }
//        
//        // write prediction values
//        BufferedWriter writer = IOUtils.getBufferedWriter(predFilepath);
//        for(String model : modelList) // header
//            writer.write(model + "\t");
//        writer.write("\n");
//        
//        for(int r=0; r<newWords.length; r++){
//            for(int m=0; m<predResponsesList.size(); m++)
//                writer.write(predResponsesList.get(m)[r] + "\t");
//            writer.write("\n");
//        }
//        writer.close();
//        
//        // average predicted response over different models
//        double[] finalPredResponses = new double[D];
//        for(int d=0; d<D; d++){
//            double sum = 0.0;
//            for(int i=0; i<predResponsesList.size(); i++)
//                sum += predResponsesList.get(i)[d];
//            finalPredResponses[d] = sum / predResponsesList.size();
//        }
//        return finalPredResponses;
//    }
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
        BufferedWriter writer = IOUtils.getBufferedWriter(outputResultFile);
        for (int d = 0; d < D; d++) {
            writer.write(Integer.toString(d));

            for (int ii = 0; ii < predResponsesList.size(); ii++) {
                writer.write("\t" + predResponsesList.get(ii)[d]);
            }
            writer.write("\n");
        }
        writer.close();

        return finalPredResponses;
    }

    private double[][] loadSingleIterationPredictions(String filepath, int numDocs) throws Exception {
        double[][] preds = new double[numDocs][];
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            double[] ps = new double[sline.length - 1];
            for (int ii = 0; ii < ps.length; ii++) {
                ps[ii] = Double.parseDouble(sline[ii + 1]);
            }
            preds[count] = ps;
            count++;
        }
        reader.close();
        return preds;
    }

    public void computeSingleFinal(String filepath, double[] trueResponses) throws Exception {
        if (verbose) {
            logln("Computing single-final result. " + filepath + "single-final.txt");
        }

        String iterPredFolderPath = this.folder + this.getSamplerFolder() + IterPredictionFolder;
        File iterPredFolder = new File(iterPredFolderPath);
        if (!iterPredFolder.exists()) {
            throw new RuntimeException("Prediction folder does not exist");
        }
        String[] filenames = iterPredFolder.list();

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath + "single-final.txt");
        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];

            double[][] predictions = loadSingleIterationPredictions(iterPredFolderPath + filename, trueResponses.length);

            // get the predictions at the final iterations during test time
            double[] finalPred = new double[predictions.length];
            for (int d = 0; d < finalPred.length; d++) {
                finalPred[d] = predictions[d][predictions[0].length - 1];
            }
            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, finalPred);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            writer.write(filename);
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void computeSingleAverage(String filepath, double[] trueResponses) throws Exception {
        if (verbose) {
            logln("Computing single-average result. " + filepath + "single-avg.txt");
        }

        String iterPredFolderPath = this.folder + this.getSamplerFolder() + IterPredictionFolder;
        File iterPredFolder = new File(iterPredFolderPath);
        if (!iterPredFolder.exists()) {
            throw new RuntimeException("Prediction folder does not exist");
        }
        String[] filenames = iterPredFolder.list();

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath + "single-avg.txt");
        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];

            double[][] predictions = loadSingleIterationPredictions(iterPredFolderPath + filename, trueResponses.length);

            // compute the prediction values as the average values
            double[] avgPred = new double[predictions.length];
            for (int d = 0; d < avgPred.length; d++) {
                avgPred[d] = StatisticsUtils.mean(predictions[d]);
            }

            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, avgPred);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            writer.write(filename);
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void computeMultipleFinal(String filepath, double[] trueResponses) throws Exception {
        if (verbose) {
            logln("Computing multiple-final result. " + filepath + "multiple-final.txt");
        }

        String iterPredFolderPath = this.folder + this.getSamplerFolder() + IterPredictionFolder;
        File iterPredFolder = new File(iterPredFolderPath);
        if (!iterPredFolder.exists()) {
            throw new RuntimeException("Prediction folder does not exist");
        }
        String[] filenames = iterPredFolder.list();

        double[] predResponses = new double[trueResponses.length];
        int numModels = filenames.length;

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];

            double[][] predictions = loadSingleIterationPredictions(iterPredFolderPath + filename, trueResponses.length);

            for (int d = 0; d < trueResponses.length; d++) {
                predResponses[d] += predictions[d][predictions[0].length - 1];
            }
        }

        for (int d = 0; d < predResponses.length; d++) {
            predResponses[d] /= numModels;
        }

        RegressionEvaluation eval = new RegressionEvaluation(
                trueResponses, predResponses);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeRSquared();

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath + "multiple-final.txt");
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            writer.write("\t" + measurement.getValue());
        }
        writer.write("\n");
        writer.close();
    }

    public void computeMultipleAverage(String filepath, double[] trueResponses) throws Exception {
        if (verbose) {
            logln("Computing multiple-avg result. " + filepath + "multiple-avg.txt");
        }

        String iterPredFolderPath = this.folder + this.getSamplerFolder() + IterPredictionFolder;
        File iterPredFolder = new File(iterPredFolderPath);
        if (!iterPredFolder.exists()) {
            throw new RuntimeException("Prediction folder does not exist");
        }
        String[] filenames = iterPredFolder.list();

        double[] predResponses = new double[trueResponses.length];
        int numModels = filenames.length;

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];

            double[][] predictions = loadSingleIterationPredictions(iterPredFolderPath + filename, trueResponses.length);

            for (int d = 0; d < trueResponses.length; d++) {
                predResponses[d] += StatisticsUtils.mean(predictions[d]);
            }
        }

        for (int d = 0; d < predResponses.length; d++) {
            predResponses[d] /= numModels;
        }

        RegressionEvaluation eval = new RegressionEvaluation(
                trueResponses, predResponses);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeRSquared();

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath + "multiple-avg.txt");
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            writer.write("\t" + measurement.getValue());
        }
        writer.write("\n");
        writer.close();
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
}
