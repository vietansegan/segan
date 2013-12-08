package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import core.crossvalidation.Fold;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import optimization.GurobiMLRL2Norm;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import regression.Regressor;
import regression.RegressorUtils;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import util.CLIUtils;
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
public class LexSLDA extends AbstractSampler implements Regressor<ResponseTextDataset> {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int MU = 2;
    public static final int SIGMA = 3;
    public static final int RHO = 4;
    protected int K;
    protected int V;
    protected int D;
    // inputs
    protected int[][] words;
    protected double[] responses;
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    // optimization
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private double[] docRegressMeans;
    // topical regression
    protected double[] topicParams;
    // lexical regression
    private boolean lexReg = false;
    private double lexMean;
    private double lexSigma;
    private double[] lexicalParams;
    private double[][] docLexDesignMatrix;
    // internal
    private int numTokensChanged = 0;
    private int numTokens = 0;
    private ArrayList<double[]> regressionParameters;

    public void setLexicalRegression(double lexMean, double lexSigma) {
        this.lexReg = true;
        this.lexMean = lexMean;
        this.lexSigma = lexSigma;
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double mu, // mean of Gaussian for regression parameters
            double sigma, // stadard deviation of Gaussian for regression parameters
            double rho, // standard deviation of Gaussian for document observations
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
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
        this.regressionParameters = new ArrayList<double[]>();
        this.setName();

        if (!debug) {
            System.err.close();
        }

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(hyperparams.get(MU)));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(hyperparams.get(SIGMA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(hyperparams.get(RHO)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- # tokens:\t" + numTokens);

            logln("--- responses:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatisticsUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatisticsUtils.standardDeviation(responses)));
            int[] histogram = StatisticsUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
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
                .append("_sLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)));
        if (lexReg) {
            str.append("_lex-").append(formatter.format(lexMean))
                    .append("-").append(formatter.format(lexSigma));
        }
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void train(int[][] ws, double[] rs) {
        this.words = ws;
        this.responses = rs;
        this.D = this.words.length;

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }
    }

    @Override
    public void train(ResponseTextDataset trainData) {
        train(trainData.getWords(), trainData.getResponses());
    }

    @Override
    public void test(ResponseTextDataset testData) {
        test(testData.getWords());
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        // compute the design matrix for lexical regression
        if (lexReg) {
            this.docLexDesignMatrix = new double[D][V];
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    this.docLexDesignMatrix[d][words[d][n]]++;
                }
                for (int v = 0; v < V; v++) {
                    this.docLexDesignMatrix[d][v] /= words[d].length;
                }
            }
        }

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        // optimize regression parameters
        updateJointRegressionParameters();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            this.evaluateRegressPrediction(responses, docRegressMeans);
        }
    }
    
    private void evaluateRegressPrediction(double[] trueVals, double[] predVals) {
        RegressionEvaluation eval = new RegressionEvaluation(trueVals, predVals);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeRSquared();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
        }
    }

    @Override
    public void sample() {
        this.initialize();
        this.iterate();
    }

    private void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        topicParams = new double[K];
        for (int k = 0; k < K; k++) {
            topicParams[k] = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
        }

        if (lexReg) {
            lexicalParams = new double[V];
            for (int v = 0; v < V; v++) {
                lexicalParams[v] = SamplerUtils.getGaussian(lexMean, lexSigma);
            }
        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            case FORWARD:
                initializeForwardAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                docTopics[d].increment(z[d][n]);
                topicWords[z[d][n]].increment(words[d][n]);
            }
        }
    }

    private void initializeForwardAssignments() {
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleZ(d, n, !REMOVE, ADD, !REMOVE, ADD, OBSERVED);
            }
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
        double lda_alpha = hyperparams.get(ALPHA);
        double lda_beta = hyperparams.get(BETA);

        lda.configure(folder, words, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            File ldaZFile = new File(lda.getSamplerFolderPath(), "model.zip");
            if (ldaZFile.exists()) {
                lda.inputState(ldaZFile);
            } else {
                lda.initialize();
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaZFile);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
            ldaZ = lda.getZ();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        setLog(true);

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = ldaZ[d][n];
                docTopics[d].increment(z[d][n]);
                topicWords[z[d][n]].increment(words[d][n]);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

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

        RegressionEvaluation eval;
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            // store regression parameters after every iteration
            double[] rp = new double[K];
            System.arraycopy(topicParams, 0, rp, 0, K);
            this.regressionParameters.add(rp);

            if (verbose && iter % REP_INTERVAL == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\n" + getCurrentState());
                }
            }

            // sample topic assignments
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                }
            }

            // update the regression parameters
            updateJointRegressionParameters();

            // parameter optimization
            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        for (double p : sparams) {
                            System.out.println(p);
                        }
                    }
                }
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                eval = new RegressionEvaluation(responses, docRegressMeans);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();

                logln("--- --- After updating regression parameters Zs:\t" + getCurrentState());
                for (Measurement measurement : measurements) {
                    logln("--- --- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }

                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n");
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            }
        }

        if (report) { // output the final model
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
     * Sample topic assignment for a token
     *
     * @param d Document index
     * @param n Token index
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe Whether the response variable of this document is observed
     */
    private void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (removeFromModel) {
            topicWords[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            docTopics[d].decrement(z[d][n]);
            docRegressMeans[d] -= topicParams[z[d][n]] / words[d].length;
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k] =
                    docTopics[d].getLogLikelihood(k)
                    + topicWords[k].getLogLikelihood(words[d][n]);
            if (observe) {
                double mean = docRegressMeans[d] + topicParams[k] / (words[d].length);
                logprobs[k] += StatisticsUtils.logNormalProbability(responses[d],
                        mean, Math.sqrt(hyperparams.get(RHO)));
            }
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        if (z[d][n] != sampledZ) {
            numTokensChanged++; // for debugging
        }
        // update
        z[d][n] = sampledZ;

        if (addToModel) {
            topicWords[z[d][n]].increment(words[d][n]);
        }
        if (addToData) {
            docTopics[d].increment(z[d][n]);
            docRegressMeans[d] += topicParams[z[d][n]] / words[d].length;
        }
    }
    
    private void updateRegressionParameters() {
        // update topic regression parameters
        // update lexical regression parameters
    }

    private void updateJointRegressionParameters() {
        double[][] designMatrix;
        if (lexReg) {
            designMatrix = new double[D][K + V];
            for (int d = 0; d < D; d++) {
                double[] docEmpDist = docTopics[d].getEmpiricalDistribution();
                System.arraycopy(docEmpDist, 0, designMatrix[d], 0, K);
                System.arraycopy(docLexDesignMatrix[d], 0, designMatrix[d], K, V);
            }
        } else {
            designMatrix = new double[D][K];
            for (int d = 0; d < D; d++) {
                designMatrix[d] = docTopics[d].getEmpiricalDistribution();
            }
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setRho(hyperparams.get(RHO));

        double[] means;
        double[] sigmas;
        if (lexReg) {
            means = new double[K + V];
            sigmas = new double[K + V];
            for (int k = 0; k < K; k++) {
                means[k] = hyperparams.get(MU);
                sigmas[k] = hyperparams.get(SIGMA);
            }
            for (int v = 0; v < V; v++) {
                means[v + K] = lexMean;
                sigmas[v + K] = lexSigma;
            }
            mlr.setMeans(means);
            mlr.setSigmas(sigmas);

            double[] params = mlr.solve();
            System.arraycopy(params, 0, topicParams, 0, K);
            for (int v = 0; v < V; v++) {
                lexicalParams[v] = params[v + K];
            }
        } else {
            mlr.setMean(hyperparams.get(MU));
            mlr.setSigma(hyperparams.get(SIGMA));

            topicParams = mlr.solve();
        }

        // update current predictions
        this.docRegressMeans = new double[D];
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            for (int k = 0; k < K; k++) {
                this.docRegressMeans[d] += topicParams[k] * empDist[k];
            }

            if (lexReg) {
                for (int v = 0; v < V; v++) {
                    this.docRegressMeans[d] += lexicalParams[v] * docLexDesignMatrix[d][v];
                }
            }
        }
    }

    /**
     * Update the regression parameters using L-BFGS. This is to perform a full
     * optimization procedure until convergence.
     */
    private void updateRegressionParametersLBFGS() {
        double[][] designMatrix = new double[D][K];
        for (int d = 0; d < D; d++) {
            designMatrix[d] = docTopics[d].getEmpiricalDistribution();
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                topicParams, designMatrix, responses,
                (hyperparams.get(RHO)),
                hyperparams.get(MU),
                hyperparams.get(SIGMA));

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
            ex.printStackTrace();
        }

        if (verbose) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int i = 0; i < topicParams.length; i++) {
            topicParams[i] = optimizable.getParameter(i);
        }
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood();
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood();
        }

        double responseLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            double mean = StatisticsUtils.dotProduct(topicParams, empDist);
            responseLlh += StatisticsUtils.logNormalProbability(
                    responses[d],
                    mean,
                    Math.sqrt(hyperparams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatisticsUtils.logNormalProbability(
                    topicParams[k],
                    hyperparams.get(MU),
                    Math.sqrt(hyperparams.get(SIGMA)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". response: " + MiscUtils.formatDouble(responseLlh)
                    + ". regParam: " + MiscUtils.formatDouble(regParamLlh));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }

        double responseLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            double mean = StatisticsUtils.dotProduct(topicParams, empDist);
            responseLlh += StatisticsUtils.logNormalProbability(responses[d], mean,
                    Math.sqrt(newParams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatisticsUtils.logNormalProbability(topicParams[k],
                    newParams.get(MU), Math.sqrt(newParams.get(SIGMA)));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        for (int k = 0; k < K; k++) {
            topicWords[k].setConcentration(newParams.get(BETA) * V);
        }
        for (int d = 0; d < D; d++) {
            docTopics[d].setCenterElement(newParams.get(ALPHA) * K);
        }

        this.hyperparams = new ArrayList<Double>();
        for (double param : newParams) {
            this.hyperparams.add(param);
        }
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            docTopics[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            topicWords[k].validate(msg);
        }
    }

    @Override
    public void output(File samplerFile) {
        this.outputState(samplerFile.getAbsolutePath());
    }

    @Override
    public void input(File samplerFile) {
        this.inputModel(samplerFile.getAbsolutePath());
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(topicParams[k]).append("\n");
                modelStr.append(DirMult.output(topicWords[k])).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");

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
            for (int k = 0; k < K; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                topicParams[k] = Double.parseDouble(reader.readLine());
                topicWords[k] = DirMult.input(reader.readLine());
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
                docTopics[d] = DirMult.input(reader.readLine());

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

    public void outputLexicalRegressionParameters(File file) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }
        if (verbose) {
            logln("Outputing lexical regression parameters to " + file);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        writer.write(lexicalParams.length + "\n");
        for (int v = 0; v < lexicalParams.length; v++) {
            writer.write(wordVocab.get(v) + "\t" + lexicalParams[v] + "\n");
        }
        writer.close();
    }

    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        ArrayList<RankingItem<Integer>> sortedTopics = new ArrayList<RankingItem<Integer>>();
        for (int k = 0; k < K; k++) {
            sortedTopics.add(new RankingItem<Integer>(k, topicParams[k]));
        }
        Collections.sort(sortedTopics);

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int ii = 0; ii < K; ii++) {
            int k = sortedTopics.get(ii).getObject();
            double[] distrs = topicWords[k].getDistribution();
            String[] topWords = getTopWords(distrs, numTopWords);
            writer.write("[" + k
                    + ", " + topicWords[k].getCountSum()
                    + ", " + MiscUtils.formatDouble(topicParams[k])
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
        for (int k = 0; k < K; k++) {
            double[] distribution = this.topicWords[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + topicWords[k].getCountSum()
                    + "\t" + MiscUtils.formatDouble(score));
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void outputDocTopicDistributions(File file) throws Exception {
        if (verbose) {
            logln("Outputing per-document topic distribution to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int d = 0; d < D; d++) {
            writer.write(Integer.toString(d));
            double[] docTopicDist = this.docTopics[d].getDistribution();
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
            double[] topicWordDist = this.topicWords[k].getDistribution();
            for (int v = 0; v < V; v++) {
                writer.write("\t" + topicWordDist[v]);
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void outputTopicRegressionParameters(File file) throws Exception {
        if (verbose) {
            logln("Outputing per-topic regression parameters to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            writer.write(k + "\t" + this.topicParams[k] + "\n");
        }
        writer.close();
    }

    public double[] getRegressionValues() {
        double[] predValues = new double[D];
        for (int d = 0; d < D; d++) {
            double[] empiricalTopicDist = docTopics[d].getEmpiricalDistribution();
            predValues[d] = StatisticsUtils.dotProduct(empiricalTopicDist, topicParams);

            if (lexReg) {
                for (int v = 0; v < V; v++) {
                    predValues[d] += lexicalParams[v] * docLexDesignMatrix[d][v];
                }
            }
        }
        return predValues;
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

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }

    public void testFinal(int[][] newWords) {
        if (verbose) {
            logln("Test sampling ...");
        }

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        String finalFile = "iter-" + iter;
        File finalZipFile = new File(reportFolderPath, finalFile + ".zip");

        this.setTestConfigurations(BURN_IN, MAX_ITER, LAG);

        File iterPredFolder = getIterationPredictionFolder();
        IOUtils.createFolder(iterPredFolder);
        File partialResultFile = new File(iterPredFolder, finalFile + ".txt");

        try {
            sampleNewDocuments(
                    finalZipFile.getAbsolutePath(),
                    newWords,
                    partialResultFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while testing using final model");
        }
    }

    public void test(int[][] newWords) {
        if (verbose) {
            logln("Test sampling ...");
        }
        this.setTestConfigurations(BURN_IN, MAX_ITER, LAG);
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        File iterPredFolder = getIterationPredictionFolder();
        IOUtils.createFolder(iterPredFolder);

        try {
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File partialResultFile = new File(iterPredFolder, IOUtils.removeExtension(filename) + ".txt");
                sampleNewDocuments(
                        new File(reportFolder, filename).getAbsolutePath(),
                        newWords,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }
    private int testBurnIn = BURN_IN;
    private int testMaxIter = MAX_ITER;
    private int testSampleLag = LAG;

    public void setTestConfigurations(int tBurnIn, int tMaxIter, int tSampleLag) {
        this.testBurnIn = tBurnIn;
        this.testMaxIter = tMaxIter;
        this.testSampleLag = tSampleLag;
    }

    /**
     * Perform sampling on test documents using a single model learned during
     * training time.
     *
     * @param stateFile The state file of the trained model
     * @param newWords Test documents
     * @param outputResultFile Prediction file
     */
    private void sampleNewDocuments(
            String stateFile,
            int[][] newWords,
            String outputResultFile) throws Exception {
        if (verbose) {
            System.out.println();
            logln("Perform regression using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        words = newWords;
        responses = null; // for evaluation
        D = words.length;

        // initialize structure
        initializeDataStructure();

        if (verbose) {
            logln("test data");
            logln("--- V = " + V);
            logln("--- D = " + D);
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k = 0; k < topicWords.length; k++) {
                topicWordCount += topicWords[k].getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // initialize
        if (lexReg) {
            this.docLexDesignMatrix = new double[D][V];
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    this.docLexDesignMatrix[d][words[d][n]]++;
                }
                for (int v = 0; v < V; v++) {
                    this.docLexDesignMatrix[d][v] /= words[d].length;
                }
            }
        }
        this.docRegressMeans = new double[D];
        for (int d = 0; d < D; d++) {
            if (lexReg) {
                for (int v = 0; v < V; v++) {
                    this.docRegressMeans[d] += lexicalParams[v] * docLexDesignMatrix[d][v];
                }
            }
        }

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleZ(d, n, !REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
            }
        }

        if (verbose) {
            logln("After initialization");
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k = 0; k < topicWords.length; k++) {
                topicWordCount += topicWords[k].getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < this.testMaxIter; iter++) {
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, !REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
                }
            }

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                if (verbose) {
                    logln("--- iter = " + iter + " / " + this.testMaxIter);
                }

                double[] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);
            }
        }

        if (verbose) {
            logln("After iterating");
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k = 0; k < topicWords.length; k++) {
                topicWordCount += topicWords[k].getCountSum();
            }

            logln("\t--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("\t--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // output result during test time
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        RegressorUtils.outputSingleModelPredictions(
                new File(outputResultFile),
                predResponsesList);
    }
    // End prediction ----------------------------------------------------------

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        if (K < 10) { // only print out when number of topics is small
            str.append(MiscUtils.arrayToString(topicParams)).append("\n");
            for (int k = 0; k < K; k++) {
                str.append(topicWords[k].getCountSum()).append(", ");
            }
            str.append("\n");
        }
        return str.toString();
    }

    public double[][] getDocumentTopicDistributions() {
        double[][] docTopicDists = new double[this.D][];
        for (int d = 0; d < this.D; d++) {
            docTopicDists[d] = this.docTopics[d].getDistribution();
        }
        return docTopicDists;
    }

    public double[][] getTopicWordDistributions() {
        double[][] topicWordDists = new double[this.K][];
        for (int k = 0; k < K; k++) {
            topicWordDists[k] = this.topicWords[k].getDistribution();
        }
        return topicWordDists;
    }

    public void outputTopWords(String outputFile, ArrayList<String> vocab, int numWords)
            throws Exception {
        System.out.println("Outputing top words to file " + outputFile);
        double[][] topicWordDistr = new double[K][V];
        for (int k = 0; k < K; k++) {
            topicWordDistr[k] = topicWords[k].getDistribution();
        }
        IOUtils.outputTopWords(topicWordDistr, vocab, numWords, outputFile);
    }

    public void outputRegressionParameters(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (double[] rp : this.regressionParameters) {
            for (double r : rp) {
                writer.write(MiscUtils.formatDouble(r) + "\t");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void outputTopicsWithRegressionParameters(String outputFile,
            ArrayList<String> vocab,
            int numWords) throws Exception {
        System.out.println("Outputing topics with regression parameters to file " + outputFile);
        double[][] topicWordDistr = new double[K][V];
        for (int k = 0; k < K; k++) {
            topicWordDistr[k] = topicWords[k].getDistribution();
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int t = 0; t < topicWordDistr.length; t++) {
            // sort words
            double[] bs = topicWordDistr[t];
            ArrayList<RankingItem<Integer>> rankedWords = new ArrayList<RankingItem<Integer>>();
            for (int i = 0; i < bs.length; i++) {
                rankedWords.add(new RankingItem<Integer>(i, bs[i]));
            }
            Collections.sort(rankedWords);

            // output top words
            writer.write("Topic " + (t + 1));
            writer.write(" (" + MiscUtils.formatDouble(topicParams[t])
                    + "; " + topicWords[t].getCountSum()
                    + ") ");
            for (int i = 0; i < Math.min(numWords, vocab.size()); i++) {
                writer.write("\t" + vocab.get(rankedWords.get(i).getObject()));
            }
            writer.write("\n\n");
        }
        writer.close();
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar:dist/lib/*' " + LexSLDA.class.getName() + " -help";
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

            // model hyperparameters
            addOption("alpha", "Hyperparameter of the symmetric Dirichlet prior "
                    + "for topic distributions");
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
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
                runModel();
            }

        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    private static void runCrossValidation() throws Exception {
        String cvFolder = cmd.getOptionValue("cv-folder");
        int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));
        String runMode = cmd.getOptionValue("run-mode");
        String resultFolder = cmd.getOptionValue("output");
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        int foldIndex = -1;
        if (cmd.hasOption("fold")) {
            foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
        }

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);

        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");

        InitialState initState = InitialState.RANDOM;

        for (int ii = 0; ii < numFolds; ii++) {
            if (foldIndex != -1 && ii != foldIndex) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + foldIndex);
            }

            Fold fold = new Fold(ii, cvFolder);
            File foldFolder = new File(resultFolder, fold.getFoldName());
            ResponseTextDataset[] foldData = ResponseTextDataset.loadCrossValidationFold(fold);
            ResponseTextDataset trainData = foldData[Fold.TRAIN];
            ResponseTextDataset devData = foldData[Fold.DEV];
            ResponseTextDataset testData = foldData[Fold.TEST];

            if (cmd.hasOption("z")) {
                ResponseTextDataset.zNormalize(trainData, devData, testData);
            }

            if (verbose) {
                System.out.println("Fold " + fold.getFoldName());
                System.out.println("--- training: " + trainData.toString());
                System.out.println("--- development: " + devData.toString());
                System.out.println("--- test: " + testData.toString());
                System.out.println();
            }

            double meanResponse = StatisticsUtils.mean(trainData.getResponses());
            double stddevResponse = StatisticsUtils.standardDeviation(trainData.getResponses());
            double mu = CLIUtils.getDoubleArgument(cmd, "mu", meanResponse);
            double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", stddevResponse);
            double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);

            LexSLDA sampler = new LexSLDA();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setLog(true);
            sampler.setReport(true);
            sampler.setWordVocab(trainData.getWordVocab());

            sampler.train(trainData);
            sampler.configure(foldFolder.getAbsolutePath(),
                    trainData.getWordVocab().size(), K,
                    alpha, beta,
                    mu, sigma, rho,
                    initState, paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);

            File samplerFolder = new File(foldFolder, sampler.getSamplerFolder());
            File iterPredFolder = sampler.getIterationPredictionFolder();
            IOUtils.createFolder(samplerFolder);

            if (runMode.equals("train")) {
                sampler.sample();
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                sampler.outputDocTopicDistributions(new File(samplerFolder, "tr-doc-topic.txt"));
                sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                sampler.outputTopicRegressionParameters(new File(samplerFolder, "topic-reg-params.txt"));
            } else if (runMode.equals("test")) {
                sampler.test(testData);
                File teResultFolder = new File(samplerFolder, "te-results");
                IOUtils.createFolder(teResultFolder);
                RegressorUtils.evaluate(iterPredFolder, teResultFolder, testData.getDocIds(), testData.getResponses());
                sampler.outputDocTopicDistributions(new File(samplerFolder, "te-doc-topic.txt"));
            } else if (runMode.equals("train-test")) {
                sampler.sample();
                sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                sampler.outputDocTopicDistributions(new File(samplerFolder, "tr-doc-topic.txt"));
                sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                sampler.outputTopicRegressionParameters(new File(samplerFolder, "topic-reg-params.txt"));

                sampler.test(testData);
                File teResultFolder = new File(samplerFolder, "te-results");
                IOUtils.createFolder(teResultFolder);
                RegressorUtils.evaluate(iterPredFolder, teResultFolder, testData.getDocIds(), testData.getResponses());
                sampler.outputDocTopicDistributions(new File(samplerFolder, "te-doc-topic.txt"));
            } else {
                throw new RuntimeException("Run mode " + runMode + " is not supported");
            }
        }
    }

    /**
     * Run a model on a dataset. This is mainly used for exploratory analysis.
     */
    private static void runModel() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String outputFolder = cmd.getOptionValue("output");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);

        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");
        InitialState initState = InitialState.RANDOM;

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        ResponseTextDataset data = new ResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
        data.prepareTopicCoherence(numTopWords);

        int V = data.getWordVocab().size();
        if (cmd.hasOption("z")) {
            data.zNormalize();
        }

        double meanResponse = StatisticsUtils.mean(data.getResponses());
        double stddevResponse = StatisticsUtils.standardDeviation(data.getResponses());

        double mu = CLIUtils.getDoubleArgument(cmd, "mu", meanResponse);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", stddevResponse);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);

        if (verbose) {
            System.out.println("Running SLDA ...");
        }
        LexSLDA sampler = new LexSLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(data.getWordVocab());
        sampler.train(data);
        sampler.configure(outputFolder,
                V, K, alpha, beta, mu, sigma, rho, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File sldaFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(sldaFolder);
        sampler.sample();
        sampler.outputTopicTopWords(new File(sldaFolder, TopWordFile), numTopWords);
        sampler.outputTopicCoherence(new File(sldaFolder, TopicCoherenceFile), data.getTopicCoherence());
        sampler.outputDocTopicDistributions(new File(sldaFolder, "doc-topic.txt"));
        sampler.outputTopicWordDistributions(new File(sldaFolder, "topic-word.txt"));
        sampler.outputTopicRegressionParameters(new File(sldaFolder, "topic-reg-params.txt"));
    }
}
