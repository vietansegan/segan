package sampler.supervised.regression.author;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import data.AuthorResponseTextDataset;
import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBVar;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import regression.Regressor;
import util.PredictionUtils;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class AuthorSLDA extends AbstractSampler implements Regressor<AuthorResponseTextDataset> {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int MU = 2;
    public static final int SIGMA = 3;
    public static final int RHO = 4;
    // input data
    protected int[][] words;  // [D] x [N_d]: words
    protected int[] authors; // [D]: author of each document
    protected double[] responses; // [A]: response variables of each author
    protected int[][] authorDocIndices; // [A] x [D_a]
    private int V;
    private int D;
    private int A;
    private int K;
    private DirMult[] topicWords;
    private DirMult[] docTopics;
    private int[][] z;
    private double[] authorPredValues;
    private double[] docWeights;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    protected double[] regParams;
    private int numTokensChanged;
    private int tokenCount;
    private int[] docTokenCounts;
    private ArrayList<String> authorVocab;
    private ArrayList<double[]> regressionParameters;

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public ArrayList<String> getAuthorVocab() {
        return this.authorVocab;
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
            logln("--- num speakers:\t" + A);
            logln("--- num word types:\t" + V);
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
            logln("--- # tokens:\t" + tokenCount);

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
                .append("_author-sLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    @Override
    public void train(AuthorResponseTextDataset trainData) {
        train(trainData.getWords(), trainData.getAuthors(), trainData.getAuthorResponses());
    }

    public void train(int[][] ws, int[] as, double[] rs) {
        this.words = ws;
        this.authors = as;
        this.responses = rs;
        this.A = this.responses.length;
        this.D = this.words.length;

        // statistics
        tokenCount = 0;
        docTokenCounts = new int[D];

        for (int d = 0; d < D; d++) {
            tokenCount += words[d].length;
            docTokenCounts[d] += words[d].length;
        }

        // author document list
        ArrayList<Integer>[] authorDocList = new ArrayList[A];
        for (int a = 0; a < A; a++) {
            authorDocList[a] = new ArrayList<Integer>();
        }
        for (int d = 0; d < D; d++) {
            if (docTokenCounts[d] > 0) { // skip if this document is empty
                authorDocList[authors[d]].add(d);
            }
        }
        this.authorDocIndices = new int[A][];
        for (int a = 0; a < A; a++) {
            this.authorDocIndices[a] = new int[authorDocList[a].size()];
            for (int dd = 0; dd < this.authorDocIndices[a].length; dd++) {
                this.authorDocIndices[a][dd] = authorDocList[a].get(dd);
            }
        }
    }

    @Override
    public void test(AuthorResponseTextDataset testData) {
        test(testData.getWords(), testData.getAuthors(), testData.getAuthorVocab().size());
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \t" + getCurrentState());
            getLogLikelihood();
            evaluateRegressPrediction(responses, authorPredValues);
        }
    }

    private void updateAuthorPredictedValues() {
        authorPredValues = new double[A];
        for (int a = 0; a < A; a++) {
            for (int d : authorDocIndices[a]) {
                double[] docEmpDist = docTopics[d].getEmpiricalDistribution();
                for (int k = 0; k < K; k++) {
                    authorPredValues[a] += docWeights[d] * regParams[k] * docEmpDist[k];
                }
            }
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

    private void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        regParams = new double[K];
        for (int k = 0; k < K; k++) {
            regParams[k] = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
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

        this.docWeights = new double[D];
        for(int a=0; a<A; a++) {
            int totalLength = 0;
            for(int d : authorDocIndices[a]){
                totalLength += words[d].length;
            }
            
            for(int d : authorDocIndices[a]) {
                docWeights[d] = (double)words[d].length / totalLength;
            }
        }

        this.authorPredValues = new double[A];
    }

    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                initializePresetAssignments();
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
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        lda.configure(folder, words, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        try {
            File ldaFile = new File(lda.getSamplerFolderPath(), "author-model.zip");
            if (ldaFile.exists()) {
                lda.inputState(ldaFile);
            } else {
                lda.sample();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaFile);
            }
            lda.setWordVocab(wordVocab);
            lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while initializing topics with LDA");
        }
        setLog(true);

        // initialize assignments
        int[][] ldaZ = lda.getZ();
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = ldaZ[d][n];
                docTopics[d].increment(z[d][n]);
                topicWords[z[d][n]].increment(words[d][n]);
            }
        }

        // initialize regression parameters
        for (int k = 0; k < K; k++) {
            regParams[k] = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
        }
        updateRegressionParameters();
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
            System.arraycopy(regParams, 0, rp, 0, K);
            this.regressionParameters.add(rp);

            if (verbose && iter % REP_INTERVAL == 0) {
                String str = "Iter " + iter
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                }
            }

            // update the regression parameters
            updateRegressionParameters();

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
                evaluateRegressPrediction(responses, authorPredValues);

                logln("--- --- # tokens: " + tokenCount
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / tokenCount
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
            throw new RuntimeException("Exception at iteration " + iter);
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
        int author = authors[d];
        if (removeFromModel) {
            topicWords[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            docTopics[d].decrement(z[d][n]);
            authorPredValues[author] -= docWeights[d] * regParams[z[d][n]] / words[d].length;
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k] =
                    docTopics[d].getLogLikelihood(k)
                    + topicWords[k].getLogLikelihood(words[d][n]);
            if (observe) {
                double mean = authorPredValues[author] + docWeights[d] * regParams[k] / words[d].length;
                logprobs[k] += StatisticsUtils.logNormalProbability(responses[author],
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
            authorPredValues[author] += docWeights[d] * regParams[z[d][n]] / words[d].length;
        }
    }

    private void updateRegressionParameters() {
        double[][] designMatrix = new double[A][K];
        for (int a = 0; a < A; a++) {
            for (int d : authorDocIndices[a]) {
                double[] docEmpTopicDist = docTopics[d].getEmpiricalDistribution();

                for (int k = 0; k < K; k++) {
                    designMatrix[a][k] += docWeights[d] * docEmpTopicDist[k];
                }
            }
        }

//        GurobiMLRL2Norm mlr =
//                new GurobiMLRL2Norm(designMatrix, responses);
//        mlr.setMean(hyperparams.get(MU));
//        mlr.setRho(hyperparams.get(RHO));
//        mlr.setSigma(hyperparams.get(SIGMA));
//        regParams = mlr.solve();

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses,
                hyperparams.get(RHO),
                hyperparams.get(MU),
                hyperparams.get(SIGMA));

        this.optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // if the number of observations is less than or equal to the number of parameters
        if (verbose && converged) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int i = 0; i < regParams.length; i++) {
            regParams[i] = optimizable.getParameter(i);
        }

        updateAuthorPredictedValues();
    }

    private void updateDocumentWeights() {
        try {
            for (int a = 0; a < A; a++) {
                updateAuthorDocumentWeights(a);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while updating document weights");
        }
    }

    private void updateAuthorDocumentWeights(int a) throws Exception {
        GRBEnv env = new GRBEnv();
        GRBModel model = new GRBModel(env);

        // add variables
        GRBVar[] weightVar = new GRBVar[V];
        for (int d = 0; d < authorDocIndices[a].length; d++) {
            weightVar[d] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "var-" + d);
        }

        // objective function
        GRBLinExpr obj = new GRBLinExpr();
        obj.addConstant(-responses[a]);
        for (int d = 0; d < authorDocIndices[a].length; d++) {
            int docIdx = authorDocIndices[a][d];
            double[] empDist = docTopics[docIdx].getEmpiricalDistribution();
            double dotprod = StatisticsUtils.dotProduct(empDist, regParams);

            obj.addTerm(dotprod, weightVar[d]);
        }
        model.setObjective(obj, GRB.MINIMIZE);

        // constraints
        GRBLinExpr sumExpr = new GRBLinExpr();
        for (int d = 0; d < authorDocIndices[a].length; d++) {
            GRBLinExpr expr = new GRBLinExpr();
            expr.addTerm(1.0, weightVar[d]);
            model.addConstr(expr, GRB.GREATER_EQUAL, 0.0, "pc" + d);

            sumExpr.addTerm(1.0, weightVar[d]);
        }
        model.addConstr(sumExpr, GRB.EQUAL, 1.0, "sum");

        // optimize
        model.optimize();

        // get solution
        for (int d = 0; d < authorDocIndices[a].length; d++) {
            int docIdx = authorDocIndices[a][d];
            docWeights[docIdx] = weightVar[d].get(GRB.DoubleAttr.X);
        }

        // dispose of model and environment
        model.dispose();
        env.dispose();
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
        for (int a = 0; a < A; a++) {
            responseLlh += StatisticsUtils.logNormalProbability(
                    responses[a],
                    authorPredValues[a],
                    Math.sqrt(hyperparams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatisticsUtils.logNormalProbability(
                    regParams[k],
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
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
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
                modelStr.append(regParams[k]).append("\n");
                modelStr.append(DirMult.output(topicWords[k])).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(docWeights[d]).append("\n");
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
                regParams[k] = Double.parseDouble(reader.readLine());
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
                docWeights[d] = Double.parseDouble(reader.readLine());
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

    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        ArrayList<RankingItem<Integer>> sortedTopics = new ArrayList<RankingItem<Integer>>();
        for (int k = 0; k < K; k++) {
            sortedTopics.add(new RankingItem<Integer>(k, regParams[k]));
        }
        Collections.sort(sortedTopics);

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int ii = 0; ii < K; ii++) {
            int k = sortedTopics.get(ii).getObject();
            double[] distrs = topicWords[k].getDistribution();
            String[] topWords = getTopWords(distrs, numTopWords);
            writer.write("[" + k
                    + ", " + topicWords[k].getCountSum()
                    + ", " + MiscUtils.formatDouble(regParams[k])
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");
        }
        writer.close();
    }
    private int testBurnIn = BURN_IN;
    private int testMaxIter = MAX_ITER;
    private int testSampleLag = LAG;

    public void setTestConfigurations(int tBurnIn, int tMaxIter, int tSampleLag) {
        this.testBurnIn = tBurnIn;
        this.testMaxIter = tMaxIter;
        this.testSampleLag = tSampleLag;
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }

    public void test(int[][] newWords, int[] newAuthors, int numAuthors) {
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
                        newWords, newAuthors, numAuthors,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    private void sampleNewDocuments(
            String stateFile,
            int[][] newWords,
            int[] newAuthors,
            int numAuthors,
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
        authors = newAuthors;
        responses = null; // for evaluation
        D = words.length;
        A = numAuthors;

        // statistics
        tokenCount = 0;
        docTokenCounts = new int[D];

        for (int d = 0; d < D; d++) {
            tokenCount += words[d].length;
            docTokenCounts[d] += words[d].length;
        }

        // author document list
        ArrayList<Integer>[] authorDocList = new ArrayList[A];
        for (int a = 0; a < A; a++) {
            authorDocList[a] = new ArrayList<Integer>();
        }
        for (int d = 0; d < D; d++) {
            if (docTokenCounts[d] > 0) { // skip if this document is empty
                authorDocList[authors[d]].add(d);
            }
        }
        this.authorDocIndices = new int[A][];
        for (int a = 0; a < A; a++) {
            this.authorDocIndices[a] = new int[authorDocList[a].size()];
            for (int dd = 0; dd < this.authorDocIndices[a].length; dd++) {
                this.authorDocIndices[a][dd] = authorDocList[a].get(dd);
            }
        }

        // initialize structure
        initializeDataStructure();

        if (verbose) {
            logln("test data");
            logln("--- V = " + V);
            logln("--- D = " + D);
            logln("--- A = " + A);
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

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleZ(d, n, !REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
            }
        }

        updateAuthorPredictedValues();

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

            updateAuthorPredictedValues();

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                if (verbose) {
                    logln("--- iter = " + iter + " / " + this.testMaxIter);
                }
                double[] predResponses = new double[A];
                System.arraycopy(authorPredValues, 0, predResponses, 0, A);
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
        PredictionUtils.outputSingleModelPredictions(
                new File(outputResultFile),
                predResponsesList);
    }
}
