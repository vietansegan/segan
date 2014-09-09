package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractSampler;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import optimization.RidgeRegressionLBFGS;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;

/**
 * Implementation of Supervised Latent Dirichlet Allocation (SLDA).
 *
 * @author vietan
 */
public class SLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected double rho;
    protected double mu;
    protected double sigma;
    // inputs
    protected int[][] words; // original documents
    protected double[] responses; // 
    protected ArrayList<Integer> docIndices; // [D]: indices of considered docs
    protected int K;
    protected int V;
    // derive
    protected int D;
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    protected double[] regParams;
    // optimization
    protected double[] docRegressMeans;
    protected SparseVector[] designMatrix;
    // internal
    protected int numTokensChanged;
    protected int numTokens;
    protected double sqrtRho;
    protected boolean zNorm;

    public SLDA() {
        this.basename = "SLDA";
    }

    public SLDA(String bname) {
        this.basename = bname;
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho, // variance of Gaussian for document observations
            double mu, // mean of Gaussian for regression parameters
            double sigma, // variance of Gaussian for regression parameters
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

        this.rho = rho;
        this.mu = mu;
        this.sigma = sigma;
        this.sqrtRho = Math.sqrt(this.rho);

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

        this.report = true;

        if (verbose && folder != null) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- vocab size:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta: \t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- topic mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- topic sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_" + this.basename
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(hyperparams.get(ALPHA))
                + "_b-" + formatter.format(hyperparams.get(BETA))
                + "_r-" + formatter.format(rho)
                + "_m-" + formatter.format(mu)
                + "_s-" + formatter.format(sigma)
                + "_opt-" + this.paramOptimized;
    }

    public void setZNormalizer(boolean zNorm) {
        this.zNorm = zNorm;
    }

    public DirMult[] getTopicWords() {
        return this.topicWords;
    }

    public int[][] getZs() {
        return this.z;
    }

    public double[] getRegressionParameters() {
        return this.regParams;
    }

    protected void evaluateRegressPrediction(double[] trueVals, double[] predVals) {
        RegressionEvaluation eval = new RegressionEvaluation(trueVals, predVals);
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

    /**
     * Set training data.
     *
     * @param docWords All documents
     * @param docIndices Indices of documents under consideration
     * @param docResponses Responses of all documents
     */
    public void train(int[][] docWords, ArrayList<Integer> docIndices,
            double[] docResponses) {
        this.words = docWords;
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();
        this.responses = new double[D]; // responses of considered documents
        this.numTokens = 0;
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.numTokens += this.words[dd].length;
            this.responses[ii] = docResponses[dd];
        }

        if (zNorm) {
            ZNormalizer zNormalizer = new ZNormalizer(responses);
            this.responses = zNormalizer.normalize(responses);
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
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

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        // compute the design matrix for lexical regression
        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        // optimize regression parameters
        updateTopicRegressionParameters();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            evaluateRegressPrediction(responses, docRegressMeans);
        }
    }

    protected void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        regParams = new double[K];
        for (int k = 0; k < K; k++) {
            regParams[k] = SamplerUtils.getGaussian(mu, sigma);
        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            z[ii] = new int[words[dd].length];
        }

        docTopics = new DirMult[D];
        for (int ii = 0; ii < D; ii++) {
            docTopics[ii] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        docRegressMeans = new double[D];
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[ii][nn] = rand.nextInt(K);
                docTopics[ii].increment(z[ii][nn]);
                topicWords[z[ii][nn]].increment(words[dd][nn]);
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

        lda.configure(folder, V, K, lda_alpha, lda_beta, initState, paramOptimized,
                lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            File ldaZFile = new File(lda.getSamplerFolderPath(), basename + ".zip");
            lda.train(words, docIndices);
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
            ldaZ = lda.getZs();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(log);

        // initialize assignments
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int n = 0; n < words[dd].length; n++) {
                z[ii][n] = ldaZ[ii][n];
                docTopics[ii].increment(z[ii][n]);
                topicWords[z[ii][n]].increment(words[dd][n]);
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

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;
            if (isReporting()) {
                // store llh after every iteration
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            // update the regression parameters
            updateTopicRegressionParameters();

            // parameter optimization
            if (iter % LAG == 0 && iter > BURN_IN) {
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
                evaluateRegressPrediction(responses, docRegressMeans);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n");
                System.out.println();
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
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
    }

    /**
     * Sample topic assignments for all tokens. This is a bit faster than using
     * sampleZ.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe Whether the response variable of this document is observed
     */
    protected void sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        double totalBeta = V * hyperparams.get(BETA);
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int nn = 0; nn < words[dd].length; nn++) {
                if (removeFromModel) {
                    topicWords[z[ii][nn]].decrement(words[dd][nn]);
                }
                if (removeFromData) {
                    docTopics[ii].decrement(z[ii][nn]);
                    docRegressMeans[ii] -= regParams[z[ii][nn]] / words[dd].length;
                }

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    logprobs[k] = Math.log(docTopics[ii].getCount(k) + hyperparams.get(ALPHA))
                            + Math.log((topicWords[k].getCount(words[dd][nn]) + hyperparams.get(BETA))
                            / (topicWords[k].getCountSum() + totalBeta));
                    if (observe) {
                        double mean = docRegressMeans[ii] + regParams[k] / words[dd].length;
                        logprobs[k] += StatUtils.logNormalProbability(responses[ii], mean, sqrtRho);
                    }
                }

                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

                if (z[ii][nn] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[ii][nn] = sampledZ;

                if (addToModel) {
                    topicWords[z[ii][nn]].increment(words[dd][nn]);
                }
                if (addToData) {
                    docTopics[ii].increment(z[ii][nn]);
                    docRegressMeans[ii] += regParams[z[ii][nn]] / words[dd].length;
                }
            }
        }
    }

    /**
     * Update regression parameters by optimizing using L-BFGS.
     */
    private void updateTopicRegressionParameters() {
        designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            designMatrix[ii] = new SparseVector(K);
            for (int k : docTopics[ii].getSparseCounts().getIndices()) {
                double val = (double) docTopics[ii].getCount(k) / z[ii].length;
                designMatrix[ii].change(k, val);
            }
        }

        RidgeRegressionLBFGS optimizable = new RidgeRegressionLBFGS(
                responses, regParams, designMatrix, rho, mu, sigma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (isReporting()) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int kk = 0; kk < K; kk++) {
            regParams[kk] = optimizable.getParameter(kk);
        }

        // update current predictions
        this.docRegressMeans = new double[D];
        for (int ii = 0; ii < D; ii++) {
            for (int kk : designMatrix[ii].getIndices()) {
                this.docRegressMeans[ii] += designMatrix[ii].get(kk) * regParams[kk];
            }
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
        for (int ii = 0; ii < D; ii++) {
            double[] empDist = docTopics[ii].getEmpiricalDistribution();
            double mean = StatUtils.dotProduct(regParams, empDist);
            responseLlh += StatUtils.logNormalProbability(responses[ii],
                    mean, sqrtRho);
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[k], mu, Math.sqrt(sigma));
        }

        if (isReporting()) {
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
        for (int ii = 0; ii < D; ii++) {
            topicLlh += docTopics[ii].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }

        double responseLlh = 0.0;
        for (int ii = 0; ii < D; ii++) {
            double[] empDist = docTopics[ii].getEmpiricalDistribution();
            double mean = StatUtils.dotProduct(regParams, empDist);
            responseLlh += StatUtils.logNormalProbability(responses[ii],
                    mean, sqrtRho);
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[k], mu, Math.sqrt(sigma));
        }

        double llh = wordLlh + topicLlh + responseLlh + regParamLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.docTopics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topicWords[k].setConcentration(this.hyperparams.get(BETA) * V);
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
            for (int ii = 0; ii < D; ii++) {
                assignStr.append(ii).append("\n");
                assignStr.append(DirMult.output(docTopics[ii])).append("\n");

                for (int n = 0; n < z[ii].length; n++) {
                    assignStr.append(z[ii][n]).append("\t");
                }
                assignStr.append("\n");
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
            logln("--- Reading state from " + filepath);
        }

        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + filepath);
        }

        validate("Done reading state from " + filepath);
    }

    protected void inputModel(String zipFilepath) {
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

    protected void inputAssignments(String zipFilepath) throws Exception {
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
                for (int n = 0; n < z[d].length; n++) {
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

    public void outputTopicTopWords(File file, int numTopWords) {
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

        try {
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
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + file);
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + SLDA.class.getName() + " -help";
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

        // data output
        addOption("output-folder", "Output folder");

        // sampling
        addSamplingOptions();

        // parameters
        addOption("alpha", "Alpha");
        addOption("beta", "Beta");
        addOption("rho", "Rho");
        addOption("mu", "Mu");
        addOption("sigma", "Sigma");
        addOption("K", "Number of topics");
        addOption("num-top-words", "Number of top words per topic");

        // configurations
        addOption("init", "Initialization");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "z-normalize");
        options.addOption("help", false, "Help");
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
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 1.0);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);

        // data input
        String datasetName = cmd.getOptionValue("dataset");
        String wordVocFile = cmd.getOptionValue("word-voc-file");
        String docWordFile = cmd.getOptionValue("word-file");
        String docInfoFile = cmd.getOptionValue("info-file");

        // data output
        String outputFolder = cmd.getOptionValue("output-folder");

        ResponseTextDataset data = new ResponseTextDataset(datasetName);
        data.loadFormattedData(new File(wordVocFile),
                new File(docWordFile),
                new File(docInfoFile),
                null);
        int V = data.getWordVocab().size();

        SLDA sampler = new SLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        sampler.configure(outputFolder, V, K,
                alpha, beta, rho, mu, sigma,
                initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        double[] docResponses = data.getResponses();
        if (cmd.hasOption("z")) { // z-normalization
            ZNormalizer zNorm = new ZNormalizer(docResponses);
            docResponses = zNorm.normalize(docResponses);
        }

        ArrayList<Integer> selectedDocIndices = null;
        if (cmd.hasOption("selected-docs-file")) {
            String selectedDocFile = cmd.getOptionValue("selected-docs-file");
            selectedDocIndices = new ArrayList<>();
            BufferedReader reader = IOUtils.getBufferedReader(selectedDocFile);
            String line;
            while ((line = reader.readLine()) != null) {
                int docIdx = Integer.parseInt(line);
                if (docIdx >= data.getDocIds().length) {
                    throw new RuntimeException("Out of bound. Doc index " + docIdx);
                }
                selectedDocIndices.add(Integer.parseInt(line));
            }
            reader.close();
        }

        sampler.train(data.getWords(), selectedDocIndices, docResponses);
        sampler.initialize();
        sampler.iterate();
        sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOpitions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
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
}
