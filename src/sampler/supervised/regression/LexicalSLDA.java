package sampler.supervised.regression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import optimization.GurobiMLRL2Norm;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatisticsUtils;

/**
 *
 * @author vietan
 */
public class LexicalSLDA extends SLDA {

    public static final int TAU_MEAN = 5;   // mean of lexical regression params
    public static final int TAU_SIGMA = 6;  // variance of lexical regression params
    protected double[] lexicalParams;
    protected double[][] lexDesignMatrix;

    public void configure(LexicalSLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(RHO),
                sampler.hyperparams.get(MU),
                sampler.hyperparams.get(SIGMA),
                sampler.hyperparams.get(TAU_MEAN),
                sampler.hyperparams.get(TAU_SIGMA),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho, // standard deviation of Gaussian for document observations
            double mu, // mean of Gaussian for regression parameters
            double sigma, // stadard deviation of Gaussian for regression parameters
            double tau_mean,
            double tau_sigma,
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
        this.hyperparams.add(rho);
        this.hyperparams.add(mu);
        this.hyperparams.add(sigma);
        this.hyperparams.add(tau_mean);
        this.hyperparams.add(tau_sigma);

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

        if (!debug) {
            System.err.close();
        }

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(hyperparams.get(RHO)));
            logln("--- topic mu:\t" + MiscUtils.formatDouble(hyperparams.get(MU)));
            logln("--- topic sigma:\t" + MiscUtils.formatDouble(hyperparams.get(SIGMA)));
            logln("--- lexical mean:\t" + MiscUtils.formatDouble(hyperparams.get(TAU_MEAN)));
            logln("--- lexical sigma:\t" + MiscUtils.formatDouble(hyperparams.get(TAU_SIGMA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    @Override
    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_lex-sLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)))
                .append("_tm-").append(formatter.format(hyperparams.get(TAU_MEAN)))
                .append("_ts-").append(formatter.format(hyperparams.get(TAU_SIGMA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    @Override
    protected void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        topicParams = new double[K];
        for (int k = 0; k < K; k++) {
            topicParams[k] = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
        }

        lexicalParams = new double[V];
        for (int v = 0; v < V; v++) {
            lexicalParams[v] = SamplerUtils.getGaussian(hyperparams.get(TAU_MEAN),
                    hyperparams.get(TAU_SIGMA));
        }
    }

    @Override
    protected void initializeDataStructure() {
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        docRegressMeans = new double[D];

        lexDesignMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            double denom = 1.0 / words[d].length;
            for (int n = 0; n < words[d].length; n++) {
                lexDesignMatrix[d][words[d][n]] += denom;
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

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                String str = "Iter " + iter + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter <= BURN_IN) {
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
            int step = (int) Math.log(iter + 1) + 1;
            if (iter % step == 0) {
                updateTopicRegressionParameters();
            }

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
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
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

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                        HyperparameterFile));
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception iter = " + iter);
        }
    }

    @Override
    protected void updateTopicRegressionParameters() {
        double[][] designMatrix = new double[D][V + K];
        for (int d = 0; d < D; d++) {
            System.arraycopy(lexDesignMatrix[d], 0, designMatrix[d], 0, V);
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            System.arraycopy(empDist, 0, designMatrix[d], V, K);
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setRho(hyperparams.get(RHO));
        double[] means = new double[V + K];
        double[] sigmas = new double[V + K];
        for (int v = 0; v < V; v++) {
            means[v] = hyperparams.get(MU);
            sigmas[v] = hyperparams.get(SIGMA);
        }
        for (int k = 0; k < K; k++) {
            means[V + k] = hyperparams.get(TAU_MEAN);
            sigmas[V + k] = hyperparams.get(TAU_SIGMA);
        }
        mlr.setMeans(means);
        mlr.setSigmas(sigmas);
        double[] params = mlr.solve();
        System.arraycopy(params, 0, lexicalParams, 0, V);
        for (int k = 0; k < K; k++) {
            topicParams[k] = params[V + k];
        }

        // update current predictions
        updatePredictionValues();
    }

    @Override
    protected void updatePredictionValues() {
        this.docRegressMeans = new double[D];
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            this.docRegressMeans[d] = StatisticsUtils.dotProduct(topicParams, empDist)
                    + StatisticsUtils.dotProduct(lexicalParams, lexDesignMatrix[d]);
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

        double lexRegParamLlh = 0.0;
        for (int v = 0; v < V; v++) {
            lexRegParamLlh += StatisticsUtils.logNormalProbability(lexicalParams[v],
                    hyperparams.get(TAU_MEAN),
                    Math.sqrt(hyperparams.get(TAU_SIGMA)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". response: " + MiscUtils.formatDouble(responseLlh)
                    + ". regParam: " + MiscUtils.formatDouble(regParamLlh)
                    + ". lexRegParam: " + MiscUtils.formatDouble(lexRegParamLlh));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh
                + lexRegParamLlh;
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

        double lexRegParamLlh = 0.0;
        for (int v = 0; v < V; v++) {
            lexRegParamLlh += StatisticsUtils.logNormalProbability(lexicalParams[v],
                    hyperparams.get(TAU_MEAN),
                    Math.sqrt(hyperparams.get(TAU_SIGMA)));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh
                + lexRegParamLlh;
        return llh;
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
            for (int v = 0; v < V; v++) {
                modelStr.append(v).append("\n");
                modelStr.append(lexicalParams[v]).append("\n");
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
            throw new RuntimeException("Exception while outputing to " + filepath);
        }
    }

    @Override
    protected void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            // initialize
            topicWords = new DirMult[K];
            topicParams = new double[K];
            lexicalParams = new double[V];

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
            for (int v = 0; v < V; v++) {
                int lexIdx = Integer.parseInt(reader.readLine());
                if (lexIdx != v) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                lexicalParams[v] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    public void outputLexicalParameters(File filepath) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing lexical weights to " + filepath);
        }

        ArrayList<RankingItem<Integer>> sortedWeights = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < V; v++) {
            sortedWeights.add(new RankingItem<Integer>(v, lexicalParams[v]));
        }
        Collections.sort(sortedWeights);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            for (int v = 0; v < V; v++) {
                RankingItem<Integer> rankItem = sortedWeights.get(v);
                int lexIdx = rankItem.getObject();
                writer.write(lexIdx
                        + "\t" + this.wordVocab.get(lexIdx)
                        + "\t" + this.lexicalParams[lexIdx]
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing lexical weights to "
                    + filepath);
        }
    }

    public static void parallelTest(int[][] newWords, int[] newAuthors, int numAuthors,
            File iterPredFolder, LexicalSLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                LexicalSLDATestRunner runner = new LexicalSLDATestRunner(
                        sampler, newWords,
                        stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            runThreads(threads);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
    }
}

class LexicalSLDATestRunner implements Runnable {

    LexicalSLDA sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public LexicalSLDATestRunner(LexicalSLDA sampler,
            int[][] newWords,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        SLDA testSampler = new SLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.sampleNewDocuments(stateFile, newWords, outputFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}