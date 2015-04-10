package benchmark;

import core.AbstractExperiment;
import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import core.crossvalidation.Fold;
import data.CorpusProcessor;
import data.ResponseTextDataset;
import java.io.File;
import java.util.ArrayList;
import optimization.LBFGSLinearRegression;
import optimization.OWLQNLinearRegression;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import regression.SVR;
import sampler.HTM;
import sampler.supervised.regression.SLDA;
import sampler.supervised.regression.SNLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import util.CLIUtils;
import util.IOUtils;
import util.PredictionUtils;

/**
 *
 * @author vietan
 */
public class RegressionBenchmark<D extends ResponseTextDataset> extends AbstractExperiment<D> {

    protected static CorpusProcessor corpProc;
    private String modelFolder;
    private int numFolds;
    private double trToDevRatio;
    private String cvFolder;
    private int numTopWords;
    private ResponseTextDataset trainData;
    private ResponseTextDataset devData;
    private ResponseTextDataset testData;

    public void setDataset(D dataset) {
        this.data = dataset;
    }

    @Override
    public void setup() {
        if (verbose) {
            logln("Setting up ...");
        }

        numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
        trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 0.8);
        cvFolder = cmd.getOptionValue("cv-folder");
        modelFolder = CLIUtils.getStringArgument(cmd, "model-folder", "models");
        numTopWords = CLIUtils.getIntegerArgument(cmd, "num-topwords", 20);
    }

    @Override
    public void preprocess() throws Exception {
        if (verbose) {
            logln("Preprocessing: create cross-validated data ...");
        }
        String textInputData = cmd.getOptionValue("text-data");
        String responseFile = cmd.getOptionValue("response-file");

        int numClasses = CLIUtils.getIntegerArgument(cmd, "num-classes", 1);

        corpProc = ResponseTextDataset.createCorpusProcessor();
        data.setCorpusProcessor(corpProc);

        // load text data
        File textPath = new File(textInputData);
        if (textPath.isFile()) {
            data.loadTextDataFromFile(textInputData);
        } else if (textPath.isDirectory()) {
            data.loadTextDataFromFolder(textInputData);
        } else {
            throw new RuntimeException(textInputData + " is neither a file nor a folder");
        }
        data.loadResponses(responseFile); // load response data
        data.createCrossValidation(cvFolder, numFolds, trToDevRatio, numClasses);
    }

    @Override
    public void run() throws Exception {
        if (verbose) {
            logln("Running ...");
        }
        String model = CLIUtils.getStringArgument(cmd, "model", "slda");
        String init = CLIUtils.getStringArgument(cmd, "init", "random");
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

        burn_in = CLIUtils.getIntegerArgument(cmd, "burnIn", 5);
        max_iters = CLIUtils.getIntegerArgument(cmd, "maxIter", 10);
        sample_lag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        report_interval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        paramOpt = cmd.hasOption("paramOpt");

        ArrayList<Integer> runningFolds = new ArrayList<Integer>();
        if (cmd.hasOption("fold")) {
            String foldList = cmd.getOptionValue("fold");
            for (String f : foldList.split(",")) {
                runningFolds.add(Integer.parseInt(f));
            }
        }

        for (int ff = 0; ff < numFolds; ff++) {
            if (!runningFolds.isEmpty() && !runningFolds.contains(ff)) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + ff);
            }

            Fold fold = new Fold(ff, cvFolder);

            ResponseTextDataset[] foldData = ResponseTextDataset.loadCrossValidationFold(fold);
            trainData = foldData[Fold.TRAIN];
            devData = foldData[Fold.DEV];
            testData = foldData[Fold.TEST];

            if (cmd.hasOption("logit")) {
                ResponseTextDataset.logit(trainData, devData, testData);
            }

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

            switch (model) {
                case "svr":
                    runSVR(fold);
                    break;
                case "mlr-owlqn":
                    runMLR_OWLQN(fold);
                    break;
                case "mlr-lbfgs":
                    runMLR_LBFGS(fold);
                    break;
                case "slda":
                    runSLDA(fold);
                    break;
                case "snlda":
                    runSNLDA(fold);
                    break;
                case "htm":
                    runHTM(fold);
                    break;
                default:
                    throw new RuntimeException("Model " + model + " is not supported");
            }
            evaluate();
        }
    }

    private void runSLDA(Fold fold) throws Exception {
        String foldFolder = fold.getFoldFolderPath();

        SLDA sampler = new SLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(trainData.getWordVocab());

        paramOpt = cmd.hasOption("paramOpt");
        // model parameters
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 1.0);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);
        boolean hasBias = cmd.hasOption("bias");

        sampler.configure(new File(foldFolder, modelFolder).getAbsolutePath(),
                trainData.getWordVocab().size(), K,
                alpha, beta, rho, mu, sigma,
                initState, paramOpt, hasBias,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.train(trainData.getWords(), null, trainData.getResponses());
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);

            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            double[] trPredictions = sampler.getPredictedValues();
            evaluatePhase(trainData, TRAIN_PREFIX, Fold.TrainingExt, trResultFolder, trPredictions);
        }

        if (cmd.hasOption("test")) {
            double[] tePredictions;
            if (cmd.hasOption("parallel")) { // predict using all models
                File iterPredFolder = new File(sampler.getSamplerFolderPath(),
                        AbstractSampler.IterPredictionFolder);
                IOUtils.createFolder(iterPredFolder);
                tePredictions = SLDA.parallelTest(testData.getWords(), null, iterPredFolder, sampler);
            } else { // predict using the final model
                tePredictions = sampler.test(testData.getWords(), null, sampler.getFinalStateFile(), null);
            }

            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluatePhase(testData, TEST_PREFIX, Fold.TestExt, teResultFolder, tePredictions);
        }
    }

    private void runSNLDA(Fold fold) throws Exception {
        String foldFolder = fold.getFoldFolderPath();

        int[] Ks = CLIUtils.getIntArrayArgument(cmd, "Ks", new int[]{15, 4}, ",");
        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas", new double[]{2.0, 1.0}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", new double[]{0.5, 0.25, 0.1}, ",");
        double[] pis = CLIUtils.getDoubleArrayArgument(cmd, "pis", new double[]{0.2, 0.2}, ",");
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", new double[]{100, 10}, ",");
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", new double[]{2.5, 2.5, 2.5}, ",");

        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        PathAssumption pathAssumption = AbstractSampler.getPathAssumption(path);

        boolean isRooted = cmd.hasOption("root");

        SNLDA sampler = new SNLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(trainData.getWordVocab());

        sampler.configureContinuous(new File(foldFolder, modelFolder).getAbsolutePath(),
                trainData.getWordVocab().size(), Ks,
                alphas, betas, pis, gammas, rho, mu, sigmas,
                initState, pathAssumption, paramOpt, isRooted,
                burn_in, max_iters, sample_lag, report_interval);

        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (isTraining()) { // train
            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            sampler.train(trainData.getWords(), null, trainData.getResponses());
            sampler.initialize();
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);

            double[] trPredictions = sampler.getPredictedValues();
            evaluatePhase(trainData, TRAIN_PREFIX, Fold.TrainingExt,
                    trResultFolder, trPredictions);
        }

        if (isTesting()) { // test
            double[] tePredictions;
            if (cmd.hasOption("parallel")) { // predict using all models
                File iterPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
                IOUtils.createFolder(iterPredFolder);
                File testStateFolder = new File(samplerFolder, TEST_PREFIX + AbstractSampler.ReportFolder);
                IOUtils.createFolder(testStateFolder);

                tePredictions = SNLDA.parallelTest(testData.getWords(), null,
                        iterPredFolder, testStateFolder, sampler);
            } else { // predict using the final model
                File testPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
                IOUtils.createFolder(testPredFolder);
                File stateFile = sampler.getFinalStateFile();
                File outputPredFile = new File(testPredFolder, "iter-" + max_iters + ".txt");
                File outputStateFile = new File(testPredFolder, "iter-" + max_iters + ".zip");

                sampler.test(testData.getWords(), null);
                sampler.setContinuousResponses(testData.getResponses());
                tePredictions = sampler.sampleTest(stateFile, outputStateFile, outputPredFile);
                sampler.outputTopicTopWords(new File(samplerFolder, TEST_PREFIX + TopWordFile), numTopWords);
            }

            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluatePhase(testData, TEST_PREFIX, Fold.TestExt, teResultFolder, tePredictions);
        }
    }

    private void runHTM(Fold fold) throws Exception {
        String foldFolder = fold.getFoldFolderPath();
        String basename = CLIUtils.getStringArgument(cmd, "basename", "HTM");

        HTM sampler = new HTM();
        sampler.setBasename(basename);
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(trainData.getWordVocab());

        paramOpt = cmd.hasOption("paramOpt");
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        PathAssumption pathAssumption = AbstractSampler.getPathAssumption(path);

        int L = CLIUtils.getIntegerArgument(cmd, "L", 2);
        int[] Ks = CLIUtils.getIntArrayArgument(cmd, "Ks", new int[]{20}, ",");
        double[] globalAlphas = CLIUtils.getDoubleArrayArgument(cmd, "global-alphas",
                new double[]{2.0}, ",");
        double[] localAlphas = CLIUtils.getDoubleArrayArgument(cmd, "local-alphas",
                new double[]{2.0}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{1.0, 0.1}, ",");
        double[] pis = CLIUtils.getDoubleArrayArgument(cmd, "pis",
                new double[]{0.0}, ",");
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas",
                new double[]{0.0}, ",");
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{2.5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.0);
        boolean isRooted = cmd.hasOption("root");

        sampler.configureContinuous(new File(foldFolder, modelFolder).getAbsolutePath(),
                trainData.getWordVocab().size(), L, Ks, null, null,
                globalAlphas, localAlphas, betas,
                pis, gammas, rho, mu, sigmas, sigma,
                initState, pathAssumption, isRooted, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);

        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (isTraining()) { // train
            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            sampler.train(trainData.getWords(), null, trainData.getResponses());
            sampler.initialize();
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            if (sigma > 0) { // lexical regression
                sampler.outputRankedLexicalItems(new File(samplerFolder, "lexical-weights.txt"));
                sampler.debugRegression(new File(samplerFolder, "train-regression.debug"));
            }

            double[] trPredictions = sampler.getPredictedValues();
            evaluatePhase(trainData, TRAIN_PREFIX, Fold.TrainingExt,
                    trResultFolder, trPredictions);
        }

        if (isTesting()) { // test
            double[] tePredictions;
            if (cmd.hasOption("parallel")) { // predict using all models
                File iterPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
                IOUtils.createFolder(iterPredFolder);
                File testStateFolder = new File(samplerFolder, TEST_PREFIX + AbstractSampler.ReportFolder);
                IOUtils.createFolder(testStateFolder);

                tePredictions = HTM.parallelTest(testData.getWords(), null,
                        iterPredFolder, testStateFolder, sampler);
            } else { // predict using the final model
                sampler.test(testData.getWords(), null);
                sampler.setContinuousResponses(testData.getResponses());
                tePredictions = sampler.sampleTest(sampler.getFinalStateFile(), null, null);
                sampler.outputTopicTopWords(new File(samplerFolder, TEST_PREFIX + TopWordFile), numTopWords);

                if (sigma > 0) { // lexical regression
                    sampler.debugRegression(new File(samplerFolder, "test-regression.debug"));
                }
            }

            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            evaluatePhase(testData, TEST_PREFIX, Fold.TestExt, teResultFolder, tePredictions);
        }

        if (cmd.hasOption("debug")) {
            File trResultFolder = new File(samplerFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);
            sampler.train(trainData.getWords(), null, trainData.getResponses());
            sampler.inputFinalState();
            if (sigma > 0) { // lexical regression
                sampler.debugRegression(new File(samplerFolder, "train-regression.debug"));
            }
        }
    }

    private void runMLR_OWLQN(Fold fold) throws Exception {
        String foldFolder = fold.getFoldFolderPath();
        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 1.0);
        max_iters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);
        int V = trainData.getWordVocab().size();

        OWLQNLinearRegression mlr = new OWLQNLinearRegression("MLR-OWLQN", l1, l2, max_iters);
        File mlrFolder = new File(new File(foldFolder, modelFolder), mlr.getName());
        IOUtils.createFolder(mlrFolder);

        if (isTraining()) { // train
            File trResultFolder = new File(mlrFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);

            mlr.train(trainData.getWords(), null, trainData.getResponses(), V);

            double[] trPredictions = mlr.test(trainData.getWords(), null, V);
            evaluatePhase(trainData, TRAIN_PREFIX, Fold.TrainingExt,
                    trResultFolder, trPredictions);
        }

        if (isTesting()) { // test
            File teResultFolder = new File(mlrFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            double[] tePredictions = mlr.test(testData.getWords(), null, V);
            evaluatePhase(testData, TEST_PREFIX, Fold.TestExt,
                    teResultFolder, tePredictions);
        }
    }

    private void runMLR_LBFGS(Fold fold) throws Exception {
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 2.5);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        int V = trainData.getWordVocab().size();
        LBFGSLinearRegression mlr = new LBFGSLinearRegression("MLR-LBFGS", mu, sigma, rho);
        File mlrFolder = new File(new File(fold.getFoldFolderPath(), modelFolder), mlr.getName());

        if (isTraining()) {
            File trResultFolder = new File(mlrFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);

            mlr.train(trainData.getWords(), null, trainData.getResponses(), V);

            double[] trPredictions = mlr.test(trainData.getWords(), null, V);
            evaluatePhase(trainData, TRAIN_PREFIX, Fold.TrainingExt,
                    trResultFolder, trPredictions);
        }

        if (isTesting()) { // test
            File teResultFolder = new File(mlrFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            double[] tePredictions = mlr.test(testData.getWords(), null, V);
            evaluatePhase(testData, TEST_PREFIX, Fold.TestExt,
                    teResultFolder, tePredictions);
        }

        if (isDeveloping()) {
            File deResultFolder = new File(new File(
                    new File(fold.getFoldFolderPath(), modelFolder), mlr.getBasename()),
                    DEV_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(deResultFolder);

            mlr.tune(new File(deResultFolder, "tune-rho-sigma.txt"),
                    trainData.getWords(), null, trainData.getResponses(),
                    testData.getWords(), null, testData.getResponses(), V);
        }
    }

    private void runSVR(Fold fold) throws Exception {
        String foldFolder = fold.getFoldFolderPath();
        SVR svr = new SVR(new File(foldFolder, modelFolder).getAbsolutePath());
        if (cmd.hasOption("c")) {
            double c = Double.parseDouble(cmd.getOptionValue("c"));
            svr = new SVR(new File(foldFolder, modelFolder).getAbsolutePath(), c);
        }
        File svrFolder = new File(svr.getRegressorFolder());
        IOUtils.createFolder(svrFolder);

        int V = trainData.getWordVocab().size();
        File modelFile = new File(svrFolder, SVR.MODEL_FILE);
        if (isTraining()) {
            File trResultFolder = new File(svrFolder, TRAIN_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);

            File trainFile = new File(svrFolder, SVR.DATA_FILE + Fold.TrainingExt);
            svr.train(trainData.getWords(),
                    trainData.getResponses(), V, trainFile, modelFile);

            // test on training data
            File trResultFile = new File(trResultFolder, "svm-" + SVR.PREDICTION_FILE + Fold.TrainingExt);
            svr.test(trainFile, modelFile, trResultFile);

            double[] trPredictions = svr.getSVM().getPredictedValues(trResultFile);
            evaluatePhase(trainData, TRAIN_PREFIX, Fold.TrainingExt,
                    trResultFolder, trPredictions);
        }

        if (isTesting()) {
            File teResultFolder = new File(svrFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            File testFile = new File(svrFolder, SVR.DATA_FILE + Fold.TestExt);
            File teResultFile = new File(teResultFolder, "svm-" + SVR.PREDICTION_FILE + Fold.TestExt);
            svr.test(testData.getWords(),
                    testData.getResponses(), V, testFile, modelFile, teResultFile);
            File tePredFile = new File(teResultFolder, SVR.PREDICTION_FILE + Fold.TestExt);
            double[] tePredictions = svr.getSVM().getPredictedValues(teResultFile);
            svr.outputPredictions(tePredFile, testData.getDocIds(),
                    testData.getResponses(), tePredictions);
            evaluatePhase(testData, TEST_PREFIX, Fold.TestExt,
                    teResultFolder, tePredictions);
        }

        if (isDeveloping()) {
            File deResultFolder = new File(svrFolder, DEV_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(deResultFolder);

            File destFile = new File(svrFolder, SVR.DATA_FILE + Fold.DevelopExt);
            File deResultFile = new File(deResultFolder, "svm-" + SVR.PREDICTION_FILE + Fold.DevelopExt);
            svr.test(devData.getWords(),
                    devData.getResponses(), V, destFile, modelFile, deResultFile);
            File dePredFile = new File(deResultFolder, SVR.PREDICTION_FILE + Fold.DevelopExt);
            double[] dePredictions = svr.getSVM().getPredictedValues(deResultFile);
            svr.outputPredictions(dePredFile, devData.getDocIds(),
                    devData.getResponses(), dePredictions);
            evaluatePhase(devData, DEV_PREFIX, Fold.DevelopExt,
                    deResultFolder, dePredictions);
        }
    }

    @Override
    public void evaluate() throws Exception {
        if (verbose) {
            logln("Evaluating ...");
        }
        evaluate(cvFolder, modelFolder, numFolds, TEST_PREFIX, RESULT_FILE + Fold.TestExt);
    }

    public static void evaluatePhase(ResponseTextDataset phaseData,
            String phasePrefix,
            String phaseExt,
            File resultFolder,
            double[] predictions) {

        PredictionUtils.outputRegressionPredictions(
                new File(resultFolder, PREDICTION_FILE + phaseExt),
                phaseData.getDocIds(),
                phaseData.getResponses(),
                predictions);
        PredictionUtils.outputRegressionResults(
                new File(resultFolder, RESULT_FILE + phaseExt),
                phaseData.getResponses(),
                predictions);
        PredictionUtils.outputRankingPerformance(new File(resultFolder,
                phasePrefix + RANKING_FOLDER),
                phaseData.getDocIds(),
                phaseData.getResponses(),
                predictions);
    }

    public static void addExperimentOptions() {
        // directories
        addOption("dataset", "Folder storing processed data");
        addOption("text-data", "Directory of the text data");
        addOption("response-file", "Directory of the response file");
        addOption("format-folder", "Folder that stores formatted data");
        addOption("format-file", "Formatted file name");
        addOption("run-mode", "Run mode");
        addOption("model-folder", "Model");
        addOption("model", "Model");

        // svr
        addOption("c", "Trade-off between training error and margin");

        // mlr
        addOption("l1", "L1");
        addOption("l2", "L2");

        // slda
        addOption("K", "Number of topics");
        addOption("opt-type", "Optimization type (lbfgs or gurobi)");

        // lex-slda
        addOption("tau-mu", "Mean of lexical regression parameters");
        addOption("tau-sigma", "Variance of lexical regression parameters");

        addOption("basename", "Basename");
        addOption("L", "Number of levels");
        addOption("Ks", "Number of topics");
        addOption("local-alphas", "Local alphas");
        addOption("global-alphas", "Global alphas");
        addOption("alphas", "Alphas");
        addOption("betas", "Betas");
        addOption("pis", "Pis");
        addOption("gammas", "Gammas");
        addOption("rho", "Rho");
        addOption("mu", "Mu");
        addOption("sigmas", "Sigmas");
        addOption("sigma", "Sigma");
        addOption("path", "Path assumption");
        options.addOption("root", false, "Does root generate words?");

        // mode parameters
        addGreekParametersOptions();

        // processing options
        addCorpusProcessorOptions();

        // cross validation
        addCrossValidationOptions();

        // sampling
        addSamplingOptions();

        options.addOption("logit", false, "logit transformation");
        options.addOption("z", false, "z-normalize");
        options.addOption("parallel", false, "parallel");
        options.addOption("bias", false, "Bias");

        options.addOption("summarize", false, "Summarize results");
        options.addOption("avgperplexity", false, "Averaging perplexity");
        options.addOption("paramOpt", false, "Optimizing parameters");
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("help", false, "Help");
        options.addOption("debug", false, "debug");
    }

    public static void main(String[] args) {
        try {
            parser = new BasicParser(); // create the command line parser
            options = new Options(); // create the Options
            addExperimentOptions();
            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(RegressionBenchmark.class.getName()), options);
                return;
            }
            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");
            runExperiment();
            System.out.println("End time: " + getCompletedTime());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void runExperiment() throws Exception {
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon");
        ResponseTextDataset data = new ResponseTextDataset(datasetName);
        RegressionBenchmark expt = new RegressionBenchmark();
        expt.setDataset(data);
        expt.setup();
        String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "preprocess");
        switch (runMode) {
            case "preprocess":
                expt.preprocess();
                break;
            case "run":
                expt.run();
                break;
            case "evaluate":
                expt.evaluate();
                break;
            case "summarize":
                break;
            default:
                throw new RuntimeException("Run mode " + runMode + " is not supported");
        }
    }
}
