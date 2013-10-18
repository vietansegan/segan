/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractExperiment;
import core.AbstractRunner;
import core.AbstractSampler.InitialState;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import core.crossvalidation.RegressionDocumentInstance;
import data.SingleResponseTextDataset;
import data.TextDataset;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.supervised.regression.SLDA;
import util.IOUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class RunSLDA extends AbstractRunner {
    public static final String trainingPredictionFile = "pred-tr.txt";
    public static final String trainingResultFile = "result-tr.txt";
    public static final String testPredictionFile = "pred-te.txt";
    public static final String testResultFile = "result-te.txt";
    
    private static SingleResponseTextDataset data;
    private static MimnoTopicCoherence topicCoherence;
    private static int numTopWords;
    
    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();
            
            // directories
            addOption("dataset", "Dataset");
            addOption("folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("output", "Output folder");
            
            // sampling configurations
            addOption("burnIn", "Burn-in");
            addOption("maxIter", "Maximum number of iterations");
            addOption("sampleLag", "Sample lag");
            addOption("report", "Report interval.");
            
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
            options.addOption("s", false, "standardize (z-score normalization)");
            options.addOption("help", false, "Help");
            
            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp dist/segan.jar main.RunSLDA -help", options);
                return;
            }
            
            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
                runModels();
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.RunSLDA -help", options);
            System.exit(1);
        }
    }
    
    public static void runCrossValidation() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder"); 
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            data = new SingleResponseTextDataset(datasetName, datasetFolder);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            
            numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            topicCoherence = new MimnoTopicCoherence(data.getWords(), data.getWordVocab().size(), numTopWords);
            topicCoherence.prepare();
            
            ArrayList<RegressionDocumentInstance> instanceList = new ArrayList<RegressionDocumentInstance>();
            for (int i = 0; i < data.getDocIds().length; i++) {
                instanceList.add(new RegressionDocumentInstance(
                        data.getDocIds()[i],
                        data.getWords()[i],
                        data.getResponses()[i]));
            }
            
            String cvName = "";
            String cvFolder = cmd.getOptionValue("cv-folder");
            CrossValidation<String, RegressionDocumentInstance> crossValidation 
                    = new CrossValidation<String, RegressionDocumentInstance>(
                    cvFolder,
                    cvName,
                    instanceList);
            int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));
            
            System.out.println("\nLoading cross validation info from " + cvFolder);
            crossValidation.inputFolds(numFolds);
            
            String resultFolder = cmd.getOptionValue("output");
            if (resultFolder == null) {
                throw new RuntimeException("Result folder has not been set. Use option --output");
            }
            
            String runMode = cmd.getOptionValue("run-mode");
            int foldIndex = -1;
            if (cmd.hasOption("fold")) {
                foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
            }
            
            for (Fold<String, ? extends Instance<String>> fold : crossValidation.getFolds()) {
                if (foldIndex != -1 && fold.getIndex() != foldIndex) {
                    continue;
                }
                
                String foldFolder = new File(resultFolder, fold.getFoldFolder()).getAbsolutePath();
                
                int V = data.getWordVocab().size();
                int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
                int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
                int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
                int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
                int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
                double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1) * K;
                double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1) * V;
                boolean paramOpt = cmd.hasOption("paramOpt");
                verbose = cmd.hasOption("v");
                debug = cmd.hasOption("d");
                InitialState initState = InitialState.RANDOM;
                
                double[] responses = data.getResponses();
                if (cmd.hasOption("s")) {
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
                
                SLDA sampler = new SLDA();
                sampler.setVerbose(verbose);
                sampler.setDebug(debug);
                sampler.setLog(true);
                sampler.setReport(true);
                sampler.setWordVocab(data.getWordVocab());

                // training data
                ArrayList<Integer> trInstIndices = fold.getTrainingInstances();
                int[][] trRevWords = getRevWords(trInstIndices);
                double[] trResponses = getResponses(trInstIndices, responses);

                // test data
                ArrayList<Integer> teInstIndices = fold.getTestingInstances();
                int[][] teRevWords = getRevWords(teInstIndices);
                double[] teResponses = getResponses(teInstIndices, responses);
                
                sampler.configure(foldFolder, trRevWords, trResponses,
                        V, K, alpha, beta,
                        mu, sigma, rho,
                        initState, paramOpt,
                        burnIn, maxIters, sampleLag, repInterval);
                
                File samplerFolder = new File(foldFolder, sampler.getSamplerFolder());
                IOUtils.createFolder(samplerFolder);
                
                if (runMode.equals("train")) {
                    sampler.initialize();
                    sampler.iterate();
                    sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                    sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile), topicCoherence);
                    
                    sampler.outputDocTopicDistributions(new File(samplerFolder, "tr-doc-topic.txt"));
                    sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                    sampler.outputTopicRegressionParameters(new File(samplerFolder, "topic-reg-params.txt"));
                } else if (runMode.equals("test")) {
                    sampler.testSampler(teRevWords);
                    
                    File teResultFolder = new File(samplerFolder, "te-results");
                    IOUtils.createFolder(teResultFolder);
                    sampler.computeSingleFinal(teResultFolder, teResponses);
                    sampler.computeSingleAverage(teResultFolder, teResponses);
                    sampler.computeMultipleFinal(teResultFolder, teResponses);
                    sampler.computeMultipleAverage(teResultFolder, teResponses);
                    
                    sampler.outputDocTopicDistributions(new File(samplerFolder, "te-doc-topic.txt"));
                } else if (runMode.equals("train-test")) {
                    // train
                    sampler.initialize();
                    sampler.iterate();
                    sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                    sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile), topicCoherence);
                    sampler.outputDocTopicDistributions(new File(samplerFolder, "tr-doc-topic.txt"));
                    sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                    sampler.outputTopicRegressionParameters(new File(samplerFolder, "topic-reg-params.txt"));

                    // test
                    sampler.testSampler(teRevWords);
                    
                    File teResultFolder = new File(samplerFolder, "te-results");
                    IOUtils.createFolder(teResultFolder);
                    sampler.computeSingleFinal(teResultFolder, teResponses);
                    sampler.computeSingleAverage(teResultFolder, teResponses);
                    sampler.computeMultipleFinal(teResultFolder, teResponses);
                    sampler.computeMultipleAverage(teResultFolder, teResponses);
                    
                    sampler.outputDocTopicDistributions(new File(samplerFolder, "te-doc-topic.txt"));
                } else if (runMode.equals("hack")) {
                } else {
                    throw new RuntimeException("Run mode " + runMode + " not supported");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running cross validation");
        }
    }
    
    public static void outputPredictions(String outputFilepath,
            ArrayList<Integer> instanceIndices,
            double[] trueValues, double[] predValues,
            CrossValidation<String, RegressionDocumentInstance> crossValidation) throws Exception {
        if (instanceIndices.size() != trueValues.length || instanceIndices.size() != predValues.length) {
            throw new RuntimeException("Lengths mismatched");
        }
        
        System.out.println(">>> Output predictions to " + outputFilepath);
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFilepath);
        writer.write(instanceIndices.size() + "\n");
        for (int i = 0; i < instanceIndices.size(); i++) {
            if (trueValues == null) {
                throw new RuntimeException("trueValues");
            } else if (predValues == null) {
                throw new RuntimeException("predValues");
            } else if (instanceIndices.get(i) == null) {
                throw new RuntimeException("instanceIndices");
            }
            
            writer.write(instanceIndices.get(i)
                    + "\t" + crossValidation.getInstance(instanceIndices.get(i)).getId()
                    + "\t" + trueValues[i]
                    + "\t" + predValues[i]
                    + "\n");
        }
        writer.close();
    }
    
    public static ArrayList<Measurement> outputRegressionResults(String outputFilepath,
            double[] trueValues, double[] predValues) throws Exception {
        // output different measurements
        System.out.println(">>> Output regression results to " + outputFilepath);
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFilepath);
        RegressionEvaluation eval = new RegressionEvaluation(trueValues, predValues);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeRSquared();
        eval.computePredictiveRSquared();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement m : measurements) {
            writer.write(m.getName() + "\t" + m.getValue() + "\n");
        }
        writer.close();
        return measurements;
    }
    
    private static double[] getResponses(ArrayList<Integer> instances, double[] responses) {
        double[] res = new double[instances.size()];
        for (int i = 0; i < res.length; i++) {
            int idx = instances.get(i);
            res[i] = responses[idx];
        }
        return res;
    }
    
    private static int[][] getRevWords(ArrayList<Integer> instances) {
        int[][] revWords = new int[instances.size()][];
        for (int i = 0; i < revWords.length; i++) {
            int idx = instances.get(i);
            revWords[i] = data.getWords()[idx];
        }
        return revWords;
    }
    
    public static void runModels() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String outputFolder = cmd.getOptionValue("output");
            String formatFolder = cmd.getOptionValue("format-folder");
            data = new SingleResponseTextDataset(datasetName, datasetFolder);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            
            numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            topicCoherence = new MimnoTopicCoherence(data.getWords(), data.getWordVocab().size(), numTopWords);
            topicCoherence.prepare();
            
            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
            
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
            int V = data.getWordVocab().size();
            double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1) * K;
            double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1) * V;
            
            double[] responses = data.getResponses();
            if (cmd.hasOption("s")) {
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
            
            boolean paramOpt = cmd.hasOption("paramOpt");
            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");
            
            System.out.println("\nRunning model ...");
            Experiment expt = new Experiment(data);
            expt.configure(burnIn, maxIters, sampleLag, repInterval, numTopWords);
            expt.setup();
            
            expt.runSampler(outputFolder, data.getWords(), responses,
                    K, alpha, beta,
                    mu, sigma, rho,
                    paramOpt, verbose, debug);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.RunSLDA -help", options);
        }
    }
    
    static class Experiment extends AbstractExperiment<TextDataset> {
        
        protected static int reportInterval;
        protected static int numTopWords;
        protected static MimnoTopicCoherence topicCoherence;
        
        public Experiment(TextDataset d) {
            this.data = d;
        }
        
        public void configure(int burnIn, int maxIters, int sampleLag, int repInt, int nTopwords) {
            burn_in = burnIn;
            max_iters = maxIters;
            sample_lag = sampleLag;
            reportInterval = repInt;
            numTopWords = nTopwords;
        }
        
        @Override
        public void setup() {
            topicCoherence = new MimnoTopicCoherence(
                    data.getWords(),
                    data.getWordVocab().size(),
                    numTopWords);
            topicCoherence.prepare();
        }
        
        public void runSampler(
                String resultFolder,
                int[][] words,
                double[] responses,
                int K,
                double alpha,
                double beta,
                double mu,
                double sigma,
                double rho,
                boolean paramOpt,
                boolean verbose,
                boolean debug) throws Exception {
            int V = data.getWordVocab().size();
            InitialState initState = InitialState.RANDOM;
            
            SLDA sampler = new SLDA();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());
            
            sampler.configure(resultFolder, words, responses,
                    V, K, alpha, beta, mu, sigma, rho, initState, paramOpt,
                    burn_in, max_iters, sample_lag, reportInterval);
            
            File sldaFolder = new File(resultFolder, sampler.getSamplerFolder());
            IOUtils.createFolder(sldaFolder);
            sampler.sample();
            sampler.outputTopicTopWords(new File(sldaFolder, TopWordFile), numTopWords);
            sampler.outputTopicCoherence(new File(sldaFolder, TopicCoherenceFile), topicCoherence);
            sampler.outputDocTopicDistributions(new File(sldaFolder, "doc-topic.txt"));
            sampler.outputTopicWordDistributions(new File(sldaFolder, "topic-word.txt"));
            sampler.outputTopicRegressionParameters(new File(sldaFolder, "topic-reg-params.txt"));
        }
        
        @Override
        public void run() {
        }
        
        @Override
        public void evaluate() {
        }
    }
}
