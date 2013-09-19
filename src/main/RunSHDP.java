/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractExperiment;
import core.AbstractSampler.InitialState;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import core.crossvalidation.RegressionDocumentInstance;
import data.SingleResponseTextDataset;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampler.supervised.regression.SHDPSampler;
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
public class RunSHDP {

    public static final String TopWordFile = AbstractExperiment.TopWordFile;
    public static final String TopicCoherenceFile = AbstractExperiment.TopicCoherenceFile;
    private static CommandLineParser parser;
    private static Options options;
    private static CommandLine cmd;
    private static SingleResponseTextDataset data;
    private static MimnoTopicCoherence topicCoherence;
    ;
    private static int numTopWords;

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            options.addOption(OptionBuilder.withLongOpt("output")
                    .withDescription("Output folder")
                    .hasArg()
                    .withArgName("Output folder")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("dataset")
                    .withDescription("Dataset name")
                    .hasArg()
                    .withArgName("Dataset")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("folder")
                    .withDescription("Folder that stores the processed data")
                    .hasArg()
                    .withArgName("Folder directory")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("burnIn")
                    .withDescription("Burn-in. Default 250.")
                    .hasArg()
                    .withArgName("Burn-in")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("maxIter")
                    .withDescription("Maximum number of iterations")
                    .hasArg()
                    .withArgName("Maximum number of iterations")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("sampleLag")
                    .withDescription("Sample lag")
                    .hasArg()
                    .withArgName("sample lag")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("numTopwords")
                    .withDescription("Number of top words")
                    .hasArg()
                    .withArgName("Number of top words")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("alpha-global")
                    .withDescription("Hyperparameter of global DP")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("alpha-local")
                    .withDescription("Hyperparameter of local DP")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("beta")
                    .withDescription("Hyperparameter of the symmetric Dirichlet prior for word distributions")
                    .hasArg()
                    .withArgName("beta")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("mu")
                    .withDescription("Prior mean of regression parameters. Default: 0.0")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("sigma")
                    .withDescription("Prior variance of regression parameters. Default: 1.0")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("rho")
                    .withDescription("Variance of the response variable. Default: 1.0")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("report")
                    .withDescription("Report interval")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("cv-folder")
                    .withDescription("Cross validation folder")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("num-folds")
                    .withDescription("Number of folds")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("run-mode")
                    .withDescription("Mode of running during cross validation (train, test, hack)")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption("paramOpt", false, "Whether hyperparameter optimization using slice sampling is performed");
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
                throw new RuntimeException("Missing cv-folder");
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
            String datasetFolder = cmd.getOptionValue("folder"); // processed (format) folder
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            data = new SingleResponseTextDataset(datasetName, datasetFolder);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());

            numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            topicCoherence = new MimnoTopicCoherence(data.getWords(), data.getWordVocab().size(), numTopWords);
            topicCoherence.prepare();

            ArrayList<RegressionDocumentInstance> instanceList = new ArrayList<RegressionDocumentInstance>();
            for (int i = 0; i < data.getDocIds().length; i++) {
                instanceList.add(new RegressionDocumentInstance(data.getDocIds()[i], data.getWords()[i], data.getResponses()[i]));
            }

            String cvName = "";
            String cvFolder = cmd.getOptionValue("cv-folder");
            CrossValidation<String, RegressionDocumentInstance> crossValidation = new CrossValidation<String, RegressionDocumentInstance>(
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

            for (Fold<String, ? extends Instance<String>> fold : crossValidation.getFolds()) {
                String foldFolder = resultFolder + fold.getFoldFolder();

                int V = data.getWordVocab().size();
                int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
                int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
                int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
                int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
                int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
                double alpha_global = CLIUtils.getDoubleArgument(cmd, "alpha-global", 0.1);
                double alpha_local = CLIUtils.getDoubleArgument(cmd, "alpha-local", 0.1);
                double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
                boolean paramOpt = cmd.hasOption("paramOpt");
                boolean verbose = cmd.hasOption("v");
                boolean debug = cmd.hasOption("d");
                InitialState initState = InitialState.PRESET;

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

                SHDPSampler sampler = new SHDPSampler();
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
                        V, alpha_global, alpha_local, beta,
                        mu, sigma, rho,
                        initState, paramOpt,
                        burnIn, maxIters, sampleLag, repInterval);

                String samplerFolder = new File(foldFolder, sampler.getSamplerFolder()).getAbsolutePath();
                IOUtils.createFolder(samplerFolder);

                if (runMode.equals("train")) {
                    sampler.initialize();
                    sampler.iterate();
                    sampler.outputTopicTopWords(samplerFolder + TopWordFile, numTopWords);
                    sampler.outputTopicCoherence(samplerFolder + TopicCoherenceFile, topicCoherence);
                } else if (runMode.equals("test")) {
                    sampler.regressNewDocuments(teRevWords);

                    String teResultFolder = samplerFolder + "te-results/";
                    IOUtils.createFolder(teResultFolder);
                    sampler.computeSingleFinal(teResultFolder, teResponses);
                    sampler.computeSingleAverage(teResultFolder, teResponses);
                    sampler.computeMultipleFinal(teResultFolder, teResponses);
                    sampler.computeMultipleAverage(teResultFolder, teResponses);
                } else if (runMode.equals("train-test")) {
                    // train
                    sampler.initialize();
                    sampler.iterate();
                    sampler.outputTopicTopWords(samplerFolder + TopWordFile, numTopWords);
                    sampler.outputTopicCoherence(samplerFolder + TopicCoherenceFile, topicCoherence);

                    // test
                    sampler.regressNewDocuments(teRevWords);
                    String teResultFolder = samplerFolder + "te-results/";
                    IOUtils.createFolder(teResultFolder);
                    sampler.computeSingleFinal(teResultFolder, teResponses);
                    sampler.computeSingleAverage(teResultFolder, teResponses);
                    sampler.computeMultipleFinal(teResultFolder, teResponses);
                    sampler.computeMultipleAverage(teResultFolder, teResponses);
                } else if (runMode.equals("hack")) {
                } else {
                    throw new RuntimeException("Run mode " + runMode + " not supported");
                }

                break;
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
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
}
