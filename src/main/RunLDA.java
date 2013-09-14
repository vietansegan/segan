/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractExperiment;
import core.AbstractSampler.InitialState;
import data.TextDataset;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampler.LDASampler;
import util.IOUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class RunLDA {

    private static CommandLineParser parser;
    private static Options options;
    private static CommandLine cmd;

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

            options.addOption(OptionBuilder.withLongOpt("K")
                    .withDescription("Number of topics")
                    .hasArg()
                    .withArgName("Number of topics")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("alpha")
                    .withDescription("Hyperparameter of the symmetric Dirichlet prior for topic distributions")
                    .hasArg()
                    .withArgName("alpha")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("beta")
                    .withDescription("Hyperparameter of the symmetric Dirichlet prior for word distributions")
                    .hasArg()
                    .withArgName("beta")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("report")
                    .withDescription("Report interval")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("K")
                    .withDescription("Number of topics")
                    .hasArg()
                    .withArgName("Number of topics")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("alpha")
                    .withDescription("Hyperparameter of the symmetric Dirichlet prior for topic distributions")
                    .hasArg()
                    .withArgName("alpha")
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

            options.addOption("paramOpt", false, "Whether hyperparameter optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("s", false, "standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp dist/segan.jar main.ProcessData -help", options);
                return;
            }

            runModels();
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.ProcessData -help", options);
            System.exit(1);
        }
    }

    public static void runModels() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String outputFolder = cmd.getOptionValue("output");
            TextDataset dataset = new TextDataset(datasetName, datasetFolder);
            dataset.loadFormattedData();

            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
            int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
            double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
            double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
            boolean paramOpt = cmd.hasOption("paramOpt");
            boolean verbose = cmd.hasOption("v");
            boolean debug = cmd.hasOption("d");

            System.out.println("\nRunning model ...");
            Experiment expt = new Experiment(dataset);
            expt.configure(burnIn, maxIters, sampleLag, repInterval, numTopWords);
            expt.setup();
            expt.runSampler(outputFolder, dataset.getWords(), K, alpha, beta, paramOpt, verbose, debug);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.ProcessData -help", options);
        }
    }

    static class Experiment extends AbstractExperiment<TextDataset> {

        protected static int repInterval;
        protected static int numTopWords;
        protected static MimnoTopicCoherence topicCoherence;

        public Experiment(TextDataset d) {
            this.data = d;
        }

        public void configure(int burnIn, int maxIters, int sampleLag, int repInts, int nTopwords) {
            burn_in = burnIn;
            max_iters = maxIters;
            sample_lag = sampleLag;
            numTopWords = nTopwords;
            repInterval = repInts;
        }

        @Override
        public void setup() {
            topicCoherence = new MimnoTopicCoherence(data.getWords(), data.getWordVocab().size(), numTopWords);
            topicCoherence.prepare();
        }

        public void runSampler(
                String resultFolder,
                int[][] words,
                int K,
                double alpha,
                double beta,
                boolean paramOpt,
                boolean verbose,
                boolean debug) throws Exception {
            int V = data.getWordVocab().size();
            InitialState initState = InitialState.RANDOM;

            LDASampler sampler = new LDASampler();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());

            sampler.configure(resultFolder, words,
                    V, K, alpha, beta, initState, paramOpt,
                    burn_in, max_iters, sample_lag, repInterval);

            String ldaFolder = resultFolder + sampler.getSamplerFolder();
            IOUtils.createFolder(ldaFolder);
            sampler.sample();
            sampler.outputTopicTopWords(ldaFolder + TopWordFile, numTopWords);
            sampler.outputTopicCoherence(ldaFolder + TopicCoherenceFile, topicCoherence);
            sampler.outputDocTopicDistributions(ldaFolder + "doc-topic.txt");
            sampler.outputTopicWordDistributions(ldaFolder + "topic-word.txt");
        }

        @Override
        public void run() {
        }

        @Override
        public void evaluate() {
        }
    }
}