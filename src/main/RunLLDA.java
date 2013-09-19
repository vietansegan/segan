/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractExperiment;
import core.AbstractSampler.InitialState;
import data.MultiLabelTextData;
import java.io.File;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampler.labeled.LabeledLDASampler;
import util.IOUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class RunLLDA {
    private static CommandLineParser parser;
    private static Options options;
    private static CommandLine cmd;
    
    private static MultiLabelTextData data;
    private static MimnoTopicCoherence topicCoherence;
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
            
            options.addOption(OptionBuilder.withLongOpt("format-folder")
                    .withDescription("Folder containing formatted data")
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
        try{
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String outputFolder = cmd.getOptionValue("output");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            data = new MultiLabelTextData(datasetName, datasetFolder);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            
            numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            topicCoherence = new MimnoTopicCoherence(data.getWords(), data.getWordVocab().size(), numTopWords);
            topicCoherence.prepare();

            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            
            int K = data.getLabelVocab().size();
            int V = data.getWordVocab().size();
            double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
            double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1) * V;
            double eta = CLIUtils.getDoubleArgument(cmd, "eta", 100.0);
            
            boolean paramOpt = cmd.hasOption("paramOpt");
            boolean verbose = cmd.hasOption("v");
            boolean debug = cmd.hasOption("d");
            
            verbose = true;
            debug = true;
            
            Experiment expt = new Experiment(data);
            expt.configure(burnIn, maxIters, sampleLag, repInterval, numTopWords);
            expt.setup();

            expt.runSampler(outputFolder, data.getWords(), data.getLabels(),
                    K, alpha, beta, eta,
                    paramOpt, verbose, debug);
        }
        catch(Exception e){
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.RunLLDA -help", options);
            throw new RuntimeException("Exception while running model");
        }
    }
    
    static class Experiment extends AbstractExperiment<MultiLabelTextData> {
        protected static int reportInterval;
        protected static int numTopWords;
        protected static MimnoTopicCoherence topicCoherence;

        public Experiment(MultiLabelTextData d) {
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
                int[][] labels,
                int K,
                double alpha,
                double beta,
                double eta,
                boolean paramOpt,
                boolean verbose,
                boolean debug) throws Exception {
            int V = data.getWordVocab().size();
            InitialState initState = InitialState.RANDOM;

            LabeledLDASampler sampler = new LabeledLDASampler();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());

            sampler.configure(resultFolder, words, labels,
                    V, K, alpha, beta, eta, initState, paramOpt,
                    burn_in, max_iters, sample_lag, reportInterval);

            String sldaFolder = new File(resultFolder, sampler.getSamplerFolder()).getAbsolutePath();
            IOUtils.createFolder(sldaFolder);
            sampler.sample();
            sampler.outputTopicTopWords(sldaFolder + TopWordFile, numTopWords);
//            sampler.outputTopicCoherence(sldaFolder + TopicCoherenceFile, topicCoherence);
//            sampler.outputDocTopicDistributions(sldaFolder + "doc-topic.txt");
//            sampler.outputTopicWordDistributions(sldaFolder + "topic-word.txt");
//            sampler.outputTopicRegressionParameters(sldaFolder + "topic-reg-params.txt");
        }

        @Override
        public void run() {
        }

        @Override
        public void evaluate() {
        }
    }
}
