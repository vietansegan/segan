package main;

import util.CLIUtils;
import core.AbstractExperiment;
import core.AbstractRunner;
import core.AbstractSampler.InitialState;
import data.LabelTextData;
import java.io.File;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampler.backup.HierarchicalLabeledLDABackup;
import util.IOUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class RunHLLDA extends AbstractRunner {

    private static LabelTextData data;
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

            options.addOption(OptionBuilder.withLongOpt("run-mode")
                    .withDescription("Report interval")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("min-label-freq")
                    .withDescription("Minimum label frequency")
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
                CLIUtils.printHelp("java -cp dist/segan.jar main.RunHLLDA -help", options);
                return;
            }

            verbose = true;
            debug = true;

            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            if (runMode.equals("run")) {
                runModels();
            } else if (runMode.equals("hack")) {
                hack();
            } else {
                throw new RuntimeException("Run mode " + runMode + " is not supported.");
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.RunHLLDA -help", options);
            System.exit(1);
        }
    }

    private static void runModels() throws Exception {
        if (verbose) {
            System.out.println("\nRunning model");
        }
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "112");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "folder", "L:/Dropbox/github/data");
        String outputFolder = CLIUtils.getStringArgument(cmd, "output", "L:/Dropbox/github/data/112/hllda");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format-label");
        int minLabelFreq = CLIUtils.getIntegerArgument(cmd, "min-label-freq", 250);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 0.1);
        double mean = CLIUtils.getDoubleArgument(cmd, "mean", 0.3);
        double scale = CLIUtils.getDoubleArgument(cmd, "scale", 50);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        boolean paramOpt = cmd.hasOption("paramOpt");

        if (verbose) {
            System.out.println("--- Loading data from " + datasetFolder);
        }
        data = new LabelTextData(datasetName, datasetFolder);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
        data.filterLabels(minLabelFreq);
        data.filterDocumentWithoutLabels();
        if (verbose) {
            System.out.println("--- Data loaded.");
            System.out.println("--- --- min-label-freq: " + minLabelFreq);
            System.out.println("--- --- label vocab size: " + data.getLabelVocab().size());
            System.out.println("--- --- # documents: " + data.getWords().length
                    + ". " + data.getLabels().length);
        }

        if (verbose) {
            System.out.println("--- Sampling hierarchical labeled LDA ...");
        }
        Experiment expt = new Experiment(data);
        expt.configure(burnIn, maxIters, sampleLag, repInterval, numTopWords);
        expt.setup();
        expt.runSampler(outputFolder,
                alpha, beta, gamma, mean, scale,
                paramOpt, verbose, debug);
    }

    private static void hack() {
    }

    static class Experiment extends AbstractExperiment<LabelTextData> {

        protected static int reportInterval;
        protected static int numTopWords;
        protected static MimnoTopicCoherence topicCoherence;

        public Experiment(LabelTextData d) {
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
                double alpha,
                double beta,
                double gamma,
                double mean,
                double scale,
                boolean paramOpt,
                boolean verbose,
                boolean debug) throws Exception {
            int V = data.getWordVocab().size();

            HierarchicalLabeledLDABackup sampler = new HierarchicalLabeledLDABackup();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());
            sampler.setLabelVocab(data.getLabelVocab());

            sampler.configure(resultFolder, data.getWords(), data.getLabels(),
                    V, data.getLabelVocab().size(),
                    alpha, beta, gamma, mean, scale,
                    InitialState.RANDOM, HierarchicalLabeledLDABackup.PathAssumption.MINIMAL,
                    paramOpt,
                    burn_in, max_iters, sample_lag, reportInterval);

            File sldaFolder = new File(resultFolder, sampler.getSamplerFolder());
            IOUtils.createFolder(sldaFolder);
            sampler.initialize();
//            sampler.iterate();
//            sampler.outputTopicTopWords(
//                    new File(sldaFolder, TopWordFile).getAbsolutePath(), 
//                    numTopWords);
        }

        @Override
        public void run() {
        }

        @Override
        public void evaluate() {
        }
    }
}
