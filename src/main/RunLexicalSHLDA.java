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
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.supervised.regression.SHLDA;
import util.IOUtils;
import util.StatisticsUtils;
import util.evaluation.MimnoTopicCoherence;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class RunLexicalSHLDA extends AbstractRunner {

    private static SingleResponseTextDataset data;
    private static MimnoTopicCoherence topicCoherence;
    private static int numTopWords;

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            addOption("output", "Output folder");
            addOption("dataset", "Dataset");
            addOption("folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("burnIn", "Burn-in");
            addOption("maxIter", "Maximum number of iterations");
            addOption("sampleLag", "Sample lag");
            addOption("report", "Report interval.");
            addOption("gem-mean", "GEM mean. [0.5]");
            addOption("gem-scale", "GEM scale. [50]");
            addOption("betas", "Dirichlet hyperparameter for topic distributions."
                    + " [1, 0.5, 0.25] for a 3-level tree.");
            addOption("gammas", "DP hyperparameters. [1.0, 1.0] for a 3-level tree");
            addOption("mus", "Prior means for topic regression parameters."
                    + " [0.0, 0.0, 0.0] for a 3-level tree and standardized"
                    + " response variable.");
            addOption("sigmas", "Prior variances for topic regression parameters."
                    + " [0.0001, 0.5, 1.0] for a 3-level tree and stadardized"
                    + " response variable.");
            addOption("rho", "Prior variance for response variable. [1.0]");
            addOption("tau-mean", "Prior mean of lexical regression parameters. [0.0]");
            addOption("tau-scale", "Prior scale of lexical regression parameters. [1.0]");
            addOption("num-lex-items", "Number of non-zero lexical regression parameters."
                    + " Defaule: vocabulary size.");
            
            addOption("cv-folder", "Cross validation folder");
            addOption("num-folds", "Number of folds");
            addOption("run-mode", "Running mode");
            addOption("fold", "The cross-validation fold to run");

            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("s", false, "whether standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' "
                        + "main.RunLexicalSHLDA -help", options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");
            
            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
                runModels();
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' "
                    + "main.RunLexicalSHLDA -help", options);
            throw new RuntimeException("Exception while running lexical SHLDA");
        }
    }

    public static void runCrossValidation() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            String resultFolder = cmd.getOptionValue("output");
            String runMode = cmd.getOptionValue("run-mode");
            String cvFolder = cmd.getOptionValue("cv-folder");
            int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));

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
            CrossValidation<String, RegressionDocumentInstance> crossValidation =
                    new CrossValidation<String, RegressionDocumentInstance>(
                    cvFolder,
                    cvName,
                    instanceList);

            System.out.println("\nLoading cross validation info from " + cvFolder);
            crossValidation.inputFolds(numFolds);
            System.out.println("--- Loaded " + crossValidation.getNumFolds() + " folds");

            int foldIndex = -1;
            if (cmd.hasOption("fold")) {
                foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
            }

            for (Fold<String, ? extends Instance<String>> fold : crossValidation.getFolds()) {
                if (foldIndex != -1 && fold.getIndex() != foldIndex) {
                    continue;
                }
                if(verbose)
                    System.out.println("--- Running fold " + fold.getIndex());
                
                String foldFolder = new File(resultFolder, fold.getFoldFolder()).getAbsolutePath();

                int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
                int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
                int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
                numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
                int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
                int V = data.getWordVocab().size();
                int L = CLIUtils.getIntegerArgument(cmd, "tree-height", 3);
                double gem_mean = CLIUtils.getDoubleArgument(cmd, "gem-mean", 0.3);
                double gem_scale = CLIUtils.getDoubleArgument(cmd, "gem-scale", 50);

                double[] defaultBetas = new double[L];
                defaultBetas[0] = 1;
                for (int i = 1; i < L; i++) {
                    defaultBetas[i] = 1.0 / (i + 1);
                }
                double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", defaultBetas, ",");
                for (int i = 0; i < betas.length; i++) {
                    betas[i] = betas[i] * V;
                }

                double[] defaultGammas = new double[L - 1];
                for (int i = 0; i < defaultGammas.length; i++) {
                    defaultGammas[i] = 1.0;
                }

                double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", defaultGammas, ",");

                double[] responses = data.getResponses();
                if (cmd.hasOption("s")) {
                    ZNormalizer zNorm = new ZNormalizer(responses);
                    for (int i = 0; i < responses.length; i++) {
                        responses[i] = zNorm.normalize(responses[i]);
                    }
                }

                double meanResponse = StatisticsUtils.mean(responses);
                double[] defaultMus = new double[L];
                for (int i = 0; i < L; i++) {
                    defaultMus[i] = meanResponse;
                }
                double[] mus = CLIUtils.getDoubleArrayArgument(cmd, "mus", defaultMus, ",");

                double[] defaultSigmas = new double[L];
                defaultSigmas[0] = 0.0001; // root node
                for (int l = 1; l < L; l++) {
                    defaultSigmas[l] = 0.5 * l;
                }
                double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", defaultSigmas, ",");

                double tau_mean = CLIUtils.getDoubleArgument(cmd, "tau-mean", 0.0);
                double tau_scale = CLIUtils.getDoubleArgument(cmd, "tau-scale", 1.0);
                double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 1.0);
                double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
                int numLexicalItems = CLIUtils.getIntegerArgument(cmd, "num-lex-items", V);

                boolean paramOpt = cmd.hasOption("paramOpt");
                InitialState initState = InitialState.RANDOM;

                SHLDA sampler = new SHLDA();
                sampler.setVerbose(verbose);
                sampler.setDebug(debug);
                sampler.setWordVocab(data.getWordVocab());

                // training data
                ArrayList<Integer> trInstIndices = fold.getTrainingInstances();
                int[][][] trRevWords = getRevSentWords(trInstIndices);
                double[] trResponses = getResponses(trInstIndices, responses);

                // test data
                ArrayList<Integer> teInstIndices = fold.getTestingInstances();
                int[][][] teRevWords = getRevSentWords(teInstIndices);
                double[] teResponses = getResponses(teInstIndices, responses);
                
                if(verbose){
//                    System.out.println()
                }

                sampler.configure(foldFolder,
                        trRevWords, trResponses,
                        V, L,
                        alpha,
                        rho,
                        gem_mean, gem_scale,
                        tau_mean, tau_scale,
                        betas, gammas,
                        mus, sigmas,
                        null, numLexicalItems,
                        initState, paramOpt,
                        burnIn, maxIters, sampleLag, repInterval);

                File samplerFolder = new File(foldFolder, sampler.getSamplerFolder());
                IOUtils.createFolder(samplerFolder);

                if (runMode.equals("train")) {
                    sampler.initialize();
                    sampler.iterate();
                    sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                    sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile), topicCoherence);

                    sampler.outputTopicWordDistributions(new File(samplerFolder, "topic-word.txt"));
                    sampler.outputLexicalWeights(new File(samplerFolder, "lexical-reg-params.txt"));
                    sampler.outputDocPathAssignments(new File(samplerFolder, "doc-topic.txt"));
                } else if (runMode.equals("test")) {
                    sampler.testSampler(teRevWords);
                    File teResultFolder = new File(samplerFolder, "te-results");
                    IOUtils.createFolder(teResultFolder);
                    sampler.computeSingleFinal(teResultFolder, teResponses);
                    sampler.computeSingleAverage(teResultFolder, teResponses);
                    sampler.computeMultipleFinal(teResultFolder, teResponses);
                    sampler.computeMultipleAverage(teResultFolder, teResponses);
//                    sampler.outputDocTopicDistributions(samplerFolder + "te-doc-topic.txt");
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

    public static void runModels() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String outputFolder = cmd.getOptionValue("output");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            data = new SingleResponseTextDataset(datasetName, datasetFolder);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());

            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
            numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            int V = data.getWordVocab().size();
            int L = CLIUtils.getIntegerArgument(cmd, "tree-height", 3);
            double gem_mean = CLIUtils.getDoubleArgument(cmd, "gem-mean", 0.3);
            double gem_scale = CLIUtils.getDoubleArgument(cmd, "gem-scale", 50);

            double[] defaultBetas = new double[L];
            defaultBetas[0] = 1;
            for (int i = 1; i < L; i++) {
                defaultBetas[i] = 1.0 / (i + 1);
            }
            double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", defaultBetas, ",");
            for (int i = 0; i < betas.length; i++) {
                betas[i] = betas[i] * V;
            }

            double[] defaultGammas = new double[L - 1];
            for (int i = 0; i < defaultGammas.length; i++) {
                defaultGammas[i] = 1.0;
            }

            double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", defaultGammas, ",");
            double[] responses = data.getResponses();
            if (cmd.hasOption("s")) {
                ZNormalizer zNorm = new ZNormalizer(responses);
                for (int i = 0; i < responses.length; i++) {
                    responses[i] = zNorm.normalize(responses[i]);
                }
            }

            double meanResponse = StatisticsUtils.mean(responses);

            double[] defaultMus = new double[L];
            for (int i = 0; i < L; i++) {
                defaultMus[i] = meanResponse;
            }
            double[] mus = CLIUtils.getDoubleArrayArgument(cmd, "mus", defaultMus, ",");

            double[] defaultSigmas = new double[L];
            defaultSigmas[0] = 0.0001; // root node
            for (int l = 1; l < L; l++) {
                defaultSigmas[l] = 0.5 * l;
            }
            double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", defaultSigmas, ",");

            double tau_mean = CLIUtils.getDoubleArgument(cmd, "tau-mean", 0.0);
            double tau_scale = CLIUtils.getDoubleArgument(cmd, "tau-scale", 1.0);
            double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 1.0);
            double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
            int numLexicalItems = CLIUtils.getIntegerArgument(cmd, "num-lex-items", V);

            boolean paramOpt = cmd.hasOption("paramOpt");

            System.out.println("\nRunning model ...");
            Experiment expt = new Experiment(data);
            expt.configure(burnIn, maxIters, sampleLag, repInterval, numTopWords);
            expt.setup();
            expt.runSampler(outputFolder,
                    data.getSentenceWords(), responses,
                    L,
                    alpha, rho,
                    gem_mean, gem_scale,
                    tau_mean, tau_scale,
                    betas, gammas, mus, sigmas,
                    null, numLexicalItems,
                    paramOpt, verbose, debug);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' "
                    + "main.RunLexicalSHLDA -help", options);
        }
    }

    private static double[] getResponses(ArrayList<Integer> instances, double[] responses) {
        double[] res = new double[instances.size()];
        for (int i = 0; i < res.length; i++) {
            int idx = instances.get(i);
            res[i] = responses[idx];
        }
        return res;
    }

    private static int[][][] getRevSentWords(ArrayList<Integer> instances) {
        int[][][] revSentWords = new int[instances.size()][][];
        for (int i = 0; i < revSentWords.length; i++) {
            int idx = instances.get(i);
            revSentWords[i] = data.getSentenceWords()[idx];
        }
        return revSentWords;
    }

    static class Experiment extends AbstractExperiment<TextDataset> {

        protected static int reportInterval;
        protected static int numTopWords;
        protected static MimnoTopicCoherence topicCoherence;

        public Experiment(TextDataset d) {
            this.data = d;
        }

        public void configure(
                int burnIn,
                int maxIters,
                int sampleLag,
                int repInt,
                int nTopwords) {
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
                int[][][] words,
                double[] responses,
                int L,
                double alpha,
                double rho,
                double gem_mean,
                double gem_scale,
                double tau_mean,
                double tau_scale,
                double[] betas,
                double[] gammas,
                double[] mus,
                double[] sigmas,
                double[] lexicalWeights, // weights for all lexical items (i.e., words)
                int numLexicalItems,
                boolean paramOpt,
                boolean verbose,
                boolean debug) throws Exception {
            int V = data.getWordVocab().size();
            InitialState initState = InitialState.RANDOM;

            SHLDA sampler = new SHLDA();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());

            sampler.configure(resultFolder,
                    words, responses,
                    V, L,
                    alpha,
                    rho,
                    gem_mean, gem_scale,
                    tau_mean, tau_scale,
                    betas, gammas,
                    mus, sigmas,
                    lexicalWeights, numLexicalItems,
                    initState, paramOpt,
                    burn_in, max_iters, sample_lag, reportInterval);

            File shldaFolder = new File(resultFolder, sampler.getSamplerFolder());
            IOUtils.createFolder(shldaFolder);
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(shldaFolder, TopWordFile), numTopWords);
            sampler.outputTopicCoherence(new File(shldaFolder, TopicCoherenceFile), topicCoherence);
            sampler.outputLexicalWeights(new File(shldaFolder, "lexical-reg-params.txt"));
            sampler.outputDocPathAssignments(new File(shldaFolder, "doc-topic.txt"));
            sampler.outputTopicWordDistributions(new File(shldaFolder, "topic-word.txt"));
        }

        @Override
        public void run() {
        }

        @Override
        public void evaluate() {
        }
    }
}
