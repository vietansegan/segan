package main;

import core.AbstractRunner;
import core.AbstractSampler.InitialState;
import data.LabelSingleResponseTextDataset;
import java.io.File;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.supervised.regression.LabeledSHLDASampler;
import util.IOUtils;
import util.StatisticsUtils;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class RunLabeledSHLDA extends AbstractRunner {
    private static LabelSingleResponseTextDataset data;
    
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
//                runCrossValidation();
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
    
    public static void runModels() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String outputFolder = cmd.getOptionValue("output");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);

            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            int V = data.getWordVocab().size();
            int L = CLIUtils.getIntegerArgument(cmd, "tree-height", 3);
            double gem_mean = CLIUtils.getDoubleArgument(cmd, "gem-mean", 0.3);
            double gem_scale = CLIUtils.getDoubleArgument(cmd, "gem-scale", 50);
            
            data = new LabelSingleResponseTextDataset(datasetName, datasetFolder);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            data.prepareTopicCoherence(numTopWords);

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
            defaultSigmas[1] = 0.1;
            for (int l = 2; l < L; l++) {
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
            LabeledSHLDASampler sampler = new LabeledSHLDASampler();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());
            sampler.setLabelVocab(data.getLabelVocab());
            InitialState initState = InitialState.RANDOM;
            sampler.setNumLexicalItems(numLexicalItems);
            sampler.setLexicalWeights(null);
            sampler.configure(outputFolder,
                    data.getSentenceWords(), responses, data.getLabels(),
                    V, L,
                    alpha,
                    rho,
                    gem_mean, gem_scale,
                    tau_mean, tau_scale,
                    betas, gammas,
                    mus, sigmas,
                    initState, paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);

            File shldaFolder = new File(outputFolder, sampler.getSamplerFolder());
            IOUtils.createFolder(shldaFolder);
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(shldaFolder, TopWordFile), numTopWords);
            sampler.outputTopicCoherence(new File(shldaFolder, TopicCoherenceFile), data.getTopicCoherence());
            sampler.outputLexicalWeights(new File(shldaFolder, "lexical-reg-params.txt"));
            sampler.outputDocPathAssignments(new File(shldaFolder, "doc-topic.txt"));
            sampler.outputTopicWordDistributions(new File(shldaFolder, "topic-word.txt"));
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' "
                    + "main.RunLexicalSHLDA -help", options);
        }
    }
}
