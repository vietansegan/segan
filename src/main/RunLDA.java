/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractRunner;
import core.AbstractSampler.InitialState;
import data.TextDataset;
import java.io.File;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.LDA;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class RunLDA extends AbstractRunner {

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("output", "Output folder");
            addOption("dataset", "Dataset");
            addOption("folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");

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

            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp dist/segan.jar main.RunLDA -help", options);
                return;
            }
            
            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            runModels();
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.RunLDA -help", options);
            System.exit(1);
        }
    }

    public static void runModels() throws Exception {
        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        // data 
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("folder");
        String outputFolder = cmd.getOptionValue("output");
        String formatFolder = cmd.getOptionValue("format-folder");
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);

        // sampler
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        boolean paramOpt = cmd.hasOption("paramOpt");

        TextDataset dataset = new TextDataset(datasetName, datasetFolder);
        dataset.loadFormattedData(new File(dataset.getDatasetFolderPath(), formatFolder));
        dataset.prepareTopicCoherence(numTopWords);

        if (verbose) {
            System.out.println("\nRunning model ...");
        }
        int V = dataset.getWordVocab().size();
        InitialState initState = InitialState.RANDOM;

        LDA sampler = new LDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(dataset.getWordVocab());

        sampler.configure(outputFolder, dataset.getWords(),
                V, K, alpha, beta, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File ldaFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(ldaFolder);
        sampler.sample();
        sampler.outputTopicTopWords(new File(ldaFolder, TopWordFile), numTopWords);
        sampler.outputTopicCoherence(new File(ldaFolder, TopicCoherenceFile), dataset.getTopicCoherence());
        sampler.outputDocTopicDistributions(new File(ldaFolder, "doc-topic.txt"));
        sampler.outputTopicWordDistributions(new File(ldaFolder, "topic-word.txt"));
    }
}