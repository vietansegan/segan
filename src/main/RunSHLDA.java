/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractExperiment;
import core.AbstractSampler.InitialState;
import data.SingleResponseTextDataset;
import data.TextDataset;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampler.supervised.SHLDASampler;
import util.IOUtils;
import util.StatisticsUtils;
import util.evaluation.MimnoTopicCoherence;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class RunSHLDA {
    private static CommandLineParser parser;
    private static Options options;
    private static CommandLine cmd;
    
    public static void main(String[] args){
        try{
            // create the command line parser
            parser = new BasicParser();
            
            // create the Options
            options = new Options();
            
            options.addOption( OptionBuilder.withLongOpt( "output" )
                            .withDescription( "Output folder" )
                            .hasArg()
                            .withArgName("Output folder")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "dataset" )
                            .withDescription( "Dataset name" )
                            .hasArg()
                            .withArgName("Dataset")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "folder" )
                            .withDescription( "Folder that stores the processed data" )
                            .hasArg()
                            .withArgName("Folder directory")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "burnIn" )
                            .withDescription( "Burn-in. Default 250." )
                            .hasArg()
                            .withArgName("Burn-in")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "maxIter" )
                            .withDescription( "Maximum number of iterations" )
                            .hasArg()
                            .withArgName("Maximum number of iterations")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "sampleLag" )
                            .withDescription( "Sample lag" )
                            .hasArg()
                            .withArgName("sample lag")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "gem-mean" )
                            .withDescription( "GEM mean. Default: 0.5" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "gem-scale" )
                            .withDescription( "GEM scale. Default: 50" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "betas" )
                            .withDescription( "Vector of Dirichlet priors for topid distributions. Default: [1, 0.5, 0.25] for a 3-level tree" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "gammas" )
                            .withDescription( "Vector of hyperparameters for DPs. Default: [1.0, 1.0] for a 3-level tree" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "mus" )
                            .withDescription( "Vector of the prior means for regression parameters. Default: [0.0, 0.0, 0.0] for a 3-level tree" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "sigmas" )
                            .withDescription( "Vector of the prior variances for regression parameters. Default: [0.0001, 0.5, 1.0] for a 3-level tree" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "rho" )
                            .withDescription( "Variance of the response variable. Default: 1.0" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption( OptionBuilder.withLongOpt( "report" )
                            .withDescription( "Report interval" )
                            .hasArg()
                            .withArgName("")
                            .create() );
            
            options.addOption("paramOpt", false, "Whether hyperparameter optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("s", false, "whether standardize (z-score normalization)");
            options.addOption("help", false, "Help");
            
            cmd = parser.parse( options, args );
            if(cmd.hasOption("help")){
                CLIUtils.printHelp("java -cp dist/segan.jar main.RunSHLDA -help", options);
                return;
            }
            
            runModels();
        }
        catch(Exception e){
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.RunSHLDA -help", options);
            System.exit(1);
        }
    }
    
    public static void runModels(){
        try{
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String outputFolder = cmd.getOptionValue("output");
            SingleResponseTextDataset dataset = new SingleResponseTextDataset(datasetName, datasetFolder);
            dataset.loadFormattedData();
            
            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
            int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            int V = dataset.getWordVocab().size();
            int L = CLIUtils.getIntegerArgument(cmd, "tree-height", 3);
            double gem_mean = CLIUtils.getDoubleArgument(cmd, "gem-mean", 0.3);
            double gem_scale = CLIUtils.getDoubleArgument(cmd, "gem-scale", 50);
            
            double[] defaultBetas = new double[L];
            defaultBetas[0] = 1;
            for(int i=1; i<L; i++)
                defaultBetas[i] = 1.0 / (i+1);
            double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", defaultBetas, ",");
            for(int i=0; i<betas.length; i++)
                betas[i] = betas[i] * V;
            
            double[] defaultGammas = new double[L-1];
            for(int i=0; i<defaultGammas.length; i++)
                defaultGammas[i] = 1.0;
            
            double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", defaultGammas, ",");
            
            double[] responses = dataset.getResponses();
            if(cmd.hasOption("s")){
                ZNormalizer zNorm = new ZNormalizer(responses);
                for(int i=0; i<responses.length; i++)
                    responses[i] = zNorm.normalize(responses[i]);
            }
                        
            double meanResponse = StatisticsUtils.mean(responses);
            
            double[] defaultMus = new double[L];
            for(int i=0; i<L; i++)
                defaultMus[i] = meanResponse;
            double[] mus = CLIUtils.getDoubleArrayArgument(cmd, "mus", defaultMus, ",");
            
            double[] defaultSigmas = new double[L];
            defaultSigmas[0] = 0.0001; // root node
            for(int l=1; l<L; l++)
                defaultSigmas[l] = 0.5 * l;
            double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", defaultSigmas, ",");
            
            double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
            int regLevel = 1;
            
            boolean paramOpt = cmd.hasOption("paramOpt");
            boolean verbose = cmd.hasOption("v");
            boolean debug = cmd.hasOption("d");
            
            System.out.println("\nRunning model ...");
            Experiment expt = new Experiment(dataset);
            expt.configure(burnIn, maxIters, sampleLag, repInterval, numTopWords);
            expt.setup();
            
            expt.runSampler(outputFolder, dataset.getWords(), responses, 
                    L, 
                    gem_mean, gem_scale, 
                    betas, gammas, mus, sigmas, rho, regLevel,
                    paramOpt, verbose, debug);
        }
        catch(Exception e){
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.RunSHLDA -help", options);
        }
    }
    
    static class Experiment extends AbstractExperiment<TextDataset>{
        protected static int reportInterval;
        protected static int numTopWords;
        protected static MimnoTopicCoherence topicCoherence;

        public Experiment(TextDataset d){
            this.data = d;
        }

        public void configure(int burnIn, int maxIters, int sampleLag, int repInt, int nTopwords){
            burn_in = burnIn;
            max_iters = maxIters;
            sample_lag = sampleLag;
            reportInterval = repInt;
            numTopWords = nTopwords;
        }

        @Override
        public void setup(){
            topicCoherence = new MimnoTopicCoherence(data.getWords(), data.getWordVocab().size(), numTopWords);
            topicCoherence.prepare();
        }
        
        public void runSampler(
                String resultFolder, 
                int[][] words,
                double[] responses,
                int L,
                double gem_mean,
                double gem_scale,
                double[] betas,
                double[] gammas,
                double[] mus, 
                double[] sigmas,
                double rho,
                int regLevel,
                boolean paramOpt,
                boolean verbose,
                boolean debug
                ) throws Exception{
            int V = data.getWordVocab().size();
            InitialState initState = InitialState.RANDOM;
            
            SHLDASampler sampler = new SHLDASampler();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());
            
            sampler.configure(resultFolder, words, responses,
                V, L, gem_mean, gem_scale, betas, gammas, 
                mus, sigmas, rho, regLevel,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, reportInterval);
            
            String shldaFolder = resultFolder + sampler.getSamplerFolder();
            IOUtils.createFolder(shldaFolder);
            
            sampler.sample();
            
            sampler.outputTopicTopWords(shldaFolder + TopWordFile, numTopWords);
            sampler.outputTopicCoherence(shldaFolder + TopicCoherenceFile, topicCoherence);
            sampler.outputDocPathAssignments(shldaFolder + "doc-topic.txt");
            sampler.outputTopicWordDistributions(shldaFolder + "topic-word.txt");
        }
        
        @Override
        public void run(){

        }

        @Override
        public void evaluate(){

        }
    }
}
