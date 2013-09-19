/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import data.CorpusProcessor;
import data.MultiLabelTextData;
import data.SingleResponseTextDataset;
import data.TextDataset;
import java.io.File;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;

/**
 *
 * @author vietan
 */
public class ProcessData {

    private static CommandLineParser parser;
    private static Options options;
    private static CommandLine cmd;

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            options.addOption(OptionBuilder.withLongOpt("dataset")
                    .withDescription("Dataset name")
                    .hasArg()
                    .withArgName("Dataset")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("output-folder")
                    .withDescription("Folder that stores the processed data")
                    .hasArg()
                    .withArgName("Folder directory")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("text-data")
                    .withDescription("Directory of the text data")
                    .hasArg()
                    .withArgName("Text data")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("response-file")
                    .withDescription("Directory of the response file")
                    .hasArg()
                    .withArgName("Response file")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("label-file")
                    .withDescription("Directory of the label file")
                    .hasArg()
                    .withArgName("Response file")
                    .create());
            
            options.addOption(OptionBuilder.withLongOpt("format-folder")
                    .withDescription("Folder holding formatted data")
                    .hasArg()
                    .withArgName("Response file")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("u")
                    .withDescription("The minimum count of raw unigrams")
                    .hasArg()
                    .withArgName("Unigram count cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("b")
                    .withDescription("The minimum count of raw bigrams")
                    .hasArg()
                    .withArgName("Bigram count cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("bs")
                    .withDescription("The minimum score of bigrams")
                    .hasArg()
                    .withArgName("Bigram score cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("V")
                    .withDescription("The maximum vocab size")
                    .hasArg()
                    .withArgName("Maximum vocab size")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("min-tf")
                    .withDescription("The minimum term frequency")
                    .hasArg()
                    .withArgName("Term frequency minimum cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("max-tf")
                    .withDescription("The maximum term frequency")
                    .hasArg()
                    .withArgName("Term frequency maximum cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("min-df")
                    .withDescription("The minimum document frequency")
                    .hasArg()
                    .withArgName("Document frequency minimum cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("max-df")
                    .withDescription("The maximum documnet frequency")
                    .hasArg()
                    .withArgName("Document frequency maximum cutoff")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("min-doc-length")
                    .withDescription("The minimum document length")
                    .hasArg()
                    .withArgName("Document minimum length")
                    .create());

            options.addOption("s", false, "Whether stopwords are filtered");
            options.addOption("l", false, "Whether lemmatization is performed");
            options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData -help", options);
                return;
            }

            processData();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void processData() {
        System.out.println("\nProcessing data ...");

        try {
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("output-folder");
            String textInputData = cmd.getOptionValue("text-data");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");

            int unigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "u", 0);
            int bigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "b", 0);
            double bigramScoreCutoff = CLIUtils.getDoubleArgument(cmd, "bs", 5.0);
            int maxVocabSize = CLIUtils.getIntegerArgument(cmd, "V", Integer.MAX_VALUE);
            int vocTermFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-tf", 0);
            int vocTermFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-tf", Integer.MAX_VALUE);
            int vocDocFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-df", 0);
            int vocDocFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-df", Integer.MAX_VALUE);
            int docTypeCountCutoff = CLIUtils.getIntegerArgument(cmd, "min-doc-length", 1);

            boolean stopwordFilter = cmd.hasOption("s");
            boolean lemmatization = cmd.hasOption("l");

            CorpusProcessor corpProc = new CorpusProcessor(
                    unigramCountCutoff,
                    bigramCountCutoff,
                    bigramScoreCutoff,
                    maxVocabSize,
                    vocTermFreqMinCutoff,
                    vocTermFreqMaxCutoff,
                    vocDocFreqMinCutoff,
                    vocDocFreqMaxCutoff,
                    docTypeCountCutoff,
                    stopwordFilter,
                    lemmatization);

            if (cmd.hasOption("response-file")) {
                String responseFile = cmd.getOptionValue("response-file");
                SingleResponseTextDataset dataset = new SingleResponseTextDataset(datasetName, datasetFolder, corpProc);

                // load text data
                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }
                dataset.loadResponses(responseFile); // load response data
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            } else if (cmd.hasOption("label-file")) {
                String labelFile = cmd.getOptionValue("label-file");
                MultiLabelTextData dataset = new MultiLabelTextData(datasetName, datasetFolder, corpProc);

                // load text data
                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }
                dataset.loadLabels(labelFile);
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            } else {
                TextDataset dataset = new TextDataset(datasetName, datasetFolder, corpProc);

                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.ProcessData -help", options);
        }
    }
}