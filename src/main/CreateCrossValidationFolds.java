/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
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
import util.IOUtils;
import util.StatisticsUtils;

/**
 *
 * @author vietan
 */
public class CreateCrossValidationFolds {

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

            options.addOption(OptionBuilder.withLongOpt("folder")
                    .withDescription("Folder that stores the processed data")
                    .hasArg()
                    .withArgName("Folder directory")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("output")
                    .withDescription("Output folder")
                    .hasArg()
                    .withArgName("Output folder")
                    .create());
            
            options.addOption(OptionBuilder.withLongOpt("format-folder")
                    .withDescription("Folder holding formatted data")
                    .hasArg()
                    .withArgName("Response file")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("num-classes")
                    .withDescription("Number of classes that the response "
                    + "variables are discretized into to perform stratified sampling. "
                    + "Default 1.")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("num-folds")
                    .withDescription("Number of folds. Default 5.")
                    .hasArg()
                    .withArgName("")
                    .create());

            options.addOption(OptionBuilder.withLongOpt("tr2dev-ratio")
                    .withDescription("Training-to-development ratio. "
                    + "Default 0.8.")
                    .hasArg()
                    .withArgName("")
                    .create());

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds -help", options);
                return;
            }

            stratifiedSampling();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void stratifiedSampling() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("folder");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
            SingleResponseTextDataset dataset = new SingleResponseTextDataset(datasetName, datasetFolder);
            dataset.loadFormattedData(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());

            ArrayList<RegressionDocumentInstance> instanceList = new ArrayList<RegressionDocumentInstance>();
            for (int d = 0; d < dataset.getDocIds().length; d++) {
                instanceList.add(new RegressionDocumentInstance(
                        dataset.getDocIds()[d],
                        dataset.getWords()[d],
                        dataset.getResponses()[d]));
            }

            String outputFolder = cmd.getOptionValue("output");
            IOUtils.createFolder(outputFolder);

            String cvName = "";
            CrossValidation<String, RegressionDocumentInstance> cv =
                    new CrossValidation<String, RegressionDocumentInstance>(
                    outputFolder,
                    cvName,
                    instanceList);

            int numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
            double trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 0.8);
            int numClasses = CLIUtils.getIntegerArgument(cmd, "num-classes", 1);

            // create groupIdList based on the response variable
            ArrayList<Integer> groupIdList = StatisticsUtils.discretize(dataset.getResponses(), numClasses);

            System.out.println("\nStratified sampling ... " + outputFolder);
            cv.stratify(groupIdList, numFolds, trToDevRatio);
            cv.outputFolds();
            for (Fold<String, RegressionDocumentInstance> fold : cv.getFolds()) {
                outputLexicalSVMLightData(fold);
            }
            System.out.println("--- Cross validation data are written to " + outputFolder);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds -help", options);
        }
    }

    private static void outputLexicalSVMLightData(Fold<String, RegressionDocumentInstance> fold) throws Exception {
        String featureType = "lexical";
        BufferedWriter writer = IOUtils.getBufferedWriter(fold.getFolder() + "fold-" + fold.getIndex() + "-" + featureType + Fold.TrainingExt);
        for (int idx : fold.getTrainingInstances()) {
            RegressionDocumentInstance inst = fold.getInstance(idx);
            writer.write(inst.getFullVocabSVMLigthString() + "\n");
        }
        writer.close();

        writer = IOUtils.getBufferedWriter(fold.getFolder() + "fold-" + fold.getIndex() + "-" + featureType + Fold.DevelopExt);
        for (int idx : fold.getDevelopmentInstances()) {
            RegressionDocumentInstance inst = fold.getInstance(idx);
            writer.write(inst.getFullVocabSVMLigthString() + "\n");
        }
        writer.close();

        writer = IOUtils.getBufferedWriter(fold.getFolder() + "fold-" + fold.getIndex() + "-" + featureType + Fold.TestExt);
        for (int idx : fold.getTestingInstances()) {
            RegressionDocumentInstance inst = fold.getInstance(idx);
            writer.write(inst.getFullVocabSVMLigthString() + "\n");
        }
        writer.close();
    }
}
