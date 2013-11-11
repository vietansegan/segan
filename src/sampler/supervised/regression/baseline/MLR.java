package sampler.supervised.regression.baseline;

import core.AbstractRunner;
import core.crossvalidation.Fold;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import optimization.GurobiMLRL1Norm;
import optimization.GurobiMLRL2Norm;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.supervised.Regressor;
import util.CLIUtils;
import util.IOUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class MLR extends AbstractRunner implements Regressor<ResponseTextDataset> {

    public static enum Regularizer {

        L1, L2
    }
    private Regularizer regularizer;
    private double[] weights;
    private double[] predictions;
    private double param;

    public MLR(Regularizer reg, double t) {
        this.regularizer = reg;
        this.param = t;
    }

    @Override
    public String getName() {
        return "MLR-" + regularizer + "-" + param;
    }

    @Override
    public void train(ResponseTextDataset trainData) {
        if (verbose) {
            System.out.println("Training ...");
        }
        int[][] trWords = trainData.getWords();
        double[] trResponses = trainData.getResponses();

        int D = trWords.length;
        int V = trainData.getWordVocab().size();
        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < trWords[d].length; n++) {
                designMatrix[d][trWords[d][n]]++;
            }
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] /= trWords[d].length;
            }
        }

        if (regularizer == Regularizer.L1) {
            GurobiMLRL1Norm mlr = new GurobiMLRL1Norm(designMatrix, trResponses, param);
            this.weights = mlr.solve();
        } else if (regularizer == Regularizer.L2) {
            GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, trResponses, param);
            this.weights = mlr.solve();
        } else {
            throw new RuntimeException(regularizer + " regularization is not supported");
        }

        if (debug) {
            double[] predResponses = new double[D];
            for (int d = 0; d < D; d++) {
                for (int v = 0; v < V; v++) {
                    predResponses[d] += designMatrix[d][v] * weights[v];
                }
            }
            RegressionEvaluation eval = new RegressionEvaluation(
                    trResponses, predResponses);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                System.out.println(measurement.getName()
                        + "\t" + measurement.getValue()
                        + "\n");
            }
        }
    }

    @Override
    public void test(ResponseTextDataset testData) {
        if (verbose) {
            System.out.println("Testing ...");
        }
        int[][] teWords = testData.getWords();

        int D = teWords.length;
        int V = testData.getWordVocab().size();
        double[][] desginMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < teWords[d].length; n++) {
                desginMatrix[d][teWords[d][n]]++;
            }
            for (int v = 0; v < V; v++) {
                desginMatrix[d][v] /= teWords[d].length;
            }
        }

        predictions = new double[D];
        for (int d = 0; d < D; d++) {
            double predVal = 0.0;
            for (int v = 0; v < V; v++) {
                predVal += desginMatrix[d][v] * this.weights[v];
            }

            predictions[d] = predVal;
        }
        
        if (debug) {
            RegressionEvaluation eval = new RegressionEvaluation(
                    testData.getResponses(), predictions);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                System.out.println(measurement.getName()
                        + "\t" + measurement.getValue()
                        + "\n");
            }
        }
    }

    public double[] getPredictions() {
        return this.predictions;
    }

    public void outputPrediction(File predictionFile) {
        if (verbose) {
            System.out.println("Outputing predictions to " + predictionFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(predictionFile);
            writer.write(predictions.length + "\n");
            for (int d = 0; d < predictions.length; d++) {
                writer.write(Integer.toString(d)
                        + "\t" + predictions[d]
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing predictions to "
                    + predictionFile);
        }
    }

    public void inputPrediction(File predictionFile) {
        if (verbose) {
            System.out.println("Inputing predictions from " + predictionFile);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(predictionFile);
            int num = Integer.parseInt(reader.readLine());
            String line;
            int count = 0;
            predictions = new double[num];
            while ((line = reader.readLine()) != null) {
                predictions[count++] = Double.parseDouble(line.split("\t")[1]);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictions from "
                    + predictionFile);
        }
    }

    public void outputEvaluation(File evaluationFile,
            double[] trueResponses, double[] predResponses) {
        if (verbose) {
            System.out.println("Outputing evaluation to " + evaluationFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(evaluationFile);
            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, predResponses);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                writer.write(measurement.getName()
                        + "\t" + measurement.getValue()
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing evaluation to "
                    + evaluationFile);
        }
    }

    @Override
    public void output(File file) {
        if (verbose) {
            System.out.println("Outputing model to " + file);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(weights.length + "\n");
            for (int ii = 0; ii < weights.length; ii++) {
                writer.write(weights[ii] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    @Override
    public void input(File file) {
        if (verbose) {
            System.out.println("Inputing model from " + file);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            int V = Integer.parseInt(reader.readLine());
            this.weights = new double[V];
            for (int ii = 0; ii < V; ii++) {
                this.weights[ii] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + file);
        }
    }

    public double[] getWeights() {
        return this.weights;
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar:dist/lib/*' " + MLR.class.getName() + " -help";
    }

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("dataset", "Dataset");
            addOption("data-folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("format-file", "Formatted file name");
            addOption("output", "Output folder");

            // running configurations
            addOption("cv-folder", "Cross validation folder");
            addOption("num-folds", "Number of folds");
            addOption("fold", "The cross-validation fold to run");
            addOption("run-mode", "Running mode");

            addOption("regularizer", "Regularizer (L1, L2)");
            addOption("param", "Parameter");

            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("z", false, "standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
                runModel();
            }

        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    private static void runModel() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String outputFolder = cmd.getOptionValue("output");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        String regularizer = cmd.getOptionValue("regularizer");
        double param = Double.parseDouble(cmd.getOptionValue("param"));

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        ResponseTextDataset data = new ResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());

        if (cmd.hasOption("z")) {
            data.zNormalize();
        }

        if (verbose) {
            System.out.println("--- Loaded. " + data.toString());
        }

        MLR mlr;
        if (regularizer.equals("L1")) {
            mlr = new MLR(Regularizer.L1, param);
        } else if (regularizer.equals("L2")) {
            mlr = new MLR(Regularizer.L2, param);
        } else {
            throw new RuntimeException(regularizer + " regularization is not supported");
        }
        File mlrFolder = new File(outputFolder, mlr.getName());
        IOUtils.createFolder(mlrFolder);
        mlr.train(data);
        mlr.output(new File(mlrFolder, "model"));
    }

    private static void runCrossValidation() throws Exception {
        String cvFolder = cmd.getOptionValue("cv-folder");
        int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));
        String resultFolder = cmd.getOptionValue("output");

        String regularizer = cmd.getOptionValue("regularizer");
        double param = Double.parseDouble(cmd.getOptionValue("param"));
        int foldIndex = -1;
        if (cmd.hasOption("fold")) {
            foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
        }

        for (int ii = 0; ii < numFolds; ii++) {
            if (foldIndex != -1 && ii != foldIndex) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + foldIndex);
            }

            Fold fold = new Fold(ii, cvFolder);
            File foldFolder = new File(resultFolder, fold.getFoldName());
            ResponseTextDataset[] foldData = ResponseTextDataset.loadCrossValidationFold(fold);
            ResponseTextDataset trainData = foldData[Fold.TRAIN];
            ResponseTextDataset devData = foldData[Fold.DEV];
            ResponseTextDataset testData = foldData[Fold.TEST];

            if (cmd.hasOption("z")) {
                ResponseTextDataset.zNormalize(trainData, devData, testData);
            }

            if (verbose) {
                System.out.println("Fold " + fold.getFoldName());
                System.out.println("--- training: " + trainData.toString());
                System.out.println("--- development: " + devData.toString());
                System.out.println("--- test: " + testData.toString());
                System.out.println();
            }

            MLR mlr;
            if (regularizer.equals("L1")) {
                mlr = new MLR(Regularizer.L1, param);
            } else if (regularizer.equals("L2")) {
                mlr = new MLR(Regularizer.L2, param);
            } else {
                throw new RuntimeException(regularizer + " regularization is not supported");
            }
            File mlrFolder = new File(foldFolder, mlr.getName());
            IOUtils.createFolder(mlrFolder);
            mlr.train(trainData);
            mlr.test(testData);
            mlr.outputPrediction(new File(mlrFolder, "predictions.txt"));
            mlr.outputEvaluation(new File(mlrFolder, "evaluation.txt"),
                    testData.getResponses(), mlr.predictions);
        }
    }
}
