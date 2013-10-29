package sampler.supervised.regression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;
import util.StatisticsUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class GibbsRegressorUtils {

    public static final String SINGLE_FINAL = "single-final.txt";
    public static final String SINGLE_AVG = "single-avg.txt";
    public static final String MULTIPLE_FINAL = "multiple-final.txt";
    public static final String MULTIPLE_AVG = "multiple-avg.txt";

    /**
     * Output the predictions of a single model (learning at an iteration during
     * training) on test documents.
     *
     * @param file The output file
     * @param predictions The list of predicted values, each for a test
     * document.
     */
    public static void outputSingleModelPredictions(
            File file,
            ArrayList<double[]> predictions) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int d = 0; d < predictions.size(); d++) {
                writer.write(Integer.toString(d));

                for (int ii = 0; ii < predictions.size(); ii++) {
                    writer.write("\t" + predictions.get(ii)[d]);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing predictions to "
                    + file);
        }
    }

    /**
     * Load the predictions that a model at a single iteration (learned during
     * training) makes on a set of test documents.
     *
     * @param file The file containing the prediction result
     * @param numDocs Number of test documents
     */
    public static double[][] inputSingleModelPredictions(File file, int numDocs) {
        double[][] preds = new double[numDocs][];
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            String line;
            String[] sline;
            int count = 0;
            while ((line = reader.readLine()) != null) {
                sline = line.split("\t");
                double[] ps = new double[sline.length - 1];
                for (int ii = 0; ii < ps.length; ii++) {
                    ps[ii] = Double.parseDouble(sline[ii + 1]);
                }
                preds[count] = ps;
                count++;
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading predictions from "
                    + file);
        }
        return preds;
    }

    public static void evaluate(File iterPredFolder,
            File outputFolder,
            double[] trueResponses) {
        computeSingleFinal(iterPredFolder, outputFolder, trueResponses);
        computeSingleAverage(iterPredFolder, outputFolder, trueResponses);
        computeMultipleFinal(iterPredFolder, outputFolder, trueResponses);
        computeMultipleAverage(iterPredFolder, outputFolder, trueResponses);

    }

    /**
     * Evaluation using only the predicted values at the final iteration during
     * test time. This will output the results using all reported models in
     * iterPredFolder.
     *
     * @param iterPredFolder Folder containing predictions, each file in which
     * corresponds to a model learned during training
     * @param outputFolder The output folder
     * @param trueResponses The true values
     */
    public static void computeSingleFinal(
            File iterPredFolder,
            File outputFolder,
            double[] trueResponses) {
        try {
            File outputFile = new File(outputFolder, SINGLE_FINAL);
            String[] filenames = iterPredFolder.list();

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                double[][] predictions = inputSingleModelPredictions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                // get the predictions at the final iterations during test time
                double[] finalPred = new double[predictions.length];
                for (int d = 0; d < finalPred.length; d++) {
                    finalPred[d] = predictions[d][predictions[0].length - 1];
                }
                RegressionEvaluation eval = new RegressionEvaluation(
                        trueResponses, finalPred);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();

                writer.write(filename);
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getValue());
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating single-final");
        }
    }

    /**
     * Evaluating by averaging the predicted values across different iterations
     * during test time from a single model.
     *
     * @param iterPredFolder Folder containing predictions, each file in which
     * corresponds to a model learned during training
     * @param outputFolder The output folder
     * @param trueResponses The true values
     */
    public static void computeSingleAverage(
            File iterPredFolder,
            File outputFolder,
            double[] trueResponses) {
        try {
            File outputFile = new File(outputFolder, SINGLE_AVG);
            String[] filenames = iterPredFolder.list();

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                double[][] predictions = inputSingleModelPredictions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                // compute the prediction values as the average values
                double[] avgPred = new double[predictions.length];
                for (int d = 0; d < avgPred.length; d++) {
                    avgPred[d] = StatisticsUtils.mean(predictions[d]);
                }

                RegressionEvaluation eval = new RegressionEvaluation(
                        trueResponses, avgPred);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();

                writer.write(filename);
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getValue());
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating single-avg.");
        }
    }

    /**
     * Evaluating by averaging the final predicted values from multiple models
     * learned during training.
     *
     * @param iterPredFolder Folder containing predictions, each file in which
     * corresponds to a model learned during training
     * @param outputFolder The output folder
     * @param trueResponses The true values
     */
    public static void computeMultipleFinal(
            File iterPredFolder,
            File outputFolder,
            double[] trueResponses) {
        try {
            File outputFile = new File(outputFolder, MULTIPLE_FINAL);
            String[] filenames = iterPredFolder.list();

            double[] predResponses = new double[trueResponses.length];
            int numModels = filenames.length;

            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                double[][] predictions = inputSingleModelPredictions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                for (int d = 0; d < trueResponses.length; d++) {
                    predResponses[d] += predictions[d][predictions[0].length - 1];
                }
            }

            for (int d = 0; d < predResponses.length; d++) {
                predResponses[d] /= numModels;
            }

            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, predResponses);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating multiple-final.");
        }
    }

    /**
     * Evaluating by averaging over multiple averaged predicted values.
     *
     * @param iterPredFolder Folder containing predictions, each file in which
     * corresponds to a model learned during training
     * @param outputFolder The output folder
     * @param trueResponses The true values
     */
    public static void computeMultipleAverage(
            File iterPredFolder,
            File outputFolder,
            double[] trueResponses) {
        try {
            File outputFile = new File(outputFolder, MULTIPLE_AVG);
            String[] filenames = iterPredFolder.list();

            double[] predResponses = new double[trueResponses.length];
            int numModels = filenames.length;

            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                double[][] predictions = inputSingleModelPredictions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                for (int d = 0; d < trueResponses.length; d++) {
                    predResponses[d] += StatisticsUtils.mean(predictions[d]);
                }
            }

            for (int d = 0; d < predResponses.length; d++) {
                predResponses[d] /= numModels;
            }

            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, predResponses);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating multiple-avg.");
        }
    }
}
