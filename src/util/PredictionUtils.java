package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class PredictionUtils {

    public static final int POSITVE = 1;
    public static final int NEGATIVE = -1;
    public static final String SINGLE_FINAL = "single-final.txt";
    public static final String SINGLE_AVG = "single-avg.txt";
    public static final String MULTIPLE_FINAL = "multiple-final.txt";
    public static final String MULTIPLE_AVG = "multiple-avg.txt";

    /**
     * Input predictions.
     *
     * @param inputFile The input file
     */
    public static double[] inputPredictions(File inputFile) {
        double[] predResponses = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            int numInst = Integer.parseInt(reader.readLine());
            predResponses = new double[numInst];
            for (int i = 0; i < numInst; i++) {
                String[] sline = reader.readLine().split("\t");
                predResponses[i] = Double.parseDouble(sline[2]);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading predictions from "
                    + inputFile);
        }
        return predResponses;
    }

    /**
     * Output predictions.
     *
     * @param outputFile The output file
     * @param instanceIds List of instance IDs
     * @param trueValues List of true values
     * @param predValues List of predicted values
     *
     */
    public static void outputRegressionPredictions(
            File outputFile,
            String[] instanceIds,
            double[] trueValues,
            double[] predValues) {
        if (instanceIds.length != trueValues.length
                || instanceIds.length != predValues.length) {
            throw new RuntimeException("Lengths mismatched. "
                    + "\t" + instanceIds.length
                    + "\t" + trueValues.length
                    + "\t" + predValues.length);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(instanceIds.length + "\n");
            for (int i = 0; i < instanceIds.length; i++) {
                writer.write(instanceIds[i]
                        + "\t" + trueValues[i]
                        + "\t" + predValues[i]
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing predictions to "
                    + outputFile);
        }
    }

    public static void outputClassificationPredictions(
            File outputFile,
            String[] instanceIds,
            int[] trueLabels,
            double[] predValues) {
        if (instanceIds.length != trueLabels.length
                || instanceIds.length != predValues.length) {
            throw new RuntimeException("Lengths mismatched. "
                    + "\t" + instanceIds.length
                    + "\t" + trueLabels.length
                    + "\t" + predValues.length);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(instanceIds.length + "\n");
            for (int i = 0; i < instanceIds.length; i++) {
                writer.write(instanceIds[i]
                        + "\t" + trueLabels[i]
                        + "\t" + predValues[i]
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing predictions to "
                    + outputFile);
        }
    }

    /**
     * Output classification results.
     *
     * @param outputFile The output file
     * @param labels List of true labels
     * @param preds List of predicted labels
     */
    public static ArrayList<Measurement> outputBinaryClassificationResults(
            File outputFile,
            int[] labels,
            int[] preds) {
        System.out.println("Outputing binary classification results to " + outputFile);
        ArrayList<Measurement> measurements = null;
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            ClassificationEvaluation eval = new ClassificationEvaluation(labels, preds);
            eval.computePRF1();
            measurements = eval.getMeasurements();
            for (Measurement m : measurements) {
                writer.write(m.getName() + "\t" + m.getValue() + "\n");
            }
            writer.close();
            
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing results to "
                    + outputFile);
        }
        return measurements;
    }

    /**
     * Output regression results.
     *
     * @param outputFile The output file
     * @param trueValues List of true values
     * @param predValues List of predicted values
     */
    public static ArrayList<Measurement> outputRegressionResults(
            File outputFile,
            double[] trueValues,
            double[] predValues) {
        ArrayList<Measurement> measurements = null;
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            RegressionEvaluation eval = new RegressionEvaluation(trueValues, predValues);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            eval.computePredictiveRSquared();
            measurements = eval.getMeasurements();
            for (Measurement m : measurements) {
                writer.write(m.getName() + "\t" + m.getValue() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing regression "
                    + "results to " + outputFile);
        }
        return measurements;
    }

    public static ArrayList<Measurement> outputBinaryClassificationResults(
            File outputFile,
            int[] trueLabels,
            double[] predValues) {
        
        int numPositives = 0;
        for (int ii = 0; ii < trueLabels.length; ii++) {
            if (trueLabels[ii] == POSITVE) {
                numPositives++;
            }
        }

        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<RankingItem<Integer>>();
        for (int d = 0; d < predValues.length; d++) {
            rankDocs.add(new RankingItem<Integer>(d, predValues[d]));
        }
        Collections.sort(rankDocs);
        int[] preds = new int[predValues.length];
        for (int ii = 0; ii < numPositives; ii++) {
            int d = rankDocs.get(ii).getObject();
            preds[d] = POSITVE;
        }

        return outputBinaryClassificationResults(outputFile, trueLabels, preds);
    }

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
            for (int d = 0; d < predictions.get(0).length; d++) {
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

    /**
     * Evaluating regression predictions.
     *
     * @param iterPredFolder Prediction folder
     * @param outputFolder Output folder
     * @param trueResponses Ground truth responses
     */
    public static double[] evaluateRegression(
            File iterPredFolder,
            File outputFolder,
            String[] docIds,
            double[] trueResponses) {
        computeSingleFinal(iterPredFolder, outputFolder, docIds, trueResponses);
        computeSingleAverage(iterPredFolder, outputFolder, docIds, trueResponses);
        computeMultipleFinal(iterPredFolder, outputFolder, docIds, trueResponses);
        return computeMultipleAverage(iterPredFolder, outputFolder, docIds, trueResponses);
    }

    public static double[] evaluateBinaryClassification(
            File iterPredFolder,
            File outputFolder,
            String[] docIds,
            int[] trueLabels) {
        return computeBinaryClassificationMultipleAverage(iterPredFolder, outputFolder, docIds, trueLabels);
    }

    public static double[] computeBinaryClassificationMultipleAverage(
            File iterPredFolder,
            File outputFolder,
            String[] docIds,
            int[] trueLabels) {
        double[] predResponses = null;
        try {
            String[] filenames = iterPredFolder.list();

            predResponses = new double[trueLabels.length];
            int numModels = filenames.length;

            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                double[][] predictions = inputSingleModelPredictions(
                        new File(iterPredFolder, filename),
                        trueLabels.length);

                for (int d = 0; d < trueLabels.length; d++) {
                    predResponses[d] += StatisticsUtils.mean(predictions[d]);
                }
            }

            for (int d = 0; d < predResponses.length; d++) {
                predResponses[d] /= numModels;
            }

            outputClassificationPredictions(new File(outputFolder, MULTIPLE_AVG + ".pred"),
                    docIds, trueLabels, predResponses);
            outputBinaryClassificationResults(new File(outputFolder, MULTIPLE_AVG + ".result"),
                    trueLabels, predResponses);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating multiple-avg.");
        }
        return predResponses;
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
            String[] docIds,
            double[] trueResponses) {
        try {
            String[] filenames = iterPredFolder.list();
            
            // debug
//            System.out.println("iter folder: " + iterPredFolder);
//            System.out.println("# files: " + filenames.length);
            
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

                outputRegressionPredictions(
                        new File(outputFolder, SINGLE_FINAL + "-" + filename + ".pred"),
                        docIds, trueResponses, finalPred);
                outputRegressionResults(
                        new File(outputFolder, SINGLE_FINAL + "-" + filename + ".result"),
                        trueResponses, finalPred);
            }
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
            String[] docIds,
            double[] trueResponses) {
        try {
            String[] filenames = iterPredFolder.list();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                // load predicted values
                double[][] predictions = inputSingleModelPredictions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                // compute the prediction values as the average values
                double[] avgPred = new double[predictions.length];
                for (int d = 0; d < avgPred.length; d++) {
                    avgPred[d] = StatisticsUtils.mean(predictions[d]);
                }

                outputRegressionPredictions(
                        new File(outputFolder, SINGLE_AVG + "-" + filename + ".pred"),
                        docIds, trueResponses, avgPred);
                outputRegressionResults(
                        new File(outputFolder, SINGLE_AVG + "-" + filename + ".result"),
                        trueResponses, avgPred);
            }
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
            String[] docIds,
            double[] trueResponses) {
        try {
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

            outputRegressionPredictions(new File(outputFolder, MULTIPLE_FINAL + ".pred"),
                    docIds, trueResponses, predResponses);
            outputRegressionResults(new File(outputFolder, MULTIPLE_FINAL + ".result"),
                    trueResponses, predResponses);
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
    public static double[] computeMultipleAverage(
            File iterPredFolder,
            File outputFolder,
            String[] docIds,
            double[] trueResponses) {
        double[] predResponses = null;
        try {
            String[] filenames = iterPredFolder.list();

            predResponses = new double[trueResponses.length];
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

            outputRegressionPredictions(new File(outputFolder, MULTIPLE_AVG + ".pred"),
                    docIds, trueResponses, predResponses);
            outputRegressionResults(new File(outputFolder, MULTIPLE_AVG + ".result"),
                    trueResponses, predResponses);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating multiple-avg.");
        }
        return predResponses;
    }
}
