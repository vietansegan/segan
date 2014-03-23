package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.MultilabelClassificationEvaluation;
import util.evaluation.RankingPerformance;
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

    public static double[] inputPredictedValues(File inputFile) {
        double[] predVals = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            int count = Integer.parseInt(reader.readLine());
            predVals = new double[count];
            for (int ii = 0; ii < count; ii++) {
                predVals[ii] = (Double.parseDouble(reader.readLine().split("\t")[2]));
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictions from "
                    + inputFile);
        }
        return predVals;
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
        System.out.println("Outputing regression results to " + outputFile);
        ArrayList<Measurement> measurements = null;
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            RegressionEvaluation eval = new RegressionEvaluation(trueValues, predValues);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeMeanAbsoluteError();
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

    /**
     * Output ranking performance.
     *
     * @param rankFolder Output ranking folder
     * @param instanceIds Instance IDs
     * @param trueValues The true values
     * @param predValues The predicted values
     */
    public static void outputRankingPerformance(
            File rankFolder,
            String[] instanceIds,
            double[] trueValues,
            double[] predValues) {
        System.out.println("Outputing ranking performance " + rankFolder);
        IOUtils.createFolder(rankFolder);

        // predictions
        RankingItemList<String> preds = new RankingItemList<String>();
        for (int ii = 0; ii < instanceIds.length; ii++) {
            preds.addRankingItem(new RankingItem<String>(instanceIds[ii], predValues[ii]));
        }
        preds.sortDescending();

        // groundtruth
        RankingItemList<String> truths = new RankingItemList<String>();
        for (int ii = 0; ii < instanceIds.length; ii++) {
            truths.addRankingItem(new RankingItem<String>(instanceIds[ii], trueValues[ii]));
        }
        truths.sortDescending();

        RankingPerformance<String> rankPerf = new RankingPerformance<String>(preds,
                rankFolder.getAbsolutePath());
        rankPerf.computeAndOutputNDCGsNormalize(truths);
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
     * Output the predictions of a single model for classification.
     *
     * @param predictions D x L 2D array
     */
    public static void outputSingleModelClassifications(File file,
            double[][] predictions) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(predictions.length + "\n");
            for (int dd = 0; dd < predictions.length; dd++) {
                writer.write(Integer.toString(dd));
                for (int jj = 0; jj < predictions[dd].length; jj++) {
                    writer.write("\t" + predictions[dd][jj]);
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
     * Input the predictions for classification from a file.
     *
     * @param file The prediction file
     */
    public static double[][] inputSingleModelClassifications(File file) {
        double[][] predictions = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            int count = Integer.parseInt(reader.readLine());
            predictions = new double[count][];

            for (int dd = 0; dd < count; dd++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != dd) {
                    throw new RuntimeException("Mismatch");
                }

                predictions[dd] = new double[sline.length - 1];
                for (int ii = 0; ii < predictions[dd].length; ii++) {
                    predictions[dd][ii] = Double.parseDouble(sline[ii + 1]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictions from "
                    + file);
        }
        return predictions;
    }

    public static double[][] evaluateClassifications(
            File iterPredFolder,
            File outputFile,
            int[][] trueLabels) {
        System.out.println("Evaluating predictions in folder " + iterPredFolder
                + "\nAnd outputing to " + outputFile);

        double[][] avgPreds = null;
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            String[] filenames = iterPredFolder.list();
            for (int ff = 0; ff < filenames.length; ff++) {
                double[][] singlePredictions = inputSingleModelClassifications(
                        new File(iterPredFolder, filenames[ff]));
                MultilabelClassificationEvaluation eval = new MultilabelClassificationEvaluation(trueLabels, singlePredictions);
                eval.computeMeasurements(); 

                if (ff == 0) {
                    writer.write("File");
                    for (Measurement m : eval.getMeasurements()) {
                        writer.write("\t" + m.getName());
                    }
                    writer.write("\n");

                    avgPreds = new double[singlePredictions.length][singlePredictions[0].length];
                }
                writer.write(filenames[ff]);
                for (Measurement m : eval.getMeasurements()) {
                    writer.write("\t" + m.getValue());
                }
                writer.write("\n");

                for (int dd = 0; dd < avgPreds.length; dd++) {
                    for (int jj = 0; jj < avgPreds[dd].length; jj++) {
                        avgPreds[dd][jj] += singlePredictions[dd][jj];
                    }
                }
            }
            // average
            for (int dd = 0; dd < avgPreds.length; dd++) {
                for (int jj = 0; jj < avgPreds[dd].length; jj++) {
                    avgPreds[dd][jj] /= filenames.length;
                }
            }
            MultilabelClassificationEvaluation eval = new MultilabelClassificationEvaluation(trueLabels, avgPreds);
            eval.computeMeasurements();
            writer.write("Average");
            for (Measurement m : eval.getMeasurements()) {
                writer.write("\t" + m.getValue());
            }
            writer.write("\n");

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating classification");
        }
        return avgPreds;
    }

    /**
     * Output the predictions of a single model (learned at an iteration during
     * training) on test documents.
     *
     * @param file The output file
     * @param predictions The list of predicted values, each for a test
     * document.
     */
    public static void outputSingleModelRegressions(
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
    public static double[][] inputSingleModelRegressions(File file, int numDocs) {
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
        double[] singleFinalPred = computeSingleFinal(iterPredFolder, outputFolder, docIds, trueResponses);
        double[] singleAvgPred = computeSingleAverage(iterPredFolder, outputFolder, docIds, trueResponses);
        double[] multipleFinalPred = computeMultipleFinal(iterPredFolder, outputFolder, docIds, trueResponses);
        double[] multipleAvgPred = computeMultipleAverage(iterPredFolder, outputFolder, docIds, trueResponses);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(outputFolder, "summary.txt"));
            ArrayList<Measurement> sfRe = evaluateRegression(trueResponses, singleFinalPred);
            ArrayList<Measurement> saRe = evaluateRegression(trueResponses, singleAvgPred);
            ArrayList<Measurement> mfRe = evaluateRegression(trueResponses, multipleFinalPred);
            ArrayList<Measurement> maRe = evaluateRegression(trueResponses, multipleAvgPred);

            // headers
            for (Measurement m : sfRe) {
                writer.write("\t" + m.getName());
            }
            writer.write("\n");

            // contents
            writer.write("single-final");
            for (Measurement m : sfRe) {
                writer.write("\t" + m.getValue());
            }
            writer.write("\n");

            writer.write("single-avg");
            for (Measurement m : saRe) {
                writer.write("\t" + m.getValue());
            }
            writer.write("\n");

            writer.write("multiple-final");
            for (Measurement m : mfRe) {
                writer.write("\t" + m.getValue());
            }
            writer.write("\n");

            writer.write("multiple-avg");
            for (Measurement m : maRe) {
                writer.write("\t" + m.getValue());
            }
            writer.write("\n");

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating regression "
                    + outputFolder);
        }

        return multipleAvgPred;
    }

    private static ArrayList<Measurement> evaluateRegression(double[] trueValues, double[] predValues) {
        RegressionEvaluation eval = new RegressionEvaluation(trueValues, predValues);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeMeanAbsoluteError();
        eval.computeRSquared();
        eval.computePredictiveRSquared();
        return eval.getMeasurements();
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

                double[][] predictions = inputSingleModelRegressions(
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
    public static double[] computeSingleFinal(
            File iterPredFolder,
            File outputFolder,
            String[] docIds,
            double[] trueResponses) {
        double[] finalPred = null;
        try {
            String[] filenames = iterPredFolder.list();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                double[][] predictions = inputSingleModelRegressions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                // get the predictions at the final iterations during test time
                finalPred = new double[predictions.length];
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
        return finalPred;
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
    public static double[] computeSingleAverage(
            File iterPredFolder,
            File outputFolder,
            String[] docIds,
            double[] trueResponses) {
        double[] avgPred = null;
        try {
            String[] filenames = iterPredFolder.list();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];

                // load predicted values
                double[][] predictions = inputSingleModelRegressions(
                        new File(iterPredFolder, filename),
                        trueResponses.length);

                // compute the prediction values as the average values
                avgPred = new double[predictions.length];
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
        return avgPred;
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
    public static double[] computeMultipleFinal(
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

                double[][] predictions = inputSingleModelRegressions(
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
        return predResponses;
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

                double[][] predictions = inputSingleModelRegressions(
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
