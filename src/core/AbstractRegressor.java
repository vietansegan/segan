package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public abstract class AbstractRegressor extends AbstractRunner {
    public static final String DATA_FILE = "data";
    public static final String MODEL_FILE = "model";
    public static final String PREDICTION_FILE = "predictions";
    public static final String RESULT_FILE = "result";

    protected String folder;

    public AbstractRegressor(String folder) {
        this.folder = folder;
    }

    public abstract String getName();

    public String getFolder() {
        return this.folder;
    }

    public String getRegressorFolder() {
        return new File(folder, getName()).getAbsolutePath();
    }

    public double[] inputPredictions(File inputFile) throws Exception {
        if (verbose) {
            logln(">>> Input predictions to " + inputFile);
        }
        BufferedReader reader = IOUtils.getBufferedReader(inputFile);
        int numInst = Integer.parseInt(reader.readLine());
        double[] predResponses = new double[numInst];
        for (int i = 0; i < numInst; i++) {
            String[] sline = reader.readLine().split("\t");
            predResponses[i] = Double.parseDouble(sline[3]);
        }
        reader.close();
        return predResponses;
    }

    public void outputPredictions(File outputFile,
            String[] instanceIds,
            double[] trueValues,
            double[] predValues) {
        if (instanceIds.length != trueValues.length || instanceIds.length != predValues.length) {
            throw new RuntimeException("Lengths mismatched");
        }

        if (verbose) {
            logln(">>> Output predictions to " + outputFile);
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

    public ArrayList<Measurement> outputRegressionResults(
            File outputFile,
            double[] trueValues, double[] predValues) {
        // output different measurements
        if (verbose) {
            logln(">>> Output regression results to " + outputFile);
        }
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
            throw new RuntimeException("Exception while outputing regression results to "
                    + outputFile);
        }
        return measurements;
    }

    public ArrayList<Measurement> outputClassificationResults(File outputFile,
            int[] trueClasses, int[] predClasses) throws Exception {
        // output different measurements
        if (verbose) {
            logln(">>> Output classification results to " + outputFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        ClassificationEvaluation eval = new ClassificationEvaluation(trueClasses, predClasses);
        eval.computePRF1();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement m : measurements) {
            writer.write(m.getName() + "\t" + m.getValue() + "\n");
        }
        writer.close();
        return measurements;
    }

    public static void log(String msg) {
        System.out.print("[LOG] " + msg);
    }

    public static void logln(String msg) {
        System.out.println("[LOG] " + msg);
    }
}
