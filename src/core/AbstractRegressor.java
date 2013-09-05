/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package core;

import core.crossvalidation.CrossValidation;
import core.crossvalidation.Instance;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import util.IOUtils;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public abstract class AbstractRegressor<I, T extends Instance<I>> {
    public static enum CVPHASE {tr, de, te};
    
    public static final String TrainingPredictionFile = "pred.tr";
    public static final String DevelopmentPredictionFile = "pred.de";
    public static final String TestPredictionFile = "pred.te";
    
    public static final String TrainingResultFile = "result.tr";
    public static final String DevelopmentResultFile = "result.de";
    public static final String TestResultFile = "result.te";
    
    protected String folder;
    protected CrossValidation<I, T> crossValidation;
    protected boolean train = false;;
    protected boolean develop = false;
    protected boolean test = false;
    
    public AbstractRegressor(String folder, CrossValidation<I, T> cv){
        this.folder = folder;
        this.crossValidation = cv;
    }
    
    public abstract void regress() throws Exception;
    
    public String getFolder(){
        return this.folder;
    }
    
    public CrossValidation<I, T> getCrossValidation(){
        return this.crossValidation;
    }
    
    public double[] inputPredictions(String inputFolder, CVPHASE phase) throws Exception{
        String inputFilepath = inputFolder;
        if(phase == CVPHASE.tr)
            inputFilepath += TrainingPredictionFile;
        else if(phase == CVPHASE.de)
            inputFilepath += DevelopmentPredictionFile;
        else if(phase == CVPHASE.te)
            inputFilepath += TestPredictionFile;
        else
            throw new RuntimeException("Unknown phase");
        
        logln(">>> Input predictions to " + inputFilepath);
        BufferedReader reader = IOUtils.getBufferedReader(inputFilepath);
        int numInst = Integer.parseInt(reader.readLine());
//        double[] trueResponses = new double[numInst];
        double[] predResponses = new double[numInst];
        for(int i=0; i<numInst; i++){
            String[] sline = reader.readLine().split("\t");
//            trueResponses[i] = Double.parseDouble(sline[2]);
            predResponses[i] = Double.parseDouble(sline[3]);
        }
        reader.close();
        return predResponses;
    }
    
    public void outputPredictions(String outputFolder, CVPHASE phase, ArrayList<Integer> instanceIndices, 
            double[] trueValues, double[] predValues) throws Exception{
        if(instanceIndices.size() != trueValues.length || instanceIndices.size() != predValues.length)
            throw new RuntimeException("Lengths mismatched");
        
        String outputFilepath = outputFolder;
        if(phase == CVPHASE.tr)
            outputFilepath += TrainingPredictionFile;
        else if(phase == CVPHASE.de)
            outputFilepath += DevelopmentPredictionFile;
        else if(phase == CVPHASE.te)
            outputFilepath += TestPredictionFile;
        else
            throw new RuntimeException("Unknown phase");
        
        logln(">>> Output predictions to " + outputFilepath);
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFilepath);
        writer.write(instanceIndices.size() + "\n");
        for(int i=0; i<instanceIndices.size(); i++)
            writer.write(instanceIndices.get(i)
                    + "\t" + crossValidation.getInstance(instanceIndices.get(i)).getId()
                    + "\t" + trueValues[i]
                    + "\t" + predValues[i]
                    + "\n"
                    );
        writer.close();
    }
    
    public ArrayList<Measurement> outputRegressionResults(String outputFolder, CVPHASE phase, double[] trueValues, double[] predValues) throws Exception{
        String outputFilepath = outputFolder;
        if(phase == CVPHASE.tr)
            outputFilepath += TrainingResultFile;
        else if(phase == CVPHASE.de)
            outputFilepath += DevelopmentResultFile;
        else if(phase == CVPHASE.te)
            outputFilepath += TestResultFile;
        else
            throw new RuntimeException("Unknown phase");
        
        // output different measurements
        logln(">>> Output regression results to " + outputFilepath);
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFilepath);
        RegressionEvaluation eval = new RegressionEvaluation(trueValues, predValues);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeRSquared();
        eval.computePredictiveRSquared();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for(Measurement m : measurements)
            writer.write(m.getName() + "\t" + m.getValue() + "\n");
        writer.close();
        return measurements;
    }
    
    public ArrayList<Measurement> outputClassificationResults(String outputFolder, CVPHASE phase, int[] trueClasses, int[] predClasses) throws Exception{
        String outputFilepath = outputFolder + "classify-";
        if(phase == CVPHASE.tr)
            outputFilepath += TrainingResultFile;
        else if(phase == CVPHASE.de)
            outputFilepath += DevelopmentResultFile;
        else if(phase == CVPHASE.te)
            outputFilepath += TestResultFile;
        else
            throw new RuntimeException("Unknown phase");
        
        // output different measurements
        logln(">>> Output classification results to " + outputFilepath);
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFilepath);
        ClassificationEvaluation eval = new ClassificationEvaluation(trueClasses, predClasses);
        eval.computePRF1();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for(Measurement m : measurements)
            writer.write(m.getName() + "\t" + m.getValue() + "\n");
        writer.close();
        return measurements;
    }

    public void setTrain(boolean train) {
        this.train = train;
    }

    public void setDevelop(boolean develop) {
        this.develop = develop;
    }

    public void setTest(boolean test) {
        this.test = test;
    }
    
    public static void log(String msg){
        System.out.print("[LOG] " + msg);
    }

    public static void logln(String msg){
        System.out.println("[LOG] " + msg);
    }
}
