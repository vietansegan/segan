/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package svm;

/**
 *
 * @author vanguyen
 */
import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import util.IOUtils;
import util.RankingItem;
import util.RankingItemList;
import util.evaluation.RankingPerformance;

public class SVMLight {

    public static final String Svm2WeightFile = "lib/svm2weight.pl";
    // Inputs
    private String folderPath;
    private String svmLightLearn;
    private String svmLightClassify;
    // Outputs
    private RankingPerformance<Integer> performance;

    public SVMLight(String folder) {
        this.folderPath = folder;
        this.svmLightLearn = "lib/svm_light_windows/svm_learn.exe";
        this.svmLightClassify = "lib/svm_light_windows/svm_classify.exe";

        IOUtils.createFolder(this.folderPath);
    }

    public SVMLight(String folder, String svmLightLearn, String svmLightClassify) {
        this.folderPath = folder;
        this.svmLightLearn = svmLightLearn;
        this.svmLightClassify = svmLightClassify;

        IOUtils.createFolder(this.folderPath);
    }

    public void learn(String[] options,
            String trainingFilePath, String modelFile) throws Exception {
        System.out.println("\nStart learning ...");
        String cmd = svmLightLearn;
        if (options != null) {
            for (int i = 0; i < options.length; i++) {
                cmd += " " + options[i];
            }
        }

        cmd += " " + trainingFilePath + " " + folderPath + modelFile;
        cmd = cmd.replace("/", "\\"); // for Windows
        System.out.println("Learn cmd: " + cmd);
        Process proc = Runtime.getRuntime().exec(cmd);
        InputStream istr = proc.getInputStream();
        BufferedReader in = new BufferedReader(new InputStreamReader(istr));
        String line;
        while ((line = in.readLine()) != null) {
            System.out.println(line);
        }
    }

    public void classify(String[] options,
            String testingFilePath,
            String modelFile, String resultFile) throws Exception {
        System.out.println("Start classifying ...");

        String cmd = svmLightClassify;
        if (options != null) {
            for (int i = 0; i < options.length; i++) {
                cmd += " " + options[i];
            }
        }
        cmd += " " + testingFilePath + " " + (folderPath + modelFile) + " " + (folderPath + resultFile);
        cmd = cmd.replace("/", "\\");
        System.out.println("Classify cmd: " + cmd);
        Process proc = Runtime.getRuntime().exec(cmd);

        InputStream istr = proc.getInputStream();
        BufferedReader in = new BufferedReader(new InputStreamReader(istr));
        String line;
        while ((line = in.readLine()) != null) {
            System.out.println(line);
        }
    }

    /**
     * Evaluate SVM-based ranker (using SVMRank)
     */
    public void evaluateRanker(String testingFilePath, String resultFile) throws Exception {
        // load ranked result
        RankingItemList<Integer> resultRankingItemList = new RankingItemList<Integer>();
        BufferedReader reader = IOUtils.getBufferedReader(this.folderPath + resultFile);
        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            double score = Double.parseDouble(line);
            resultRankingItemList.addRankingItem(new RankingItem<Integer>(count, score));
            count++;
        }
        reader.close();
        resultRankingItemList.sortDescending();

        // load groundtruth from testingFilepath
        RankingItemList<Integer> groundtruthRankingItemList = new RankingItemList<Integer>();
        reader = IOUtils.getBufferedReader(testingFilePath);
        count = 0;
        while ((line = reader.readLine()) != null) {
            double score = Double.parseDouble(line.split(" ")[0]);
            groundtruthRankingItemList.addRankingItem(new RankingItem<Integer>(count, score));
            count++;
        }
        reader.close();

        performance = new RankingPerformance<Integer>(resultRankingItemList, this.folderPath);
        performance.computeAndOutputNDCGs(groundtruthRankingItemList);
    }

    /**
     * Evaluate SVM-based classifier (using SVMLight)
     */
    public void evaluateClassifier(String testingFilePath, String resultFile) throws Exception {
        // Load actual classes
        Set<Integer> positiveSet = new HashSet<Integer>();
        BufferedReader testIn = IOUtils.getBufferedReader(testingFilePath);
        String line;
        int count = 0;
        while ((line = testIn.readLine()) != null) {
            String[] sline = line.split(" ");
            int cls = Integer.parseInt(sline[0]);
            if (cls == 1) {
                positiveSet.add(count);
            }
            count++;
        }
        testIn.close();
        System.out.println("# positive set: " + positiveSet.size());

        // Load predicted scores
        RankingItemList<Integer> rankingItemList = new RankingItemList<Integer>();

        BufferedReader resultIn = IOUtils.getBufferedReader(this.folderPath + resultFile);
        count = 0;
        while ((line = resultIn.readLine()) != null) {
            double score = Double.parseDouble(line);
            rankingItemList.addRankingItem(new RankingItem<Integer>(count, score));
            count++;
        }
        resultIn.close();
        rankingItemList.sortDescending();
        System.out.println("# total data points: " + rankingItemList.size());

        performance = new RankingPerformance<Integer>(rankingItemList, positiveSet, this.folderPath);
        performance.outputRankingResultsWithGroundtruth();
        performance.computePrecisionsAndRecalls();
        performance.outputPrecisionRecallF1();
        performance.outputAUCListFile();
        performance.computeAUC();
        performance.outputAUC();
    }

    //String testFilePath, String resultFilePath, String outputFilePath
    public int[][] confusionMatrix(String testingFilePath, String resultFile, String confusionMatrixFile) throws Exception {
        int[][] conMatrix = new int[2][2];
        BufferedReader testIn = IOUtils.getBufferedReader(testingFilePath);
        BufferedReader resultIn = IOUtils.getBufferedReader(this.folderPath + resultFile);

        ArrayList<Integer> test = new ArrayList<Integer>();
        ArrayList<Integer> result = new ArrayList<Integer>();

        String line;
        while ((line = testIn.readLine()) != null) {
            String[] sline = line.split(" ");
            test.add(Integer.valueOf(sline[0]));
        }

        while ((line = resultIn.readLine()) != null) {
            double res = Double.parseDouble(line);
            if (res > 0) {
                result.add(Integer.valueOf(1));
            } else {
                result.add(Integer.valueOf(-1));
            }
        }

        for (int i = 0; i < test.size(); i++) {
            int actual = test.get(i);
            int predict = result.get(i);
            if (actual == 1 && predict == 1) {
                conMatrix[0][0]++;
            } else if (actual == 1 && predict == -1) {
                conMatrix[1][0]++;
            } else if (actual == -1 && predict == 1) {
                conMatrix[0][1]++;
            } else if (actual == -1 && predict == -1) {
                conMatrix[1][1]++;
            } else {
                System.out.println("Error");
            }
        }

        testIn.close();
        resultIn.close();

        for (int i = 0; i < conMatrix.length; i++) {
            for (int j = 0; j < conMatrix[0].length; j++) {
                System.out.print(conMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        // output
        BufferedWriter writer = IOUtils.getBufferedWriter(this.folderPath + confusionMatrixFile);
        writer.write(conMatrix[0][0] + "\t" + conMatrix[0][1] + "\n");
        writer.write(conMatrix[1][0] + "\t" + conMatrix[1][1] + "\n");
        writer.close();

        return conMatrix;
    }

    public double[] getFeatureWeights(String modelFile, String perlSvm2Weight, String featureWeightsFile) throws Exception {
        File aFile = new File(this.folderPath + modelFile);

        String cmd = "perl " + perlSvm2Weight + " " + aFile.getAbsolutePath();
        System.out.println("Processing command " + cmd);
        Process proc = Runtime.getRuntime().exec(cmd);
        InputStream istr = proc.getInputStream();
        BufferedReader in = new BufferedReader(new InputStreamReader(istr));
        String line;
        ArrayList<String> w = new ArrayList<String>();
        while ((line = in.readLine()) != null) {
//            System.out.println(line);
            String[] sline = line.split(":");
            if (sline.length > 1) {
                w.add(sline[1]);
            } else {
                w.add("0.0");
            }
        }
        double[] weights = new double[w.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Double.parseDouble(w.get(i));
        }

        //output
        BufferedWriter writer = IOUtils.getBufferedWriter(this.folderPath + featureWeightsFile);
        for (int i = 0; i < weights.length; i++) {
            writer.write(weights[i] + "\n");
        }
        writer.close();

        return weights;
    }

    public double[] loadFeatureWeights(String featureWeightsFile) throws Exception {
        ArrayList<Double> weights = new ArrayList<Double>();
        BufferedReader reader = IOUtils.getBufferedReader(this.folderPath + featureWeightsFile);
        String line;
        while ((line = reader.readLine()) != null) {
            weights.add(Double.parseDouble(line));
        }
        reader.close();
        double[] featureWeights = new double[weights.size()];
        for (int i = 0; i < featureWeights.length; i++) {
            featureWeights[i] = weights.get(i);
        }
        return featureWeights;
    }

    public void loadClassifierPerformanceMeasure() {
        this.performance = new RankingPerformance<Integer>(this.folderPath);
        this.performance.loadAUCAndF1();
    }

    public void loadRankerPerformanceMeasure() {
        this.performance = new RankingPerformance<Integer>(this.folderPath);
        this.performance.inputNDCGs();
    }

    public void print() {
        System.out.println("In print");
    }

    public String getFolderPath() {
        return this.folderPath;
    }

    public RankingPerformance<Integer> getPerformanceMeasure() {
        return this.performance;
    }
}
