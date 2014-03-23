package util.evaluation;

import java.util.ArrayList;
import java.util.Collections;
import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.OneError;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public class MultilabelClassificationEvaluation {
    
    private boolean[][] trueLabels;
    private double[][] predictedScores;
    private ArrayList<Measurement> measurements;
    private int numLabels;
    
    public MultilabelClassificationEvaluation(boolean[][] truth, double[][] predicts) {
        this.numLabels = predicts[0].length;
        this.trueLabels = truth;
        this.predictedScores = predicts;
        this.measurements = new ArrayList<Measurement>();
    }
    
    public MultilabelClassificationEvaluation(int[][] truth, double[][] predicts) {
        this.numLabels = predicts[0].length;
        this.trueLabels = new boolean[predicts.length][numLabels];
        for (int dd = 0; dd < trueLabels.length; dd++) {
            for (int ii = 0; ii < truth[dd].length; ii++) {
                this.trueLabels[dd][truth[dd][ii]] = true;
            }
        }
        this.predictedScores = predicts;
        this.measurements = new ArrayList<Measurement>();
    }
    
    public ArrayList<Measurement> getMeasurements() {
        return this.measurements;
    }
    
    public void computeMeasurements() {
//        computeTopKMeasures(1);
//        computeTopKMeasures(3);
//        computeTopKMeasures(5);
//        computeTopKMeasures(10);
        computeMeanAveragePrecision();
        computeMicroAUC();
        computeMacroAUC();
        computeIsError();
        computeOneError();
    }

    public void computePRF() {
        int numDocs = 0;
        int totalCorrect = 0;
        int totalCount = 0;
        double sumPrec = 0.0;
        for (int dd = 0; dd < trueLabels.length; dd++) {
            if (trueLabels[dd].length == 0) {
                continue;
            }
            ArrayList<RankingItem<Integer>> rankLabels = new ArrayList<RankingItem<Integer>>();
            for (int ll = 0; ll < predictedScores[dd].length; ll++) {
                rankLabels.add(new RankingItem<Integer>(ll, predictedScores[dd][ll]));
            }
            Collections.sort(rankLabels);

            int numCorrect = 0;
            int k = trueLabels[dd].length;
            for (int ii = 0; ii < k; ii++) {
                int predLabel = rankLabels.get(ii).getObject();
                if (this.trueLabels[dd][predLabel]) {
                    numCorrect++;
                    totalCorrect++; // for micro precision/recall
                }
            }
            totalCount += k;
            numDocs++;
            sumPrec += (double) numCorrect / k;
        }
        
        double microPRF1 = (double) totalCorrect / totalCount;
        double macroPRF1 = sumPrec / numDocs;
        this.measurements.add(new Measurement("Micro-PRF1", microPRF1));
        this.measurements.add(new Measurement("Macro-PRF1", macroPRF1));
    }
    
    public void computeTopKMeasures(int k) {
        int numDocs = 0;
        int totalCorrect = 0;
        double sumPrec = 0.0;
        double sumRec = 0.0;
        int totalTrue = 0;
        for (int dd = 0; dd < trueLabels.length; dd++) {
            if (trueLabels[dd].length == 0) {
                continue;
            }
            ArrayList<RankingItem<Integer>> rankLabels = new ArrayList<RankingItem<Integer>>();
            for (int ll = 0; ll < predictedScores[dd].length; ll++) {
                rankLabels.add(new RankingItem<Integer>(ll, predictedScores[dd][ll]));
            }
            Collections.sort(rankLabels);
            
            int numCorrect = 0;
            for (int ii = 0; ii < k; ii++) {
                int predLabel = rankLabels.get(ii).getObject();
                if (this.trueLabels[dd][predLabel]) {
                    numCorrect++;
                    totalCorrect++; // for micro precision/recall
                }
            }
            numDocs++;
            sumRec += (double) numCorrect / trueLabels[dd].length;
            sumPrec += (double) numCorrect / k;
            totalTrue += trueLabels[dd].length;
        }
        
        double microPrec = (double) totalCorrect / (numDocs * k);
        this.measurements.add(new Measurement("Micro-P@" + k, microPrec));
        double macroPrec = sumPrec / numDocs;
        this.measurements.add(new Measurement("Macro-P@" + k, macroPrec));
        double microRecall = (double) totalCorrect / totalTrue;
        this.measurements.add(new Measurement("Micro-R@" + k, microRecall));
        double macroRecall = (double) sumRec / numDocs;
        this.measurements.add(new Measurement("Macro-R@" + k, macroRecall));
        double microF1 = 2 * microPrec * microRecall / (microPrec + microRecall);
        this.measurements.add(new Measurement("Micro-F1@" + k, microF1));
        double macroF1 = 2 * macroPrec * macroRecall / (macroPrec + macroRecall);
        this.measurements.add(new Measurement("Macro-F1@" + k, macroF1));
    }
    
    public void computeMeanAveragePrecision() {
        MeanAveragePrecision measure = new MeanAveragePrecision(numLabels);
        for (int dd = 0; dd < trueLabels.length; dd++) {
            measure.update(new MultiLabelOutput(predictedScores[dd]), trueLabels[dd]);
        }
        this.measurements.add(new Measurement("MAP", measure.getValue()));
    }
    
    public void computeMicroAUC() {
        MicroAUC microAUC = new MicroAUC(numLabels);
        for (int dd = 0; dd < trueLabels.length; dd++) {
            microAUC.update(new MultiLabelOutput(predictedScores[dd]), trueLabels[dd]);
        }
        this.measurements.add(new Measurement("Micro-AUC", microAUC.getValue()));
    }
    
    public void computeMacroAUC() {
        MacroAUC macroAUC = new MacroAUC(numLabels);
        for (int dd = 0; dd < trueLabels.length; dd++) {
            if (trueLabels[dd].length != 0) {
                macroAUC.update(new MultiLabelOutput(predictedScores[dd]), trueLabels[dd]);
            }
        }
        this.measurements.add(new Measurement("Macro-AUC", macroAUC.getValue()));
    }
    
    public void computeOneError() {
        OneError oneError = new OneError();
        for (int dd = 0; dd < trueLabels.length; dd++) {
            oneError.update(new MultiLabelOutput(predictedScores[dd]), trueLabels[dd]);
        }
        this.measurements.add(new Measurement("One-error", oneError.getValue()));
    }
    
    public void computeIsError() {
        IsError isError = new IsError();
        for (int dd = 0; dd < trueLabels.length; dd++) {
            isError.update(new MultiLabelOutput(predictedScores[dd]), trueLabels[dd]);
        }
        this.measurements.add(new Measurement("Is-error", isError.getValue()));
    }
}
