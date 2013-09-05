/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.evaluation;

import java.util.ArrayList;
import util.StatisticsUtils;

/**
 *
 * @author vietan
 */
public class RegressionEvaluation {
    private double[] trueValues;
    private double[] predValues;
    
    private ArrayList<Measurement> measurements;
    
    public RegressionEvaluation(double[] tv, double[] pv){
        this.trueValues = tv;
        this.predValues = pv;
        this.measurements = new ArrayList<Measurement>();
    }
    
    public ArrayList<Measurement> getMeasurements(){
        return this.measurements;
    }
    
    public void computePredictiveRSquared(){
        this.measurements.add(new Measurement("pR-squared", StatisticsUtils.computePredictedRSquared(trueValues, predValues)));
    }
    
    public void computeMeanSquareError(){
        this.measurements.add(new Measurement("MSE", StatisticsUtils.computeMeanSquaredError(trueValues, predValues)));
    }
    
    public void computeCorrelationCoefficient(){
        this.measurements.add(new Measurement("Correlation-coefficient", StatisticsUtils.computeCorrelationCoefficient(trueValues, predValues)));
    }
    
    public void computeRSquared(){
        this.measurements.add(new Measurement("R-squared", Math.pow(StatisticsUtils.computeCorrelationCoefficient(trueValues, predValues), 2)));
    }
}
