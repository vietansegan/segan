/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.normalizer;

import util.MiscUtils;
import util.StatisticsUtils;

/**
 *
 * @author vietan
 */
public class ZNormalizer extends AbstractNormalizer {
    private double mean;
    private double stdev;
    
    public ZNormalizer(double[] data){
        this.mean = StatisticsUtils.mean(data);
        this.stdev = StatisticsUtils.standardDeviation(data);
    }
    
    public ZNormalizer(double mean, double stdev){
        this.mean = mean;
        this.stdev = stdev;
    }
    
    @Override
    public double normalize(double originalValue){
        return (originalValue - mean) / stdev;
    }
    
    @Override
    public double denormalize(double normalizedValue){
        return normalizedValue * stdev + mean;
    }
    
    public double[] normalize(double[] originalValues){
        double[] normValues = new double[originalValues.length];
        for(int i=0; i<normValues.length; i++)
            normValues[i] = this.normalize(originalValues[i]);
        return normValues;
    }
    
    public double[] denormalize(double[] normalizedValues){
        double[] denormValues = new double[normalizedValues.length];
        for(int i=0; i<denormValues.length; i++)
            denormValues[i] = this.denormalize(normalizedValues[i]);
        return denormValues;
    }
    
    public static void main(String[] args){
        double[] data = {2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34};
        ZNormalizer n = new ZNormalizer(data);
        System.out.println("mean = " + n.mean);
        System.out.println("stdev = " + n.stdev);
        double[] normData = new double[data.length];
        for(int i=0; i<data.length; i++){
            normData[i] = n.normalize(data[i]);
            System.out.println(normData[i]);
        }
        System.out.println(MiscUtils.arrayToString(normData));
        System.out.println(StatisticsUtils.mean(normData));
        System.out.println(StatisticsUtils.standardDeviation(normData));
    }
}
