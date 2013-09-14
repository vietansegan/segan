/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.normalizer;

import util.StatisticsUtils;

/**
 *
 * @author vietan
 */
public class MinMaxNormalizer extends AbstractNormalizer {

    private double oldMin;
    private double oldMax;
    private double newMin;
    private double newMax;
    private double oldDiff;
    private double newDiff;

    public MinMaxNormalizer(double[] data, double newMin, double newMax) {
        this.oldMin = StatisticsUtils.min(data);
        this.oldMax = StatisticsUtils.max(data);
        this.newMin = newMin;
        this.newMax = newMax;

        this.oldDiff = this.oldMax - this.oldMin;
        this.newDiff = this.newMax - this.newMin;
    }

    public MinMaxNormalizer(double min, double max, double newMin, double newMax) {
        this.oldMin = min;
        this.oldMax = max;
        this.newMin = newMin;
        this.newMax = newMax;

        this.oldDiff = this.oldMax - this.oldMin;
        this.newDiff = this.newMax - this.newMin;
    }

    @Override
    public double normalize(double oriValue) {
        return (oriValue - oldMin) * newDiff / oldDiff + newMin;
    }

    @Override
    public double denormalize(double normValue) {
        return (normValue - newMin) * oldDiff / newDiff + oldMin;
    }

    public double[] normalize(double[] oriValues) {
        double[] normValues = new double[oriValues.length];
        for (int i = 0; i < normValues.length; i++) {
            normValues[i] = this.normalize(oriValues[i]);
        }
        return normValues;
    }
}
