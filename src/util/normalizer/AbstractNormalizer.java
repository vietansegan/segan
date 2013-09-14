/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.normalizer;

/**
 *
 * @author vietan
 */
public abstract class AbstractNormalizer {

    public abstract double normalize(double originalValue);

    public abstract double denormalize(double normalizedValue);
}
