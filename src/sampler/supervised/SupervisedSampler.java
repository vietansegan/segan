/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.supervised;

import java.io.File;

/**
 *
 * @author vietan
 */
public interface SupervisedSampler {

    public void trainSampler();

    public void testSampler(int[][] newWords);

    public void outputSampler(File samplerFile);

    public void inputSampler(File samplerFile);
}
