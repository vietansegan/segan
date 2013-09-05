/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling;

import sampling.util.SparseCount;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

/**
 * An abstract likelihood model of generating countable finite observations.
 * Observations are indexed by integers from the vocabulary.
 * 
 * Each likelihood model needs to store the hyperparameters of the corresponding 
 * prior.
 * 
* @author vietan
 */
public abstract class AbstractDiscreteFiniteLikelihoodModel implements Cloneable{
    public static final int RANDOM_SEED = 1123581321;
    
    // this is currently used for likelihood models that does not have/use
    // conjugate prior and we need to sample from the prior
    protected static Random rand = new Random(RANDOM_SEED); 
    
    // observations
    protected final int dimension;
    protected SparseCount sparseCounts;
    
    public AbstractDiscreteFiniteLikelihoodModel(int dim){
        this.dimension = dim;
        this.sparseCounts = new SparseCount();
    }
    
    public abstract String getModelName();
    
    public abstract double getLogLikelihood(int observation);
    
    public abstract double getLogLikelihood();
    
    public abstract double[] getDistribution();
    
    /** Sample the parameter(s) for a new component using the prior. This
     * is mainly used for non-conjugate prior where computing the likelihood of
     * the new table/component given observed data is difficult.
     */
    public abstract void sampleFromPrior();
    
    @Override
    public AbstractDiscreteFiniteLikelihoodModel clone() throws CloneNotSupportedException{
        AbstractDiscreteFiniteLikelihoodModel m = (AbstractDiscreteFiniteLikelihoodModel) super.clone();
        m.sparseCounts = this.sparseCounts.clone();
        return m;
    }
    
    public boolean isEmpty(){
        return this.sparseCounts.isEmpty();
    }
    
    public int getCount(int observation){
        return this.sparseCounts.getCount(observation);
    }
    
    public HashMap<Integer, Integer> getObservations(){
        return this.sparseCounts.getObservations();
    }
    
    public Set<Integer> getUniqueObservations(){
        return this.sparseCounts.getUniqueObservations();
    }

    public int[] getCounts(){
        int[] counts = new int[this.dimension];
        for(int obs : this.sparseCounts.getUniqueObservations()){
            counts[obs] = this.sparseCounts.getCount(obs);
        }
        return counts;
    }
    
    public SparseCount getSparseCounts(){
        return this.sparseCounts;
    }
    
    public void setCounts(int[] c){
        this.sparseCounts = new SparseCount();
        for(int i=0; i<c.length; i++){
            if(c[i] > 0)
                this.sparseCounts.setCount(i, c[i]);
        }
    }

    public int getCountSum(){
        return this.sparseCounts.getCountSum();
    }
    
    public int getDimension(){
        return this.dimension;
    }
    
    /** Change the count of a given observation
     * @param observation The observation in [0, dim)
     * @param delta Change in the count
     */
    public void changeCount(int observation, int delta){
        int count = this.getCount(observation);
        this.sparseCounts.setCount(observation, count + delta);
    }
    
    /** Decrement the count of a given observation
     * @param observation The observation whose count is decremented
     */
    public void decrement(int observation){
        this.sparseCounts.decrement(observation);
    }

    /** Increment the count of a given observation
     * @param observation The observation whose count is incremented
     */
    public void increment(int observation){
        this.sparseCounts.increment(observation);
    }
    
    public void validate(String msg){
        this.sparseCounts.validate(msg);
    }
    
    public String getDebugString(){
        StringBuilder str = new StringBuilder();
        str.append("Dimension = ").append(this.dimension).append("\n");
        str.append("Count sum = ").append(this.getCountSum()).append("\n");
        str.append("Counts = ").append(java.util.Arrays.toString(this.getCounts())).append("\n");
        return str.toString();
    }
}