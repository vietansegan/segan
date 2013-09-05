/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling.util;

import java.util.HashMap;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class SparseCount implements Cloneable {
    private HashMap<Integer, Integer> counts;
    private int countSum;
    
    public SparseCount(){
        this.counts = new HashMap<Integer, Integer>();
        this.countSum = 0;
    }
    
    @Override
    public SparseCount clone() throws CloneNotSupportedException{
        SparseCount sc = (SparseCount) super.clone();
        sc.counts = (HashMap<Integer, Integer>) this.counts.clone();
        return sc;
    }
    
    public HashMap<Integer, Integer> getObservations(){
        return this.counts;
    }
    
    public void setCount(int observation, int count){
        if(count < 0)
            throw new RuntimeException("Setting a negative count. " + count);
        int curCount = this.getCount(observation);
        this.counts.put(observation, count);
        this.countSum += count - curCount;
    }
    
    public Set<Integer> getUniqueObservations(){
        return this.counts.keySet();
    }
    
    public int getCountSum(){
        return this.countSum;
    }
    
    public int getCount(int observation){
        Integer count = this.counts.get(observation);
        if(count == null)
            return 0;
        else
            return count;
    }
    
    public void changeCount(int observation, int delta){
        int count = getCount(observation);
        this.setCount(observation, count + delta);
    }
    
    public void increment(int observation){
        Integer count = this.counts.get(observation);
        if(count == null)
            this.counts.put(observation, 1);
        else
            this.counts.put(observation, count + 1);
        this.countSum ++;
    }
    
    public void decrement(int observation){
        Integer count = this.counts.get(observation);
        if(count == null){
            for(Integer obs : this.counts.keySet())
                System.out.println(obs + ": " + this.counts.get(obs));
            throw new RuntimeException("Removing observation that does not exist " + observation);
        }
        if(count == 1)
            this.counts.remove(observation);
        else
            this.counts.put(observation, count - 1);
        this.countSum --;
        
        if(counts.get(observation) != null && this.counts.get(observation) < 0)
            throw new RuntimeException("Negative count for observation " + observation
                    + ". count = " + this.counts.get(observation));
        if(countSum < 0)
            throw new RuntimeException("Negative count sumze " + countSum);
    }
    
    public boolean isEmpty(){
        return this.countSum == 0;
    }
    
    @Override
    public String toString(){
        StringBuilder str = new StringBuilder();
        for(int obs : this.getUniqueObservations())
            str.append(obs).append(":").append(getCount(obs)).append(" ");
        return str.toString();
    }
    
    public void validate(String msg){
        if(this.countSum < 0)
            throw new RuntimeException(msg + ". Negative countSum");
        
        int totalCount = 0;
        for(int obs : this.counts.keySet())
            totalCount += this.counts.get(obs);
        if(totalCount != this.countSum)
            throw new RuntimeException(msg + ". Total counts mismatched. " + totalCount + " vs. " + countSum);
    }
    
    public static String output(SparseCount sc){
        StringBuilder str = new StringBuilder();
        for(int obs : sc.counts.keySet())
            str.append(obs).append(":").append(sc.counts.get(obs)).append("\t");
        return str.toString();
    }
    
    public static SparseCount input(String line){
        SparseCount sp = new SparseCount();
        if(!line.isEmpty()){
            String[] sline = line.trim().split("\t");
            for(String obsCount : sline){
                String[] parse = obsCount.split(":");
                int obs = Integer.parseInt(parse[0]);
                int count = Integer.parseInt(parse[1]);
                sp.changeCount(obs, count);
            }
        }
        return sp;
    }
}
