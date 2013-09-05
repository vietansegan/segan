/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package core.crossvalidation;

/**
 * Note: groupId can be discarded since it is only used for stratified sampling.
 * Should pass a separate array to the stratified sampling function.
 * @author vietan
 */
public class Instance <I> {
    private final I id;
    
    public Instance(I id){
        this.id = id;
    }
    
    public I getId(){
        return this.id;
    }
}
