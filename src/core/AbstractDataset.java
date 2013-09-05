/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

/**
 *
 * @author vietan
 */
public abstract class AbstractDataset {
    protected final String name;

    public AbstractDataset(String name){
        this.name = name;
    }

    public String getName(){
        return this.name;
    }
    
    public String getFolder(){
        return this.name + "/";
    }
    
    public void logln(String msg){
        System.out.println("[LOG] " + msg);
    }
}
