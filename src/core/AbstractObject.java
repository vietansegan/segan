/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package core;

import java.util.HashMap;

/**
 *
 * @author vietan
 */
public abstract class AbstractObject {
    protected final String id;
    protected HashMap<String, String> properties;
    
    public AbstractObject(String id){
        this.id = id;
        this.properties = new HashMap<String, String>();
    }
    
    public String getId(){
        return this.id;
    }
    
    public void addProperty(String propName, String propValue){
        if(this.properties.containsKey(propName))
            System.out.println("[WARNING] Adding to existing property"
                    + ". object id = " + id
                    + ". property = " + propName
                    + ". current value = " + properties.get(propName)
                    + ". new value = " + propValue);
        this.properties.put(propName, propValue);
    }
    
    public String getProperty(String propName){
        return this.properties.get(propName);
    }
    
    public boolean hasProperty(String propName){
        return this.properties.containsKey(propName);
    }
}
