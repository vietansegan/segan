/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.data;

import java.util.HashMap;

/**
 *
 * @author vietan
 */
public class Product {
    public static final String TYPE = "type";
    public static final String BRAND = "brand";
    
    private final String id;
    private final String name;
    private HashMap<String, String> features;
    
    public Product(String id, String name){
        this.id = id;
        this.name = name;
        this.features = new HashMap<String, String>();
    }
    
    public String getId(){
        return this.id;
    }
    
    public String getName(){
        return this.name;
    }
    
    public void setFeatureValue(String feature, String value){
        this.features.put(feature, value);
    }

    public String getFeatureValue(String feature){
        return this.features.get(feature);
    }
    
    @Override
    public String toString(){
        return this.id + "\t" + this.name;
    }
}
