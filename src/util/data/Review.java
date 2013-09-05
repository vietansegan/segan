/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util.data;

import java.util.Hashtable;

/**
 *
 * @author vietan
 */
public class Review {
    public static final String UNKNOWN = "unknown";

    private String id;
    private String text;
    private String reviewer;
    private double rating;

    private Hashtable<String, String> features;
    private int numHelpfuls;
    private int numFeedbacks;

    public Review(String id){
        this(id, new String(), UNKNOWN);
    }
    
    public Review(String id, String text){
        this.id = id;
        this.text = text;
        this.reviewer = UNKNOWN;
        this.features = new Hashtable<String, String>();
    }

    public Review(String id, String text, String reviewer){
        this.id = id;
        this.text = text;
        this.reviewer = reviewer;
        if(this.reviewer.equals(""))
            this.reviewer = UNKNOWN;
        this.features = new Hashtable<String, String>();
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getReviewer() {
        return reviewer;
    }

    public void setReviewer(String reviewer) {
        this.reviewer = reviewer;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public double getRating(){
        return this.rating;
    }

    public void setRating(double r){
        this.rating = r;
    }

    public int getNumFeedbacks() {
        return numFeedbacks;
    }

    public void setNumFeedbacks(int numFeedbacks) {
        this.numFeedbacks = numFeedbacks;
    }

    public int getNumHelpfuls() {
        return numHelpfuls;
    }

    public void setNumHelpfuls(int numHelpfuls) {
        this.numHelpfuls = numHelpfuls;
    }

    public void setFeatureValue(String feature, String value){
        this.features.put(feature, value);
    }

    public String getFeatureValue(String feature){
        return this.features.get(feature);
    }
}
