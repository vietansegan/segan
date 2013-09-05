/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class SingleResponseTextDataset extends TextDataset{
//    protected ArrayList<Double> responseList;
    protected double[] responses;
    
    public SingleResponseTextDataset(String name, String folder){
        super(name, folder);
    }
    
    public SingleResponseTextDataset(String name, String folder, 
            CorpusProcessor corpProc){
        super(name, folder, corpProc);
    }
    
    public double[] getResponses(){
        return this.responses;
    }
    
    public void loadResponses(String responseFilepath) throws Exception{
        logln("--- Loading response from file " + responseFilepath);
        
        if(this.docIdList == null)
            throw new RuntimeException("docIdList is null. Load text data first.");
        
        this.responses = new double[this.docIdList.size()];
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(responseFilepath);
        while((line = reader.readLine()) != null){
            String[] sline = line.split("\t");
            String docId = sline[0];
            double docResponse = Double.parseDouble(sline[1]);
            int index = this.docIdList.indexOf(docId);
            this.responses[index] = docResponse;
        }
        reader.close();
    }
    
    @Override
    protected void outputInfo(String outputFolder) throws Exception{
        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFolder + name + docInfoExt);
        for(int docIndex : this.processedDocIndices)
            infoWriter.write(this.docIdList.get(docIndex) 
                    + "\t" + this.responses[docIndex]
                    + "\n");
        infoWriter.close();
    }
    
    @Override
    public void inputDocumentInfo(String filepath) throws Exception{
        logln("--- Reading document info from " + filepath);
        
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<Double> responseList = new ArrayList<Double>();
        
        while((line = reader.readLine()) != null){
            sline = line.split("\t");
            docIdList.add(sline[0]);
            responseList.add(Double.parseDouble(sline[1]));
        }
        reader.close();
        
        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.responses = new double[responseList.size()];
        for(int i=0; i<this.responses.length; i++)
            this.responses[i] = responseList.get(i);
    }
}
