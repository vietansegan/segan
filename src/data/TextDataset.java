/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import util.DataUtils;
import util.IOUtils;

/**
 * A set of documents
 * 
 * @author vietan
 */
public class TextDataset extends AbstractTokenizeDataset{
    protected ArrayList<String> docIdList; // original list of document ids
    protected ArrayList<String> textList; // raw text of documents
    protected ArrayList<Integer> processedDocIndices; // list of document ids after pre-processing
    
    protected ArrayList<String> wordVocab;
    protected String[] docIds;
    protected int[][] words;
    protected int[][][] sentWords;
    
    public TextDataset(
            String name, 
            String folder){
        super(name, folder);
        
        this.docIdList = new ArrayList<String>();
        this.textList = new ArrayList<String>();
        this.processedDocIndices = new ArrayList<Integer>();
    }
    
    public TextDataset(
            String name, 
            String folder, 
            CorpusProcessor corpProc){
        super(name, folder, corpProc);
        
        this.docIdList = new ArrayList<String>();
        this.textList = new ArrayList<String>();
        this.processedDocIndices = new ArrayList<Integer>();
    }
    
    public int[][][] getSentenceWords(){
        return this.sentWords;
    }
    
    public ArrayList<String> getWordVocab(){
        return this.wordVocab;
    }
    
    /** Load text data from a single file where each line has the following format
     * <doc_Id>\t<token_1>\t<token_2> ...
     * 
     * @param textFilepath The input data file
     */
    public void loadTextDataFromFile(String textFilepath) throws Exception{
        logln("--- Loading text data from file " + textFilepath);
        
        BufferedReader reader = IOUtils.getBufferedReader(textFilepath);
        String line;
        while((line = reader.readLine()) != null){
            docIdList.add(line.substring(0, line.indexOf("\t")));
            textList.add(line.substring(line.indexOf("\t")+1));
        }
        reader.close();
        
        logln("--- --- Loaded " + docIdList.size() + " document(s)");
    }
    
    /** Load text data from a folder where each file contains the text of a 
     * document, each filename is in the form of <doc_Id>.txt
     * 
     * @param textFolderPath The input data folder
     */
    public void loadTextDataFromFolder(String textFolderPath) throws Exception{
        logln("--- Loading text data from folder " + textFolderPath);
        
        File fd = new File(textFolderPath);
        String[] filenames = fd.list();
        BufferedReader reader;
        String line;
        StringBuilder docText;
        for(String filename : filenames){
            docIdList.add(IOUtils.removeExtension(filename));
            reader = IOUtils.getBufferedReader(textFolderPath + filename);
            docText = new StringBuilder();
            while((line = reader.readLine()) != null)
                docText.append(line).append("\n");
            reader.close();
            
            textList.add(docText.toString());
        }
        
        logln("--- --- Loaded " + docIdList.size() + " document(s)");
    }
    
    /** Format input data 
     * @param outputFolder The directory of the folder that processed data will
     * be stored
     */
    public void format(String outputFolder) throws Exception{
        logln("--- Processing data ...");
        String[] rawTexts = textList.toArray(new String[textList.size()]);
        corpProc.setRawTexts(rawTexts);
        corpProc.process();
        
        // output the data into format used by samplers
        logln("--- Outputing word vocab ... " + outputFolder + name + wordVocabExt);
        IOUtils.createFolder(outputFolder);
        DataUtils.outputVocab(outputFolder + name + wordVocabExt, corpProc.getVocab());
        
        logln("--- Outputing main numeric data ... " + outputFolder + name + numDocDataExt);
        this.outputTextData(outputFolder);
        
        logln("--- Outputing document info ... " + outputFolder + name + docInfoExt);
        outputInfo(outputFolder);
        
        logln("--- Outputing sentence data ... " + outputFolder + name + numSentDataExt);
        outputSentTextData(outputFolder);
    }
    
    protected void outputTextData(String outputFolder) throws Exception{
        // output main numeric
        int[][] numDocs = corpProc.getNumerics();
        
        BufferedWriter dataWriter = IOUtils.getBufferedWriter(outputFolder + name + numDocDataExt);
        for(int d=0; d<numDocs.length; d++){
            HashMap<Integer, Integer> typeCounts = new HashMap<Integer, Integer>();
            for(int j=0; j<numDocs[d].length; j++){
                Integer count = typeCounts.get(numDocs[d][j]);
                if(count == null)
                    typeCounts.put(numDocs[d][j], 1);
                else
                    typeCounts.put(numDocs[d][j], count + 1);
            }
            
            // skip short documents
            if(typeCounts.size() < corpProc.docTypeCountCutoff)
                continue;
            
            // write main data
            dataWriter.write(Integer.toString(typeCounts.size()));
            for(int type : typeCounts.keySet())
                dataWriter.write(" " + type + ":" + typeCounts.get(type));
            dataWriter.write("\n");
            
            // save the doc id
            this.processedDocIndices.add(d);
        }
        dataWriter.close();
    }
    
    protected void outputSentTextData(String outputFolder) throws Exception{
        int[][][] numSents = corpProc.getNumericSentences();
        BufferedWriter sentWriter = IOUtils.getBufferedWriter(outputFolder + name + numSentDataExt);
        for(int d : this.processedDocIndices){
            StringBuilder docStr = new StringBuilder();
            for(int s=0; s<numSents[d].length; s++){
                HashMap<Integer, Integer> sentTypeCounts = new HashMap<Integer, Integer>();
                for(int w=0; w<numSents[d][s].length; w++){
                    Integer count = sentTypeCounts.get(numSents[d][s][w]);
                    if(count == null)
                        sentTypeCounts.put(numSents[d][s][w], 1);
                    else
                        sentTypeCounts.put(numSents[d][s][w], count + 1);
                }
                
                if(sentTypeCounts.size() > 0){
                    StringBuilder str = new StringBuilder();
                    for(int type : sentTypeCounts.keySet())
                        str.append(type).append(":").append(sentTypeCounts.get(type)).append(" ");
                    docStr.append(str.toString().trim()).append("\t");
                }
            }
            sentWriter.write(docStr.toString().trim() + "\n");
        }
        sentWriter.close();
    }
    
    protected void outputInfo(String outputFolder) throws Exception{
        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFolder + name + docInfoExt);
        for(int docIndex : this.processedDocIndices)
            infoWriter.write(this.docIdList.get(docIndex) + "\n");
        infoWriter.close();
    }
    
    public String[] getDocIds(){
        return docIds;
    }
    
    public int[][] getWords(){
        return this.words;
    }
    
    public void loadFormattedData(){
        String fFolder = this.getFormatPath();
        logln("--- Loading formatted data ...");
        try{
            inputWordVocab(fFolder + name + wordVocabExt);
            inputTextData(fFolder + name + numDocDataExt);
            inputDocumentInfo(fFolder + name + docInfoExt);
            inputSentenceTextData(fFolder + name + numSentDataExt);
        }
        catch(Exception e){
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    protected void inputWordVocab(String filepath) throws Exception{
        wordVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        while((line = reader.readLine()) != null)
            wordVocab.add(line);
        reader.close();
    }
    
    protected void inputTextData(String filepath) throws Exception{
        logln("--- Reading text data from " + filepath);
        
        BufferedReader reader = IOUtils.getBufferedReader(filepath);

        ArrayList<int[]> wordList = new ArrayList<int[]>();
        String line;
        String[] sline;
        while((line = reader.readLine()) != null){
            sline = line.split(" ");
            
            int numTypes = Integer.parseInt(sline[0]);
            int[] types = new int[numTypes];
            int[] counts = new int[numTypes];
            
            int numTokens = 0;
            for (int ii = 0; ii < numTypes; ++ii) {
                String[] entry = sline[ii+1].split(":");
                int count = Integer.parseInt(entry[1]);
                int id = Integer.parseInt(entry[0]);
                numTokens += count;
                types[ii] = id;
                counts[ii] = count;
            }

            int[] gibbsString = new int[numTokens];
            int index = 0;
            for (int ii = 0; ii < numTypes; ++ii) {
                for (int jj = 0; jj < counts[ii]; ++jj) {
                    gibbsString[index++] = types[ii];
                }
            }
            wordList.add(gibbsString);
        }
        words = wordList.toArray(new int[wordList.size()][]);
        reader.close();
    }
    
    protected void inputSentenceTextData(String filepath) throws Exception{
        logln("--- Reading sentence text data from " + filepath);
        
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        ArrayList<int[][]> sentWordList = new ArrayList<int[][]>();
        String line; String[] sline;
        while((line = reader.readLine()) != null){
            sline = line.split("\t");
            int numSents = sline.length;
            int[][] sents = new int[numSents][];
            for(int s=0; s<numSents; s++){    
                String[] sSent = sline[s].split(" ");
                int numTokens = 0;
                HashMap<Integer, Integer> typeCounts = new HashMap<Integer, Integer>();
                
                for(String sSentWord : sSent){
                    int type = Integer.parseInt(sSentWord.split(":")[0]);
                    int count = Integer.parseInt(sSentWord.split(":")[1]);
                    numTokens += count;
                    typeCounts.put(type, count);
                }
                
                int[] tokens = new int[numTokens];
                int idx = 0;
                for(int type : typeCounts.keySet()){
                    for(int ii=0; ii<typeCounts.get(type); ii++)
                        tokens[idx++] = type;
                }
                sents[s] = tokens;
            }
            sentWordList.add(sents);
        }
        reader.close();
        
        sentWords = new int[sentWordList.size()][][];
        for(int i=0; i<sentWords.length; i++)
            sentWords[i] = sentWordList.get(i);
    }
    
    protected void inputDocumentInfo(String filepath) throws Exception{
        logln("--- Reading document info from " + filepath);
        
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        ArrayList<String> dIdList = new ArrayList<String>();
        
        while((line = reader.readLine()) != null){
            dIdList.add(line);
        }
        reader.close();
        
        this.docIds = dIdList.toArray(new String[dIdList.size()]);
    }
}
