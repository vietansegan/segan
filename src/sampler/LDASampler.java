/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler;

import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampling.likelihood.DirichletMultinomialModel;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class LDASampler extends AbstractSampler{
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    
    protected int K;
    protected int V; // vocabulary size
    protected int D; // number of documents
    
    protected int[][] words;  // [D] x [Nd]: words
    
    protected int[][] z;
    
    protected DirichletMultinomialModel[] doc_topics;
    protected DirichletMultinomialModel[] topic_words;
    
    private int numTokens;
    private int numTokensChanged;
    
    public void configure(String folder, int[][] words,
            int V, int K,
            double alpha,
            double beta, 
            AbstractSampler.InitialState initState, 
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt){
        this.folder = folder;
        this.words = words;
                
        this.K = K;
        this.V = V;
        this.D = this.words.length;
        
        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();
    }
    
    protected void setName(){
        this.name = this.prefix
                + "_LDA"
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(this.hyperparams.get(ALPHA))
                + "_b-" + formatter.format(this.hyperparams.get(BETA))
                + "_opt-" + this.paramOptimized;
    }
    
    public void configureNewDocuments(int[][] ws){
        this.words = ws;
        this.D = this.words.length;
        
        doc_topics = new DirichletMultinomialModel[D];
        for(int d=0; d<D; d++)
            doc_topics[d] = new DirichletMultinomialModel(K, hyperparams.get(ALPHA)*K, 1.0/K);
        
        // initialize
        z = new int[D][];
        for(int d=0; d<D; d++){
            z[d] = new int[words[d].length];
            for(int n=0; n<words[d].length; n++){
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
            }
        }
    }
    
    public void sampleNewDocuments(int[][] ws){
        if(verbose)
            logln("Sampling for new documents ...");
        
        configureNewDocuments(ws);
        
        // sample
        logLikelihoods = new ArrayList<Double>();
        for(iter=0; iter<MAX_ITER; iter++){
            numTokensChanged = 0;
            
            for(int d=0; d<D; d++){
                for(int t=0; t<words[d].length; t++){
                    sampleZ(d, t, !REMOVE, !ADD);
                }
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % REP_INTERVAL == 0){
                if(iter < BURN_IN)
                    logln("--- Burning in. Iter " + iter 
                            + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                            + ". numTokensChanged = " + numTokensChanged
                            + ". change ratio = " + MiscUtils.formatDouble((double)numTokensChanged / numTokens));
                else
                    logln("--- Sampling. Iter " + iter 
                            + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                            + ". numTokensChanged = " + numTokensChanged
                            + ". change ratio = " + MiscUtils.formatDouble((double)numTokensChanged / numTokens));
            }
        }
    }
    
    @Override
    public void initialize(){
        if(verbose)
            logln("Initializing ...");

        initializeHierarchies();

        initializeAssignments();

        if(debug)
            validate("Initialized");
    }
    
    protected void initializeHierarchies() {
        if(verbose)
            logln("--- Initializing topic hierarchy ...");
     
        numTokens = 0;
        doc_topics = new DirichletMultinomialModel[D];
        for(int d=0; d<D; d++){
            doc_topics[d] = new DirichletMultinomialModel(K, hyperparams.get(ALPHA)*K, 1.0/K);
            numTokens += words[d].length;
        }
            
        topic_words = new DirichletMultinomialModel[K];
        for(int k=0; k<K; k++)
            topic_words[k] = new DirichletMultinomialModel(V, hyperparams.get(BETA)*V, 1.0/V);
        
        z = new int[D][];
        for(int d=0; d<D; d++)
            z[d] = new int[words[d].length];
    }
    
    protected void initializeAssignments() {
        if(verbose)
            logln("--- Initializing assignments ...");
     
        for(int d=0; d<D; d++){
            for(int n=0; n<words[d].length; n++){
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
                topic_words[z[d][n]].increment(words[d][n]);
            }
        }
    }
    
    @Override
    public void iterate(){
        if(verbose)
            logln("Iterating ...");
        logLikelihoods = new ArrayList<Double>();

        for(iter=0; iter<MAX_ITER; iter++){
            numTokensChanged = 0;
            
            for(int d=0; d<D; d++){
                for(int t=0; t<words[d].length; t++){
                    sampleZ(d, t, REMOVE, ADD);
                }
            }

            if(debug)
                validate("Iter " + iter);

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            
            if (verbose && iter % REP_INTERVAL == 0){
                if(iter < BURN_IN)
                    logln("--- Burning in. Iter " + iter 
                            + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                            + ". numTokensChanged = " + numTokensChanged
                            + ". change ratio = " + MiscUtils.formatDouble((double)numTokensChanged / numTokens));
                else
                    logln("--- Sampling. Iter " + iter 
                            + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                            + ". numTokensChanged = " + numTokensChanged
                            + ". change ratio = " + MiscUtils.formatDouble((double)numTokensChanged / numTokens));
            }
        }
    }
    
    /** Sample the topic assignment for each token
     * @param d The document index
     * @param n The token index
     * @param remove Whether this token should be removed from the current assigned topic
     * @param add Whether this token should be added to the sampled topic
     */
    private void sampleZ(int d, int n, boolean remove, boolean add){
        doc_topics[d].decrement(z[d][n]);
        if(remove){
            topic_words[z[d][n]].decrement(words[d][n]);
        }
        
        double[] logprobs = new double[K];
        for(int k=0; k<K; k++){
            logprobs[k] = 
                    doc_topics[d].getLogLikelihood(k) + 
                    topic_words[k].getLogLikelihood(words[d][n]);
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
        if(sampledZ != z[d][n])
            numTokensChanged ++;
        z[d][n] = sampledZ;
        
        doc_topics[d].increment(z[d][n]);
        if(add){
            topic_words[z[d][n]].increment(words[d][n]);
        }
    }
    
    public int[][] getZ(){
        return this.z;
    }
    
    @Override
    public String getCurrentState(){
        StringBuilder str = new StringBuilder();
        
        return str.toString();
    }
    
    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for(int d=0; d<D; d++)
            docTopicLlh += doc_topics[d].getLogLikelihood();
        double topicWordLlh = 0;
        for(int k=0; k<K; k++)
            topicWordLlh += topic_words[k].getLogLikelihood();
        return docTopicLlh + topicWordLlh;
    }
    
    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if(newParams.size() != this.hyperparams.size())
            throw new RuntimeException("Number of hyperparameters mismatched");
        double llh = 0;
        for(int d=0; d<D; d++)
            llh += doc_topics[d].getLogLikelihood(newParams.get(ALPHA)*K, 1.0/K);
        for(int k=0; k<K; k++)
            llh += topic_words[k].getLogLikelihood(newParams.get(BETA)*V, 1.0/V);
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams){
        this.hyperparams = newParams;
        for(int d=0; d<D; d++)
            this.doc_topics[d].setConcentration(this.hyperparams.get(ALPHA)*K);
        for(int k=0; k<K; k++)
            this.topic_words[k].setConcentration(this.hyperparams.get(BETA)*V);
    }
    
    public void outputTopicTopWords(String filepath, int numTopWords) throws Exception{
        if(this.wordVocab == null)
            throw new RuntimeException("The word vocab has not been assigned yet");
        
        if(verbose)
            System.out.println("Outputing topics to file " + filepath);
        
        double[][] distrs = new double[K][];
        for(int k=0; k<K; k++)
            distrs[k] = topic_words[k].getDistribution();
        IOUtils.outputTopWords(distrs, wordVocab, numTopWords, filepath);
    }

    public void outputTopicTopWordsCummProbs(String filepath, int numTopWords) throws Exception{
        if(this.wordVocab == null)
            throw new RuntimeException("The word vocab has not been assigned yet");
        
        double[][] distrs = new double[K][];
        for(int k=0; k<K; k++)
            distrs[k] = topic_words[k].getDistribution();
        IOUtils.outputTopWordsCummProbs(distrs, wordVocab, numTopWords, filepath);
    }
    
    public void outputTopicWordDistribution(String outputFile) throws Exception{
        double[][] pi = new double[K][];
        for(int k=0; k<K; k++)
            pi[k] = this.topic_words[k].getDistribution();
        IOUtils.outputDistributions(pi, outputFile);
    }
    
    public double[][] inputTopicWordDistribution(String inputFile) throws Exception{
        return IOUtils.inputDistributions(inputFile);
    }
    
    public void outputDocumentTopicDistribution(String outputFile) throws Exception{
        double[][] theta = new double[D][];
        for(int d=0; d<D; d++)
            theta[d] = this.doc_topics[d].getDistribution();
        IOUtils.outputDistributions(theta, outputFile);
    }
    
    public double[][] inputDocumentTopicDistribution(String inputFile) throws Exception{
        return IOUtils.inputDistributions(inputFile);
    }

    @Override
    public void validate(String msg){
        
    }
    
    @Override
    public void outputState(String filepath){
        if(verbose)
            logln("--- Outputing current state to " + filepath);
        try{
            StringBuilder modelStr = new StringBuilder();
            for(int k=0; k<K; k++){
                modelStr.append(k).append("\n");
                modelStr.append(topic_words[k].getConcentration()).append("\n");
                for(int v=0; v<V; v++)
                    modelStr.append(topic_words[k].getCenterElement(v)).append("\t");
                modelStr.append("\n");
            }

            StringBuilder assignStr = new StringBuilder();
            for(int d=0; d<D; d++){
                assignStr.append(d).append("\n");
                assignStr.append(doc_topics[d].getConcentration()).append("\n");
                for(int k=0; k<K; k++)
                    assignStr.append(doc_topics[d].getCenterElement(k)).append("\t");
                assignStr.append("\n");

                for(int n=0; n<words[d].length; n++)
                    assignStr.append(z[d][n]).append("\t");
                assignStr.append("\n");
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            data = assignStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
        }
        catch(Exception e){
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    @Override
    public void inputState(String filepath){
        this.initializeHierarchies();
        
        if(verbose)
            logln("--- Reading state from " + filepath);
        
        try{
            ZipFile zipFile = new ZipFile(filepath);
            Enumeration<? extends ZipEntry> entries = zipFile.entries();
            String line; String[] sline;

            // read in model
            ZipEntry modelEntry = entries.nextElement();
            BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
            for(int k=0; k<K; k++){
                reader.readLine();
                double concentration = Double.parseDouble(reader.readLine());
                sline = reader.readLine().split("\t");
                double[] mean = new double[V];
                if(sline.length != V)
                    throw new RuntimeException("Dimensions mismatch");
                for(int v=0; v<V; v++)
                    mean[v] = Double.parseDouble(sline[v]);
                this.topic_words[k] = new DirichletMultinomialModel(V, concentration, mean);
            }
            reader.close();

            // read in assignment
            ZipEntry assignEntry = entries.nextElement();
            reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(assignEntry), "UTF-8"));
            for(int d=0; d<D; d++){
                reader.readLine();
                double concentration = Double.parseDouble(reader.readLine());

                line = reader.readLine();
                sline = line.split("\t");
                double[] mean = new double[K];
                if(sline.length != K)
                    throw new RuntimeException("Dimensions mismatch. d = " + d
                            + ". " + sline.length
                            + " vs. " + K
                            + ". " + line);
                for(int k=0; k<K; k++)
                    mean[k] = Double.parseDouble(sline[k]);
                this.doc_topics[d] = new DirichletMultinomialModel(K, concentration, mean);

                line = reader.readLine();
                sline = line.split("\t");
                if(sline.length != words[d].length)
                    throw new RuntimeException("Dimensions mismatch. d = " + d 
                            + ". " + sline.length 
                            + " vs. " + words[d].length
                            + ". " + line);
                for(int n=0; n<words[d].length; n++){
                    z[d][n] = Integer.parseInt(sline[n]);
                    doc_topics[d].increment(z[d][n]);
                    topic_words[z[d][n]].increment(words[d][n]);
                }
            }
            reader.close();
        }
        catch(Exception e){
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    public void outputTopicCoherence(
            String filepath, 
            MimnoTopicCoherence topicCoherence) throws Exception{
        if(verbose)
            System.out.println("Outputing topic coherence to file " + filepath);
        
        if(this.wordVocab == null)
            throw new RuntimeException("The word vocab has not been assigned yet");
        
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for(int k=0; k<K; k++){
            double[] distribution = this.topic_words[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + topic_words[k].getCountSum()
                    + "\t" + score);
            for(int i=0; i<topicCoherence.getNumTokens(); i++)
                writer.write("\t" + this.wordVocab.get(topic[i]));
            writer.write("\n");
        }
        writer.close();
    }
    
    public double[][] getDocumentEmpiricalDistributions(){
        double[][] docEmpDists = new double[D][K];
        for(int d=0; d<D; d++)
            docEmpDists[d] = doc_topics[d].getEmpiricalDistribution();
        return docEmpDists;
    }
    
    public void outputDocTopicDistributions(String filepath) throws Exception{
        if(verbose)
            logln("Outputing per-document topic distribution to " + filepath);
        
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for(int d=0; d<D; d++){
            writer.write(Integer.toString(d));
            double[] docTopicDist = this.doc_topics[d].getDistribution();
            for(int k=0; k<K; k++)
                writer.write("\t" + docTopicDist[k]);
            writer.write("\n");
        }
        writer.close();
    }
    
    public void outputTopicWordDistributions(String filepath) throws Exception{
        if(verbose)
            logln("Outputing per-topic word distribution to " + filepath);
        
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for(int k=0; k<K; k++){
            writer.write(Integer.toString(k));
            double[] topicWordDist = this.topic_words[k].getDistribution();
            for(int v=0; v<V; v++)
                writer.write("\t" + topicWordDist[v]);
            writer.write("\n");
        }
        writer.close();
    }
}
