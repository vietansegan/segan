/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.labeled;

import core.AbstractSampler;
import java.util.ArrayList;
import sampling.likelihood.DirichletMultinomialModel;

/**
 *
 * @author vietan
 */
public class LabeledLDASampler {
//extends AbstractSampler {
//    public static final int ALPHA = 0;
//    public static final int BETA = 1;
//    public static final int GAMMA = 2;
//    
//    protected int C; // number of categories
//    protected int K; // number of topics;
//    protected int V; // vocab size
//    protected int D; // number of documents
//    protected int T; // number of topics per document
//    
//    protected int[][] words;
//    protected int[] categories;
//    protected int[][] topics;
//    
//    protected int[][] z;
//    protected int[][] t;
//    
//    private DirichletMultinomialModel[] topic_word_dists;
//    private DirichletMultinomialModel[] category_topic_dists;
//    private DirichletMultinomialModel[] doc_topic_dists;
//    
//    private int numTokens;
//    private int numTopics;
//    private int numTokensChange;
//    private int numTopicsChange;
//    
//    public void configure(String folder, 
//            int[][] words, 
//            int[] categories,
//            int[][] topics,
//            int V, int K, int C, int T,
//            double alpha,
//            double beta, 
//            double gamma,
//            InitialState initState, 
//            boolean paramOpt,
//            int burnin, int maxiter, int samplelag, int repInt){
//        if(verbose)
//            logln("Configuring ...");
//        
//        this.folder = folder;
//        this.words = words;
//        this.categories = categories;
//        this.topics = topics;
//        
//        this.K = K;
//        this.V = V;
//        this.C = C;
//        this.T = T;
//        this.D = this.words.length;
//        
//        this.hyperparams = new ArrayList<Double>();
//        this.hyperparams.add(alpha);
//        this.hyperparams.add(beta);
//        this.hyperparams.add(gamma);
//        
//        this.sampledParams = new ArrayList<ArrayList<Double>>();
//        this.sampledParams.add(cloneHyperparameters());
//
//        this.BURN_IN = burnin;
//        this.MAX_ITER = maxiter;
//        this.LAG = samplelag;
//        this.REP_INTERVAL = repInt;
//        
//        this.initState = initState;
//        this.paramOptimized = paramOpt;
//        this.prefix += initState.toString();
//        this.setName();
//        
//        if(!debug)
//            System.err.close();
//        
//        numTopics = 0;
//        numTokens = 0;
//        for(int d=0; d<D; d++){
//            numTokens += words[d].length;
//            
//        }
//        
//        if(verbose){
//            logln("--- folder\t" + folder);
//            logln("--- num topics:\t" + K);
//            logln("--- num topics per document:\t" + T);
//            logln("--- vocab size:\t" + V);
//            logln("--- num documents:\t" + D);
//            logln("--- alpha:\t" + hyperparams.get(ALPHA));
//            logln("--- beta:\t" + hyperparams.get(BETA));
//            logln("--- gamma:\t" + hyperparams.get(GAMMA));
//            logln("--- burn-in:\t" + BURN_IN);
//            logln("--- max iter:\t" + MAX_ITER);
//            logln("--- sample lag:\t" + LAG);
//            logln("--- paramopt:\t" + paramOptimized);
//            logln("--- initialize:\t" + initState);
//            logln("--- # tokens:\t" + numTokens);
//            logln("--- # latent topics:\t" + numTopics);
//        }
//    }
//    
//    protected void setName(){
//        StringBuilder str = new StringBuilder();
//        str.append(this.prefix)
//                .append("_ctLDA")
//                .append("_B-").append(BURN_IN)
//                .append("_M-").append(MAX_ITER)
//                .append("_L-").append(LAG)
//                .append("_K-").append(K)
//                .append("_C-").append(C)
//                .append("_T-").append(T)
//                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
//                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
//                .append("_g-").append(formatter.format(hyperparams.get(GAMMA)))
//                ;
//        str.append("_opt-").append(this.paramOptimized);
//        this.name = str.toString();
//    }
//    
//    @Override
//    public void initialize(){
//        if(verbose)
//            logln("Initializing ...");
//
//        initializeModelStructure();
//        
//        initializeDataStructure();
//        
//        initializeAssignments();
//
//        if(debug)
//            validate("Initialized");
//    }
//    
//    private void initializeModelStructure(){
//        this.topic_word_dists = new DirichletMultinomialModel[K];
//        for(int k=0; k<K; k++)
//            this.topic_word_dists[k] = new DirichletMultinomialModel(V, hyperparams.get(BETA), 1.0/V);
//        
//        this.category_topic_dists = new DirichletMultinomialModel[C];
//        for(int c=0; c<C; c++)
//            this.category_topic_dists[c] = new DirichletMultinomialModel(K, hyperparams.get(GAMMA), 1.0/K);
//    }
//    
//    private void initializeDataStructure(){
//        this.doc_topic_dists = new DirichletMultinomialModel[D];
//        for(int d=0; d<D; d++)
//            this.doc_topic_dists[d] = new DirichletMultinomialModel(T, hyperparams.get(ALPHA), 1.0/T);
//        
//        this.z = new int[D][];
//        for(int d=0; d<D; d++)
//            this.z[d] = new int[words[d].length];
//        
//        this.t = new int[D][T];
//    }
//    
//    protected void initializeAssignments() {
//        switch(initState){
//            case RANDOM :
//                this.initializeRandomAssignments();
//                break;
//            default:
//                throw new RuntimeException("Initialization not supported");
//        }
//    }
//    
//    private void initializeRandomAssignments(){
//        for(int d=0; d<D; d++){
//            for(int i=0; i<T; i++){
//                if(i < this.topics[d].length)
//                    t[d][i] = topics[d][i];
//                else
//                    t[d][i] = rand.nextInt(K);
//                category_topic_dists[categories[d]].increment(t[d][i]);
//            }
//            
//            for(int n=0; n<words[d].length; n++){
//                z[d][n] = rand.nextInt(T);
//                doc_topic_dists[d].increment(z[d][n]);
//                topic_word_dists[t[d][z[d][n]]].increment(words[d][n]);
//            }
//        }
//    }
//    
//    @Override
//    public void iterate(){
//        if(verbose)
//            logln("Iterating ...");        
//        logLikelihoods = new ArrayList<Double>();
//        
//        try{
//            if(report)
//                IOUtils.createFolder(this.folder + this.getSamplerFolder() + ReportFolder);
//        }
//        catch(Exception e){
//            e.printStackTrace();
//            System.exit(1);
//        }
//                
//        if(log && !isLogging())
//            openLogger();
//        
//        logln(getClass().toString());
//        startTime = System.currentTimeMillis();
//        
//        for(iter=0; iter<MAX_ITER; iter++){
//            numTokensChange = 0;
//            numTopicsChange = 0;
//            
//            // store llh after every iteration
//            double loglikelihood = this.getLogLikelihood();
//            logLikelihoods.add(loglikelihood);
//            
//            if (verbose && iter % REP_INTERVAL == 0){
//                if(iter < BURN_IN )
//                    logln("--- Burning in. Iter " + iter
//                            + "\t llh = " + loglikelihood
//                            + "\n" + getCurrentState()
//                            );
//                else
//                    logln("--- Sampling. Iter " + iter
//                            + "\t llh = " + loglikelihood
//                            + "\n" + getCurrentState()
//                            );
//            }
//            
//            // sample topic assignments
//            for(int d=0; d<D; d++){
//                for(int n=0; n<words[d].length; n++)
//                    sampleZ(d, n, REMOVE, ADD, OBSERVED);
//            }
//                        
//            // parameter optimization
//            if(iter % LAG == 0 && iter >= BURN_IN){
//                if(paramOptimized){ // slice sampling
//                    sliceSample();
//                    ArrayList<Double> sparams = new ArrayList<Double>();
//                    for(double param : this.hyperparams)
//                        sparams.add(param);
//                    this.sampledParams.add(sparams);
//                    
//                    if(verbose){
//                        for(double p : sparams)
//                            System.out.println(p);
//                    }
//                }
//            }
//            
////            if(verbose && iter % REP_INTERVAL == 0){
////                double[] trPredResponses = getRegressionValues();
////                eval = new RegressionEvaluation(responses, trPredResponses);
////                eval.computeCorrelationCoefficient();
////                eval.computeMeanSquareError();
////                eval.computeRSquared();
////                ArrayList<Measurement> measurements = eval.getMeasurements();
////            
////                logln("--- --- After updating regression parameters Zs:\t" + getCurrentState());
////                for(Measurement measurement : measurements)
////                    logln("--- --- --- " + measurement.getName() + ":\t" + measurement.getValue());
////                
////                logln("--- --- # optimized: " + optimizeCount
////                        + ". # converged: " + convergeCount
////                        + ". convergence ratio: " + (double)convergeCount / optimizeCount);
////                logln("--- --- # tokens: " + numTokens
////                        + ". # token changed: " + numTokensChanged
////                        + ". change ratio: " + (double)numTokensChanged / numTokens
////                        + "\n");
////            }
//            
//            if(debug)
//                validate("iter " + iter);
//            
//            if(verbose && iter % REP_INTERVAL == 0)
//                System.out.println();
//            
//            // store model
//            if(report && iter >= BURN_IN && iter % LAG == 0)
//                outputState(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + ".zip");
//        }
//        
//        if(report)
//            outputState(this.folder + this.getSamplerFolder() + "final.zip");
//        
//        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
//        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");
//
//        if(log && isLogging())
//            closeLogger();
//        
//        try{
//            if(paramOptimized && log)
//                this.outputSampledHyperparameters(this.folder + this.getSamplerFolder() + "hyperparameters.txt");
//        }
//        catch(Exception e){
//            e.printStackTrace();
//            System.exit(1);
//        }
//    }
}
