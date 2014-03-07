/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.supervised.classification;

import core.AbstractSampler;
import java.util.ArrayList;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class SLDA {
//extends AbstractSampler {
//    public static final int ALPHA = 0;
//    public static final int BETA = 1;
//    public static final int MU = 2;
//    public static final int SIGMA = 3;
//    
//    protected int K; // number of topics
//    protected int V; // vocab size
//    protected int D; // number of documents
//    protected int C; // number of classes
//    
//    protected int[][] words; // documents
//    protected int[] responses; // class labels
//    
//    protected int[][] z;
//    
//    protected DirichletMultinomialModel[] docTopics;
//    protected DirichletMultinomialModel[] topicWords;
//    protected double[][] regParams;
//    
//    protected ArrayList<double[][]> regParamsList;
//    private int numTokens;
//    private int numTokensChanged;
//    
//    public void configure(String folder, 
//            int[][] words, int[] y,
//            int V, int K, int C,
//            double alpha,
//            double beta, 
//            double mu, // mean of Gaussian for regression parameters
//            double sigma, // stadard deviation of Gaussian for regression parameters
//            InitialState initState, 
//            boolean paramOpt,
//            int burnin, int maxiter, int samplelag){
//        if(verbose)
//            logln("Configuring ...");
//        
//        this.folder = folder;
//        this.words = words;
//        this.responses = y;
//        
//        this.K = K;
//        this.V = V;
//        this.C = C;
//        this.D = this.words.length;
//        
//        this.hyperparams = new ArrayList<Double>();
//        this.hyperparams.add(alpha);
//        this.hyperparams.add(beta);
//        this.hyperparams.add(mu);
//        this.hyperparams.add(sigma);
//        
//        this.sampledParams = new ArrayList<ArrayList<Double>>();
//        this.sampledParams.add(cloneHyperparameters());
//
//        this.BURN_IN = burnin;
//        this.MAX_ITER = maxiter;
//        this.LAG = samplelag;
//        
////        this.regressor = new OLSMultipleLinearRegression();
////        this.regressor.setNoIntercept(true);
//        
//        this.initState = initState;
//        this.paramOptimized = paramOpt;
//        this.prefix += initState.toString();
////        this.regressionParameters = new ArrayList<double[]>();
//        this.setName();
//        
//        if(!debug)
//            System.err.close();
//        
//        numTokens = 0;
//        for(int d=0; d<D; d++)
//            numTokens += words[d].length;
//        
//        if(verbose){
//            logln("--- folder\t" + folder);
//            logln("--- num topics:\t" + K);
//            logln("--- alpha:\t" + hyperparams.get(ALPHA));
//            logln("--- beta:\t" + hyperparams.get(BETA));
//            logln("--- reg mu:\t" + hyperparams.get(MU));
//            logln("--- reg sigma:\t" + hyperparams.get(SIGMA));
//            logln("--- burn-in:\t" + BURN_IN);
//            logln("--- max iter:\t" + MAX_ITER);
//            logln("--- sample lag:\t" + LAG);
//            logln("--- paramopt:\t" + paramOptimized);
//            logln("--- initialize:\t" + initState);
//            logln("--- # tokens:\t" + numTokens);
//        }
//        
//        if(!debug)
//            System.err.close();
//    }
//    
//    protected void setName(){
//        StringBuilder str = new StringBuilder();
//        str.append(this.prefix)
//                .append("_sLDA-cls")
//                .append("_B-").append(BURN_IN)
//                .append("_M-").append(MAX_ITER)
//                .append("_L-").append(LAG)
//                .append("_K-").append(K)
//                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
//                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
//                .append("_m-").append(formatter.format(hyperparams.get(MU)))
//                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)))
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
//        topicWords = new DirichletMultinomialModel[K];
//        for(int k=0; k<K; k++)
//            topicWords[k] = new DirichletMultinomialModel(V, hyperparams.get(BETA), 1.0/V);
//        
//        regParams = new double[C][K];
//        for(int c=0; c<C; c++){
//            for(int k=0; k<K; k++){
//                // how to initialize the regression parameters
//            }
//        }
//    }
//    
//    protected void initializeDataStructure() {
//        z = new int[D][];
//        for(int d=0; d<D; d++)
//            z[d] = new int[words[d].length];
//        
//        docTopics = new DirichletMultinomialModel[D];
//        for(int d=0; d<D; d++)
//            docTopics[d] = new DirichletMultinomialModel(K, hyperparams.get(ALPHA), 1.0/K);
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
//            for(int n=0; n<words[d].length; n++){
//                z[d][n] = rand.nextInt(K);
//                docTopics[d].increment(z[d][n]);
//                topicWords[z[d][n]].increment(words[d][n]);
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
//        RegressionEvaluation eval;
//        for(iter=0; iter<MAX_ITER; iter++){
//            numTokensChanged = 0;
//            
//            // store llh after every iteration
//            double loglikelihood = this.getLogLikelihood();
//            logLikelihoods.add(loglikelihood);
//            
//            // store regression parameters after every iteration
//            double[][] rps = new double[C][K];
//            for(int c=0; c<C; c++){
//                for(int k=0; k<K; k++)
//                    rps[c][k] = regParams[c][k];
//            }
//            this.regParamsList.add(rps);
//            
//            if (verbose){
//                if(iter < BURN_IN)
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
//            // update the regression parameters
//            updateRegressionParameters();
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
//            if(verbose){
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
////                logln("--- --- # tokens: " + numTokens
////                        + ". # token changed: " + numTokensChanged
////                        + ". change ratio: " + (double)numTokensChanged / numTokens
////                        + "\n");
//            }
//            
//            if(debug)
//                validate("iter " + iter);
//            
//            if(verbose)
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
//    
//    private void sampleZ(int d, int n, boolean remove, boolean add, boolean observe){
//        if(remove){
//            docTopics[d].decrement(z[d][n]);
//            topicWords[z[d][n]].decrement(words[d][n]);
//        }
//        
//        double[] logprobs = new double[K];
//        for(int k=0; k<K; k++){
//            logprobs[k] =
//                    docTopics[d].getLogLikelihood(k)
//                    + topicWords[k].getLogLikelihood(words[d][n]);
//            if(observe){
//                
//            }
//        }
//    }
}
