package sampler.labeled;

import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class PriorLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1; // 0.1
    public static final int ETA = 2;
    protected int[][] words; // [D] x [N_d]
    protected int[][] topics; // [D] x [T_d] observed topics; for some doc, this can be partially or totally unobserved
    protected int[][] z;
    protected int K; // number of topics;
    protected int V; // vocab size
    protected int D; // number of documents
    private DirMult[] topic_word_dists; // K multinomials over V words
    private DirMult[] doc_topic_dists; // D multinomials over K topics
    private int numTokens;      // number of token assignments to be sampled
    private int numTokensChange;
    private ArrayList<String> topicVocab;
    private double[] empTopicL2Norms;
    private double[][] empTopicWordDists;
    private double[] idfs; // V-dim vector

    public void configure(String folder,
            int[][] words,
            int[][] topics,
            int V, int K,
            double alpha,
            double beta,
            double eta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.words = words;
        this.topics = topics;

        this.K = K;
        this.V = V;

        if (this.words != null) { // during test time
            this.D = this.words.length;
            numTokens = 0;
            for (int d = 0; d < D; d++) {
                numTokens += words[d].length;
            }
        }

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.hyperparams.add(eta);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- vocab size:\t" + V);
            logln("--- num documents:\t" + D);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- eta:\t" + MiscUtils.formatDouble(hyperparams.get(ETA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- # tokens:\t" + numTokens);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_prior-LDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_e-").append(formatter.format(hyperparams.get(ETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public File getReportFolderPath() {
        return new File(this.getSamplerFolderPath(), ReportFolder);
    }

    public void setTopicVocab(ArrayList<String> topicVocab) {
        this.topicVocab = topicVocab;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    private void initializeModelStructure() {
        this.topic_word_dists = new DirMult[K];
        for (int kk = 0; kk < K; kk++) {
            this.topic_word_dists[kk] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    private void initializeDataStructure() {
        this.doc_topic_dists = new DirMult[D];
        for (int dd = 0; dd < D; dd++) {
            double[] prior = getDocumentPrior(dd, hyperparams.get(ALPHA), hyperparams.get(ETA));
            this.doc_topic_dists[dd] = new DirMult(prior);
        }

        this.z = new int[D][];
        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
        }
    }

    private double[] getDocumentPrior(int d, double alpha, double eta) {
        double[] prior = new double[K];
        Arrays.fill(prior, hyperparams.get(ALPHA));
        for (int t : topics[d]) {
            prior[t] = eta / topics[d].length + alpha;
        }
        return prior;
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                this.z[d][n] = rand.nextInt(K);
                this.doc_topic_dists[d].increment(z[d][n]);
                this.topic_word_dists[z[d][n]].increment(words[d][n]);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        try {
            if (report) {
                IOUtils.createFolder(new File(this.getSamplerFolderPath(), ReportFolder));
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChange = 0;

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\n" + getCurrentState());
                }
            }

            // sample topic assignments
            for (int d = 0; d < D; d++) {
                if (debug && d % 10000 == 0) {
                    logln(">>> >>> Sampling d = " + d);
                }

                // sample topic for each word token
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD);
                }
            }

            // parameter optimization
            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        for (double p : sparams) {
                            System.out.println(p);
                        }
                    }
                }
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChange
                        + ". change ratio: " + (double) numTokensChange / numTokens);
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(this.getReportFolderPath(),
                        "iter-" + iter + ".zip").getAbsolutePath());
            }
        }

        if (report) { // output final model
            outputState(new File(this.getReportFolderPath(),
                    "iter-" + iter + ".zip").getAbsolutePath());
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(this.getSamplerFolderPath(),
                        "hyperparameters.txt").getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        int cur_word = words[d][n];

        if (removeFromData) {
            doc_topic_dists[d].decrement(z[d][n]);
        }
        if (removeFromModel) {
            topic_word_dists[z[d][n]].decrement(cur_word);
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            double logprob = doc_topic_dists[d].getLogLikelihood(k)
                    + topic_word_dists[k].getLogLikelihood(cur_word);
            logprobs[k] = logprob;
        }

        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        // debug
        if (z[d][n] != sampledZ) {
            numTokensChange++;
        }

        z[d][n] = sampledZ;

        if (addToData) {
            doc_topic_dists[d].increment(z[d][n]);
        }
        if (addToModel) {
            topic_word_dists[z[d][n]].increment(cur_word);
        }
    }

    @Override
    public double getLogLikelihood() {
        double doc_topic = 0.0;
        for (int d = 0; d < D; d++) {
            doc_topic += this.doc_topic_dists[d].getLogLikelihood();
        }
        double topic_word = 0.0;
        for (int k = 0; k < K; k++) {
            topic_word += this.topic_word_dists[k].getLogLikelihood();
        }

        double llh = doc_topic + topic_word;
        if (verbose && iter % REP_INTERVAL == 0) {
            logln(">>> topic-word: " + MiscUtils.formatDouble(topic_word)
                    + "\tdoc-topic: " + MiscUtils.formatDouble(doc_topic)
                    + "\tllh: " + MiscUtils.formatDouble(llh));
        }

        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        double val = 0.0;

        for (int d = 0; d < D; d++) {
            double[] prior = getDocumentPrior(d, newParams.get(ALPHA), newParams.get(ETA));
            val += this.doc_topic_dists[d].getLogLikelihood(prior);
        }

        for (int k = 0; k < K; k++) {
            val += this.topic_word_dists[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }

        return val;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        this.hyperparams = newParams;

        for (int dd = 0; dd < D; dd++) {
            double[] prior = getDocumentPrior(dd, hyperparams.get(ALPHA), hyperparams.get(ETA));
            this.doc_topic_dists[dd].setHyperparameters(prior);
        }

        for (int kk = 0; kk < K; kk++) {
            this.topic_word_dists[kk].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    @Override
    public void validate(String msg) {
        for (int kk = 0; kk < K; kk++) {
            this.topic_word_dists[kk].validate(msg);
        }
        for (int dd = 0; dd < D; dd++) {
            this.doc_topic_dists[dd].validate(msg);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            modelStr.append("K\t").append(K).append("\n");
            modelStr.append("V\t").append(V).append("\n");
            modelStr.append("alpha\t").append(hyperparams.get(ALPHA)).append("\n");
            modelStr.append("beta\t").append(hyperparams.get(BETA)).append("\n");
            modelStr.append("eta\t").append(hyperparams.get(ETA)).append("\n");
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(topic_word_dists[k])).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(doc_topic_dists[d])).append("\n");
            }

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing state to " + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }

        try {
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing state from " + filepath);
        }

        validate("Done reading state from " + filepath);
    }

    public void inputFinalModel() {
        try {
            inputModel(new File(this.getReportFolderPath(), "iter-" + MAX_ITER + ".zip").getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading final predictor model");
        }
    }

    public void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- Loading model from " + zipFilepath);
        }
        String filename = IOUtils.removeExtension(new File(zipFilepath).getName());
        BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
        K = Integer.parseInt(reader.readLine().split("\t")[1]);
        V = Integer.parseInt(reader.readLine().split("\t")[1]);
        double alpha = Double.parseDouble(reader.readLine().split("\t")[1]);
        double beta = Double.parseDouble(reader.readLine().split("\t")[1]);
        double eta = Double.parseDouble(reader.readLine().split("\t")[1]);
        hyperparams = new ArrayList<Double>();
        hyperparams.add(alpha);
        hyperparams.add(beta);
        hyperparams.add(eta);

        this.topic_word_dists = new DirMult[K];
        for (int k = 0; k < K; k++) {
            int topicIdx = Integer.parseInt(reader.readLine());
            if (topicIdx != k) {
                throw new RuntimeException("Topic indices mismatch when loading model");
            }
            topic_word_dists[k] = DirMult.input(reader.readLine());
        }
        reader.close();
    }

//    @Override
//    public void inputPredictor(File predictorFile) {
//        try {
//            this.inputModel(predictorFile.getAbsolutePath());
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while loading predictor from "
//                    + predictorFile);
//        }
//    }
//    @Override
//    public void outputPredictor(File predictorFile) {
//        this.outputState(predictorFile.getAbsolutePath());
//    }
//    @Override
//    public double[] predict(Page page) {
//        int[] newWords = page.getWords();
//
////        return predictByCosineSimilarity(newWords);
//
//        return predictBySampling(newWords);
//    }
    public void setIdfs(double[] idfs) {
        this.idfs = idfs;
    }

    /**
     * Prediction by computing cosine similarity between the TF-IDF word vector
     * of the test document with each topic's word distribution.
     *
     * @param newWords Token vector of the test document
     * @return Score vector [K-dim]
     */
    private double[] predictByCosineSimilarity(int[] newWords) {
        double[] scores = new double[K];
        if (newWords.length == 0) {
            return scores;
        }

        SparseCount typeCount = new SparseCount();
        for (int n = 0; n < newWords.length; n++) {
            typeCount.increment(newWords[n]);
        }

        if (this.empTopicL2Norms == null) {
            this.empTopicL2Norms = new double[K];
            this.empTopicWordDists = new double[K][];
            for (int k = 0; k < K; k++) {
                this.empTopicWordDists[k] = topic_word_dists[k].getDistribution();
                double topicNorm = 0.0;
                for (double e : this.empTopicWordDists[k]) {
                    topicNorm += e * e;
                }
                topicNorm = Math.sqrt(topicNorm);
                this.empTopicL2Norms[k] = topicNorm;
            }
        }

        for (int k = 0; k < K; k++) {
            double score = 0.0;
            for (int ii : typeCount.getIndices()) {
                double wordIdf = 1;
                if (this.idfs != null) {
                    wordIdf = this.idfs[ii];
                }
                score += typeCount.getCount(ii) * wordIdf * this.empTopicWordDists[k][ii];
            }
            scores[k] = score / this.empTopicL2Norms[k];
        }

        return scores;
    }
    private int testBurnIn = 100;
    private int testMaxIter = 500;
    private int testSampleLag = 5;

    /**
     * Prediction by performing Gibbs sampling using the final model.
     *
     * @param newWords Token vector of the test document
     * @return Score vector [K-dim]
     */
    private double[] predictBySampling(int[] newWords) {
        // initialize data structure
        DirMult newDocTopicDist =
                new DirMult(K, hyperparams.get(ALPHA), 1.0 / K);
        int[] newZ = new int[newWords.length];

        for (int n = 0; n < newWords.length; n++) {
            newZ[n] = rand.nextInt(K);
            newDocTopicDist.increment(newZ[n]);
        }

        ArrayList<int[]> predDocTopics = new ArrayList<int[]>();
        for (iter = 0; iter < testMaxIter; iter++) {
            // sample z
            for (int n = 0; n < newWords.length; n++) {
                newDocTopicDist.decrement(newZ[n]);

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    double logprob = newDocTopicDist.getLogLikelihood(k)
                            + topic_word_dists[k].getLogLikelihood(newWords[n]);
                    logprobs[k] = logprob;
                }
                newZ[n] = SamplerUtils.logMaxRescaleSample(logprobs);
                newDocTopicDist.increment(newZ[n]);
            }

            // copy prediction
            if (iter >= testBurnIn && iter % testSampleLag == 0) {
                int[] snapNewT = new int[newZ.length];
                System.arraycopy(newZ, 0, snapNewT, 0, snapNewT.length);
                predDocTopics.add(snapNewT);
            }
        }

        double[] topicScores = new double[K];
        int totalCount = 0;
        for (int[] predTs : predDocTopics) {
            for (int predT : predTs) {
                topicScores[predT]++;
            }
            totalCount += predTs.length;
        }
        for (int k = 0; k < topicScores.length; k++) {
            topicScores[k] /= totalCount;
        }
        return topicScores;
    }

    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (this.topicVocab == null) {
            throw new RuntimeException("The topic vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            double[] distrs = topic_word_dists[k].getDistribution();
            String[] topWords = getTopWords(distrs, numTopWords);
            writer.write("[" + k
                    + ", " + topicVocab.get(k)
                    + ", " + topic_word_dists[k].getCountSum()
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");
        }
        writer.close();
    }
}
