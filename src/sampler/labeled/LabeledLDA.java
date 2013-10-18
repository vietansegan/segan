package sampler.labeled;

import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import sampling.likelihood.DirichletMultinomialModel;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;

/**
 * This is an implementation of a Gibbs sampler for Labeled LDA (Ramage et. al.
 * EMNLP 09).
 *
 * Each document is associated with a set of labels.
 *
 * @author vietan
 */
public class LabeledLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int L;
    protected int V;
    protected int D;
    private DirichletMultinomialModel[] doc_labels;
    private DirichletMultinomialModel[] label_words;
    private int[][] z;
    private ArrayList<String> labelVocab;
    private int numTokens;
    private int numTokensChange;

    public void setLabelVocab(ArrayList<String> labelVocab) {
        this.labelVocab = labelVocab;
    }

    public void configure(String folder,
            int[][] words, int[][] labels,
            int V, int L,
            double alpha,
            double beta,
            InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.words = words;
        this.labels = labels;

        this.L = L;
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
        this.REP_INTERVAL = repInterval;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- label vocab:\t" + L);
            logln("--- word vocab:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            logln("--- beta:\t" + MiscUtils.formatDouble(beta));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_L-LDA")
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(MiscUtils.formatDouble(hyperparams.get(ALPHA)))
                .append("_b-").append(MiscUtils.formatDouble(hyperparams.get(BETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public DirichletMultinomialModel[] getTopicWordDistributions() {
        return this.label_words;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    private void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        label_words = new DirichletMultinomialModel[L];
        for (int ll = 0; ll < L; ll++) {
            label_words[ll] = new DirichletMultinomialModel(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        doc_labels = new DirichletMultinomialModel[D];
        for (int d = 0; d < D; d++) {
            doc_labels[d] = new DirichletMultinomialModel(L, hyperparams.get(ALPHA) * L, 1.0 / L);
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                int[] docLabels = labels[d];
                if (docLabels.length > 0) {
                    z[d][n] = docLabels[rand.nextInt(docLabels.length)];
                } else {
                    z[d][n] = rand.nextInt(L);
                }
                doc_labels[d].increment(z[d][n]);
                label_words[z[d][n]].increment(words[d][n]);
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
                IOUtils.createFolder(new File(getSamplerFolderPath(), ReportFolder));
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % REP_INTERVAL == 0) {
                String str = "Iter " + iter
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str + "\n");
                } else {
                    logln("--- Sampling. " + str + "\n");
                }
            }

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD);
                }
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    if (verbose) {
                        logln("*** *** Optimizing hyperparameters by slice sampling ...");
                        logln("*** *** cur param:" + MiscUtils.listToString(hyperparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }

                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        logln("*** *** new param:" + MiscUtils.listToString(sparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }
                }
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(
                        new File(getSamplerFolderPath(), ReportFolder),
                        "iter-" + iter + ".zip"));
            }
        }

        if (report) {
            outputState(new File(
                    new File(getSamplerFolderPath(), ReportFolder),
                    "iter-" + iter + ".zip"));
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (log) {
                BufferedWriter writer = IOUtils.getBufferedWriter(
                        new File(getSamplerFolderPath(), "likelihoods.txt"));
                for (int i = 0; i < logLikelihoods.size(); i++) {
                    writer.write(i + "\t" + logLikelihoods.get(i) + "\n");
                }
                writer.close();

                if (paramOptimized) {
                    this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                            "hyperparameters.txt"));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void sampleZ(int d, int n, 
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            label_words[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            doc_labels[d].decrement(z[d][n]);
        }

        int sampledZ;
        if (labels[d].length > 0) {
            double[] logprobs = new double[labels[d].length];
            for (int ii = 0; ii < labels[d].length; ii++) {
                logprobs[ii] = doc_labels[d].getLogLikelihood(labels[d][ii])
                        + label_words[labels[d][ii]].getLogLikelihood(words[d][n]);
            }
            sampledZ = labels[d][SamplerUtils.logMaxRescaleSample(logprobs)];
        } else {
            double[] logprobs = new double[L];
            for (int ll = 0; ll < L; ll++) {
                logprobs[ll] = doc_labels[d].getLogLikelihood(ll)
                        + label_words[ll].getLogLikelihood(words[d][n]);
            }
            sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
        }

        if (sampledZ != z[d][n]) {
            numTokensChange++;
        }
        z[d][n] = sampledZ;

        if (addToModel) {
            label_words[z[d][n]].increment(words[d][n]);
        }
        if (addToData) {
            doc_labels[d].increment(z[d][n]);
        }
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += doc_labels[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int l = 0; l < L; l++) {
            topicWordLlh += label_words[l].getLogLikelihood();
        }

        double llh = docTopicLlh + topicWordLlh;
        if (verbose) {
            logln(">>> doc-topic: " + MiscUtils.formatDouble(docTopicLlh)
                    + "\ttopic-word: " + MiscUtils.formatDouble(topicWordLlh)
                    + "\tllh: " + MiscUtils.formatDouble(llh));
        }
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double llh = 0;
        for (int d = 0; d < D; d++) {
            llh += doc_labels[d].getLogLikelihood(newParams.get(ALPHA) * L, 1.0 / L);
        }
        for (int l = 0; l < L; l++) {
            llh += label_words[l].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.doc_labels[d].setConcentration(this.hyperparams.get(ALPHA) * L);
        }
        for (int l = 0; l < L; l++) {
            this.label_words[l].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            this.doc_labels[d].validate(msg);
        }
        for (int l = 0; l < L; l++) {
            this.label_words[l].validate(msg);
        }

        int total = 0;
        for (int d = 0; d < D; d++) {
            total += doc_labels[d].getCountSum();
        }
        if (total != numTokens) {
            throw new RuntimeException("Token counts mismatch. "
                    + total + " vs. " + numTokens);
        }

        total = 0;
        for (int l = 0; l < L; l++) {
            total += label_words[l].getCountSum();
        }
        if (total != numTokens) {
            throw new RuntimeException("Token counts mismatch. "
                    + total + " vs. " + numTokens);
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
            for (int k = 0; k < L; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirichletMultinomialModel.output(label_words[k])).append("\n");
            }

            // data
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirichletMultinomialModel.output(doc_labels[d])).append("\n");

                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }

        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        validate("Done reading state from " + filepath);
    }

    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            // initialize
            this.initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < L; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                label_words[k] = DirichletMultinomialModel.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                doc_labels[d] = DirichletMultinomialModel.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (this.labelVocab == null) {
            throw new RuntimeException("The topic vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < L; k++) {
            double[] distrs = label_words[k].getDistribution();
            String[] topWords = getTopWords(distrs, numTopWords);
            writer.write("[" + k
                    + ", " + labelVocab.get(k)
                    + ", " + label_words[k].getCountSum()
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");
        }
        writer.close();
    }
}
