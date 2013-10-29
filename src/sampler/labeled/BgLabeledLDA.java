package sampler.labeled;

import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;

/**
 * Implementation of Background Labeled LDA - a variant of Labeled LDA.
 *
 * In labeled LDA, each document d is associated with T_d labels and tokens of d
 * can only be assigned to one of these T_d labels.
 *
 * In background labeled LDA, in addition to T_d labels, tokens of d can also be
 * assigned to a shared topic. This is to capture a background topic. I
 *
 * If a document does not have labels, its tokens can be assigned to all topics
 * (i.e., K+1 topics).
 *
 * @author vietan
 */
public class BgLabeledLDA extends AbstractSampler {

    public static final int BACKGROUND = 0;
    public static final int FOREGROUND = 1;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int L;
    protected int V;
    protected int D;
    private DirMult[] doc_labels;
    private DirMult[] label_words;
    private int[][] z;
    private ArrayList<String> labelVocab;
    private int numTokens;
    private int numTokensChange;
    private int[][] candLabels;

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

        this.candLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            int Ld = labels[d].length;
            if (Ld == 0) {
                this.candLabels[d] = new int[L + 1];
                for (int ll = 0; ll < L + 1; ll++) {
                    this.candLabels[d][ll] = ll;
                }
            } else {
                this.candLabels[d] = new int[Ld + 1];
                System.arraycopy(labels[d], 0, this.candLabels[d], 0, Ld);
                this.candLabels[d][Ld] = L;
            }
        }

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
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
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
                .append("_BG-L-LDA")
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(MiscUtils.formatDouble(hyperparams.get(ALPHA)))
                .append("_b-").append(MiscUtils.formatDouble(hyperparams.get(BETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
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

        label_words = new DirMult[L + 1];
        for (int ll = 0; ll < L + 1; ll++) {
            label_words[ll] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        doc_labels = new DirMult[D];
        for (int d = 0; d < D; d++) {
            int Ld = candLabels[d].length;
            doc_labels[d] = new DirMult(Ld, hyperparams.get(ALPHA) * Ld, 1.0 / Ld);
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
                int sampledIdx = rand.nextInt(candLabels[d].length);
                z[d][n] = sampledIdx;
                doc_labels[d].increment(z[d][n]);
                int labelIdx = candLabels[d][sampledIdx];
                label_words[labelIdx].increment(words[d][n]);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChange = 0;

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    sampleZ(d, t, REMOVE, ADD, REMOVE, ADD);
                }
            }

            if (debug) {
                validate("Iter " + iter);
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                double changeRatio = (double) numTokensChange / numTokens;
                String str = "Iter " + iter
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChange
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio);
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str + "\n");
                } else {
                    logln("--- Sampling. " + str + "\n");
                }
            }
        }
    }

    public void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        int curZ = z[d][n];
        int curLabel = candLabels[d][curZ];

        if (removeFromModel) {
            label_words[curLabel].decrement(words[d][n]);
        }
        if (removeFromData) {
            doc_labels[d].decrement(curZ);
        }

        double[] logprobs = new double[candLabels[d].length];
        for (int ii = 0; ii < candLabels[d].length; ii++) {
            int ll = candLabels[d][ii];
            logprobs[ii] = doc_labels[d].getLogLikelihood(ii)
                    + label_words[ll].getLogLikelihood(words[d][n]);
        }

        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
        if (sampledIdx != curZ) {
            numTokensChange++;
        }

        z[d][n] = sampledIdx;
        if (addToModel) {
            label_words[candLabels[d][z[d][n]]].increment(words[d][n]);
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
        for (int l = 0; l < L + 1; l++) {
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
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            this.doc_labels[d].validate(msg);
        }

        for (int k = 0; k < L + 1; k++) {
            this.label_words[k].validate(msg);
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
            for (int k = 0; k < L + 1; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(label_words[k])).append("\n");
            }

            // data
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(doc_labels[d])).append("\n");

                // topics
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
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + ModelFileExt);
            for (int k = 0; k < L + 1; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                label_words[k] = DirMult.input(reader.readLine());
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

            String[] sline;
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                doc_labels[d] = DirMult.input(reader.readLine());

                sline = reader.readLine().split("\t");
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
        for (int k = 0; k < L + 1; k++) {
            double[] distrs = label_words[k].getDistribution();
            String[] topWords = getTopWords(distrs, numTopWords);
            String labelStr = "Background";
            if (k < L) {
                labelStr = labelVocab.get(k);
            }
            writer.write("[" + k
                    + ", " + labelStr
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
