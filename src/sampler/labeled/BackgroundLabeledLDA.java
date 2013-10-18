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
 *
 * @author vietan
 */
public class BackgroundLabeledLDA extends AbstractSampler {

    public static final int BACKGROUND = 0;
    public static final int FOREGROUND = 1;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int GAMMA = 2;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int L;
    protected int V;
    protected int D;
    private DirichletMultinomialModel[] doc_labels;
    private DirichletMultinomialModel[] doc_foreback;
    private DirichletMultinomialModel[] label_words;
    private DirichletMultinomialModel background;
    private int[][] z;
    private int[][] y;
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
            double gamma,
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
        this.hyperparams.add(gamma);

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
            logln("--- gamma:\t" + MiscUtils.formatDouble(gamma));
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
                .append("_b-").append(MiscUtils.formatDouble(hyperparams.get(BETA)))
                .append("_g-").append(MiscUtils.formatDouble(hyperparams.get(GAMMA)));
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

        label_words = new DirichletMultinomialModel[L];
        for (int ll = 0; ll < L; ll++) {
            label_words[ll] = new DirichletMultinomialModel(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        background = new DirichletMultinomialModel(V, hyperparams.get(BETA) * V, 1.0 / V);
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        doc_foreback = new DirichletMultinomialModel[D];
        for (int d = 0; d < D; d++) {
            doc_foreback[d] = new DirichletMultinomialModel(2, hyperparams.get(GAMMA) * 2, 0.5);
        }

        doc_labels = new DirichletMultinomialModel[D];
        for (int d = 0; d < D; d++) {
            doc_labels[d] = new DirichletMultinomialModel(L, hyperparams.get(ALPHA) * L, 1.0 / L);
        }

        z = new int[D][];
        y = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
            y[d] = new int[words[d].length];
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                y[d][n] = rand.nextInt(2);
                doc_foreback[d].increment(y[d][n]);

                if (y[d][n] == BACKGROUND) {
                    background.increment(words[d][n]);
                } else {
                    int[] docLabels = labels[d];
                    if (docLabels.length > 0) { // if labeled
                        z[d][n] = docLabels[rand.nextInt(docLabels.length)];
                    } else // if not labeled
                    {
                        z[d][n] = rand.nextInt(L);
                    }
                    doc_labels[d].increment(z[d][n]);
                    label_words[z[d][n]].increment(words[d][n]);
                }
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
                    sampleYZ(d, t, REMOVE, ADD, REMOVE, ADD);
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

    /**
     * Sample the topic assignment for each token.
     */
    public void sampleYZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        int curY = y[d][n];
        int curZ = z[d][n];

        // remove counts
        if (removeFromModel) {
            if (y[d][n] == BACKGROUND) {
                background.decrement(words[d][n]);
            } else {
                label_words[z[d][n]].decrement(words[d][n]);
            }
        }
        if (removeFromData) {
            if (y[d][n] != BACKGROUND) {
                doc_labels[d].decrement(z[d][n]);
            }
            doc_foreback[d].decrement(y[d][n]);
        }

        ArrayList<Double> logprobs = new ArrayList<Double>();

        // foreground
        for (int k = 0; k < L; k++) {
            double lp = doc_foreback[d].getLogLikelihood(FOREGROUND)
                    + doc_foreback[d].getLogLikelihood(k)
                    + label_words[k].getLogLikelihood(words[d][n]);
            logprobs.add(lp);
        }

        // background
        double lp = doc_foreback[d].getLogLikelihood(BACKGROUND)
                + background.getLogLikelihood(words[d][n]);
        logprobs.add(lp);

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);

        // update assignments
        if (sampledIndex == L) {
            y[d][n] = BACKGROUND;
        } else {
            y[d][n] = FOREGROUND;
            z[d][n] = sampledIndex;
            doc_labels[d].increment(z[d][n]);

        }

        // debug
        if ((curY != y[d][n]) || (y[d][n] == FOREGROUND && curZ != z[d][n])) {
            numTokensChange++;
        } 

        // update counts
        if (addToModel) {
            if (y[d][n] == BACKGROUND) {
                background.increment(words[d][n]);
            } else {
                label_words[z[d][n]].increment(words[d][n]);
            }
        }
        if (addToData) {
            if (y[d][n] == FOREGROUND) {
                doc_labels[d].increment(z[d][n]);
            }
            doc_foreback[d].increment(y[d][n]);
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

        double docForeBackLlh = 0;
        for (int d = 0; d < D; d++) {
            docForeBackLlh += doc_foreback[d].getLogLikelihood();
        }

        double bgLlh = background.getLogLikelihood();

        double llh = docTopicLlh + topicWordLlh + docForeBackLlh + bgLlh;
        if (verbose) {
            logln(">>> doc-topic: " + MiscUtils.formatDouble(docTopicLlh)
                    + "\ttopic-word: " + MiscUtils.formatDouble(topicWordLlh)
                    + "\tdoc-foreback: " + MiscUtils.formatDouble(docForeBackLlh)
                    + "\tbackground: " + MiscUtils.formatDouble(bgLlh)
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
            this.doc_foreback[d].validate(msg);
        }

        for (int k = 0; k < L; k++) {
            this.label_words[k].validate(msg);
        }

        this.background.validate(msg);

        int totalCount = background.getCountSum();
        for (int k = 0; k < L; k++) {
            totalCount += label_words[k].getCountSum();
        }
        if (totalCount != numTokens) {
            throw new RuntimeException(msg + ". Total number of tokens mismatch. "
                    + totalCount + " vs. " + numTokens);
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
            modelStr.append(DirichletMultinomialModel.output(background)).append("\n");

            // data
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirichletMultinomialModel.output(doc_labels[d])).append("\n");

                // topics
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");

                // foreground/background
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(y[d][n]).append("\t");
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
            for (int k = 0; k < L; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                label_words[k] = DirichletMultinomialModel.input(reader.readLine());
            }
            background = DirichletMultinomialModel.input(reader.readLine());
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
                doc_labels[d] = DirichletMultinomialModel.input(reader.readLine());

                sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }

                sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    y[d][n] = Integer.parseInt(sline[n]);
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
        // background
        double[] bgDists = background.getDistribution();
        String[] bgTopWords = getTopWords(bgDists, numTopWords);
        writer.write("[Background"
                + ", " + background.getCountSum()
                + "]");
        for (String topWord : bgTopWords) {
            writer.write("\t" + topWord);
        }
        writer.write("\n\n");

        // foreground
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
