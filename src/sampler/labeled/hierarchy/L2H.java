package sampler.labeled.hierarchy;

import cc.mallet.types.Dirichlet;
import cc.mallet.util.Randoms;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import sampler.labeled.LabeledLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import taxonomy.AbstractTaxonomyBuilder;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.MimnoTopicCoherence;


/**
 *
 * @author vietan
 */
public class L2H extends AbstractSampler {

    public static Randoms randoms = new Randoms(1);
    public static final int INSIDE = 0;
    public static final int OUTSIDE = 1;
    // hyperparameter indices
    public static final int ALPHA = 0; // concentration parameter
    public static final int BETA = 1; // concentration parameter
    public static final int A_0 = 2;
    public static final int B_0 = 3;
    // inputs
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int V;    // vocab size
    protected int L;    // number of unique labels
    protected int D;    // number of documents
    // graph
    private SparseVector[] inWeights; // the weights of in-edges for each nodes
    // tree
    private AbstractTaxonomyBuilder treeBuilder;
    private Node root;
    private Node[] nodes;
    // latent variables
    private int[][] x;
    private int[][] z;
    private DirMult[] docSwitches;
    private SparseCount[] docLabelCounts;
    private Set<Integer>[] docMaskes;
    // configurations
    private PathAssumption pathAssumption;
    private boolean treeUpdated;
    private boolean sampleExact = false;
    // internal
    private HashMap<Integer, Set<Integer>> labelDocIndices;
    // information
    private ArrayList<String> labelVocab;
    private int numTokens;
    private int numTokensChange;
    private int numAccepts; // number of sampled nodes accepted
    private int[] labelFreqs;
    private double[] switchPrior;

    public void setLabelVocab(ArrayList<String> labelVocab) {
        this.labelVocab = labelVocab;
    }

    public void configure(L2H sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(A_0),
                sampler.hyperparams.get(B_0),
                sampler.treeBuilder,
                sampler.treeUpdated,
                sampler.sampleExact,
                sampler.initState,
                sampler.pathAssumption,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
        this.setWordVocab(sampler.wordVocab);
        this.setLabelVocab(sampler.labelVocab);
    }

    public void configure(String folder,
            int V,
            double alpha,
            double beta,
            double a0, double b0,
            AbstractTaxonomyBuilder treeBuilder,
            boolean treeUp,
            boolean sampleExact,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;

        this.treeBuilder = treeBuilder;
        this.labelVocab = treeBuilder.getLabelVocab();

        this.L = labelVocab.size();
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.hyperparams.add(a0);
        this.hyperparams.add(b0);

        this.switchPrior = new double[2];
        this.switchPrior[INSIDE] = a0;
        this.switchPrior[OUTSIDE] = b0;

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.treeUpdated = treeUp;
        this.sampleExact = sampleExact;

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInterval;

        this.initState = initState;
        this.pathAssumption = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- label vocab:\t" + L);
            logln("--- word vocab:\t" + V);

            logln("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            logln("--- beta:\t" + MiscUtils.formatDouble(beta));
            logln("--- a0:\t" + MiscUtils.formatDouble(a0));
            logln("--- b0:\t" + MiscUtils.formatDouble(b0));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- path assumption:\t" + pathAssumption);
            logln("--- tree builder:\t" + treeBuilder.getName());
            logln("--- updating tree?\t" + treeUpdated);
            logln("--- exact sampling?\t" + sampleExact);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_L2H")
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_opt-").append(this.paramOptimized)
                .append("_").append(this.pathAssumption);
        for (double hp : this.hyperparams) {
            str.append("-").append(MiscUtils.formatDouble(hp));
        }
        str.append("-").append(treeBuilder.getName());
        str.append("-").append(treeUpdated);
        str.append("-").append(sampleExact);
        this.name = str.toString();
    }

    public void train(int[][] newWords, int[][] newLabels) {
        this.words = newWords;
        this.labels = newLabels;
        this.D = this.words.length;
        this.labelDocIndices = new HashMap<Integer, Set<Integer>>();
        for (int d = 0; d < D; d++) {
            for (int ll : labels[d]) {
                Set<Integer> docIndices = this.labelDocIndices.get(ll);
                if (docIndices == null) {
                    docIndices = new HashSet<Integer>();
                }
                docIndices.add(d);
                this.labelDocIndices.put(ll, docIndices);
            }
        }

        int emptyDocCount = 0;
        this.numTokens = 0;
        this.labelFreqs = new int[L];
        for (int d = 0; d < D; d++) {
            if (labels[d].length == 0) {
                emptyDocCount++;
                continue;
            }
            this.numTokens += words[d].length;
            for (int ii = 0; ii < labels[d].length; ii++) {
                labelFreqs[labels[d][ii]]++;
            }
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # empty documents:\t" + emptyDocCount);
            logln("--- # tokens:\t" + numTokens);
        }
    }

    public void test(int[][] newWords) {
        this.words = newWords;
        this.labels = null;
        this.D = this.words.length;

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
        }
    }

    protected String getLabelString(int labelIdx) {
        return this.labelVocab.get(labelIdx) + " (" + this.labelFreqs[labelIdx] + ")";
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
            logln("Tree:\n" + this.printTree(10));
            logln("Tree structure:\n" + this.printTreeStructure());
        }
    }

    private void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        this.nodes = new Node[L];
        this.root = new Node(treeBuilder.getTreeRoot().getContent(), 0, 0,
                new SparseCount(), null);
        Stack<TreeNode<TreeNode, Integer>> stack = new Stack<TreeNode<TreeNode, Integer>>();
        stack.add(treeBuilder.getTreeRoot());
        while (!stack.isEmpty()) {
            TreeNode<TreeNode, Integer> node = stack.pop();
            for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                stack.add(child);
            }

            int labelIdx = node.getContent();

            // parents
            Node gParent = null;
            if (!node.isRoot()) {
                TreeNode<TreeNode, Integer> parent = node.getParent();
                int parentLabelIdx = parent.getContent();
                gParent = nodes[parentLabelIdx];
            }

            // global node
            Node gNode = new Node(labelIdx,
                    node.getIndex(),
                    node.getLevel(),
                    new SparseCount(),
                    gParent);
            nodes[labelIdx] = gNode;
            if (gParent != null) {
                gParent.addChild(gNode.getIndex(), gNode);
            }
            if (node.isRoot()) {
                root = gNode;
            }
        }
        estimateEdgeWeights(); // estimate edge weights
    }

    public void estimateEdgeWeights() {
        this.inWeights = new SparseVector[L];
        for (int ll = 0; ll < L; ll++) {
            this.inWeights[ll] = new SparseVector();
        }
        int[] labelFreq = new int[L];
        for (int dd = 0; dd < D; dd++) {
            for (int l : labels[dd]) {
                labelFreq[l]++;
            }
        }
        int maxLabelFreq = StatUtils.max(labelFreq);

        // pair frequencies
        for (int dd = 0; dd < D; dd++) {
            int[] docLabels = labels[dd];
            for (int ii = 0; ii < docLabels.length; ii++) {
                for (int jj = 0; jj < docLabels.length; jj++) {
                    if (ii == jj) {
                        continue;
                    }
                    Double weight = this.inWeights[docLabels[jj]].get(docLabels[ii]);
                    if (weight == null) {
                        this.inWeights[docLabels[jj]].set(docLabels[ii], 1.0);
                    } else {
                        this.inWeights[docLabels[jj]].set(docLabels[ii], weight + 1.0);
                    }
                }
            }
        }

        // root weights
        for (int l = 0; l < L; l++) {
            int lFreq = labelFreq[l];
            for (int ii : inWeights[l].getIndices()) {
                double weight = inWeights[l].get(ii) / lFreq;
                inWeights[l].set(ii, weight);
            }

            double selfWeight = (double) lFreq / maxLabelFreq;
            inWeights[l].set(L - 1, selfWeight);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }
        this.z = new int[D][];
        this.x = new int[D][];
        this.docSwitches = new DirMult[D];
        this.docLabelCounts = new SparseCount[D];
        this.docMaskes = new Set[D];

        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
            this.x[d] = new int[words[d].length];
            this.docSwitches[d] = new DirMult(new double[]{hyperparams.get(A_0),
                        hyperparams.get(B_0)});
            this.docLabelCounts[d] = new SparseCount();
            this.docMaskes[d] = new HashSet<Integer>();
            if (labels != null) { // if labels are given during training time
                updateMaskes(d);
            }
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- --- Initializing assignments. " + initState + " ...");
        }
        switch (initState) {
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
            logln("--- --- 1. Estimating topics using Labeled LDA");
            logln("--- --- 2. Forward sampling using the estimated topics");
        }
        int lda_burnin = 50;
        int lda_maxiter = 100;
        int lda_samplelag = 5;
        int lda_repInterval = 1;
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        LabeledLDA llda = new LabeledLDA();
        llda.setDebug(debug);
        llda.setVerbose(verbose);
        llda.setLog(false);
        llda.configure(folder,
                V, L, lda_alpha, lda_beta, initState, false,
                lda_burnin, lda_maxiter, lda_samplelag, lda_repInterval);
        // add the root label to all documents
        int[][] tempLabels = new int[D][];
        for (int dd = 0; dd < D; dd++) {
            tempLabels[dd] = new int[labels[dd].length + 1];
            System.arraycopy(labels[dd], 0, tempLabels[dd], 0, labels[dd].length);
            tempLabels[dd][labels[dd].length] = labelVocab.size() - 1;
        }
        llda.train(words, tempLabels);
        try {
            File lldaZFile = new File(llda.getSamplerFolderPath(), "init.zip");
            if (lldaZFile.exists()) {
                llda.inputState(lldaZFile);
            } else {
                IOUtils.createFolder(llda.getSamplerFolderPath());
                llda.sample();
                llda.outputState(lldaZFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while initializing topics "
                    + "with Labeled LDA");
        }
        setLog(true);

        DirMult[] tps = llda.getTopicWordDistributions();
        for (int ll = 0; ll < L; ll++) {
            nodes[ll].topic = tps[ll].getDistribution();
        }

        if (verbose) {
            logln("--- Initializing assignments by sampling ...");
        }
        long eTime;
        if (sampleExact) {
            eTime = sampleXZsExact(!REMOVE, ADD, !REMOVE, ADD);
        } else {
            eTime = sampleXZsMH(!REMOVE, ADD, !REMOVE, ADD);
        }
        if (verbose) {
            logln("--- --- Elapsed time: " + eTime);
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
            throw new RuntimeException("Exception while outputing to report folder");
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            // sampling x's and z's
            long sampleXZTime;
            if (sampleExact) {
                sampleXZTime = sampleXZsExact(REMOVE, ADD, REMOVE, ADD);
            } else {
                sampleXZTime = sampleXZsMH(REMOVE, ADD, REMOVE, ADD);
            }

            // sampling topics
            long sampleTopicTime = sampleTopics();

            // updating tree
            long updateTreeTime = 0;
            if (treeUpdated) {
                updateTreeTime = updateTree();
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t # tokens changed: " + numTokensChange
                        + " (" + MiscUtils.formatDouble((double) numTokensChange / numTokens) + ")"
                        + "\t # accepts: " + numAccepts
                        + " (" + MiscUtils.formatDouble((double) numAccepts / L) + ")"
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
                logln("--- Elapsed time: sXZs: " + sampleXZTime
                        + "\tsTopic: " + sampleTopicTime
                        + "\tuTree: " + updateTreeTime);
                System.out.println();
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

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(getReportFolderPath(), "iter-" + iter + ".zip"));
            }
        }

        if (report) {
            outputState(new File(getReportFolderPath(), "iter-" + iter + ".zip"));
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Update the structure of the tree. This is done by (1) proposing a new
     * parent node for each node in the tree using the edge weights of the
     * background graph, (2) accepting or rejecting the proposed new parent
     * using Metropolis-Hastings.
     */
    private long updateTree() {
        long sTime = System.currentTimeMillis();

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("--- Updating tree ...");
        }

        numAccepts = 0;
        for (int ll = 0; ll < L; ll++) {

            Node node = nodes[ll];
            if (node.isRoot()) {
                continue;
            }

            // current and new parents
            Node currentParent = node.getParent();
            Node proposeParent = proposeParent(node);

            if (proposeParent.equals(node.getParent())) { // if the same node, move on
                numAccepts++;
                continue;
            }

            Set<Integer> subtreeDocs = getSubtreeDocumentIndices(node);
            Set<Integer> subtreeNodes = node.getSubtree();

            // current x & z log prob
            double curXLogprob = 0.0;
            double curZLogprob = 0.0;
            for (int d : subtreeDocs) {
                curXLogprob += docSwitches[d].getLogLikelihood();
                curZLogprob += computeDocLabelLogprob(docLabelCounts[d], docMaskes[d]);
            }

            // phi
            double curPhiLogprob = computeWordLogprob(node, currentParent);
            double newPhiLogprob = computeWordLogprob(node, proposeParent);
            double newXLogprob = 0.0;
            double newZLogprob = 0.0;
            HashMap<Integer, Set<Integer>> proposedMasks = new HashMap<Integer, Set<Integer>>();
            for (int d : subtreeDocs) {
                Set<Integer> proposedMask = getProposedMask(d, node.id, subtreeNodes, proposeParent);
                newZLogprob += computeDocLabelLogprob(docLabelCounts[d], proposedMask);
                proposedMasks.put(d, proposedMask);

                int[] proposedSwitchCount = new int[2];
                for (int n = 0; n < words[d].length; n++) {
                    if (proposedMask.contains(z[d][n])) {
                        proposedSwitchCount[INSIDE]++;
                    } else {
                        proposedSwitchCount[OUTSIDE]++;
                    }
                }
                newXLogprob += SamplerUtils.computeLogLhood(proposedSwitchCount,
                        words[d].length, switchPrior);
            }

            double curLogprob = curPhiLogprob + curXLogprob + curZLogprob;
            double newLogprob = newPhiLogprob + newXLogprob + newZLogprob;
            double mhRatio = Math.exp(newLogprob - curLogprob);

            if (rand.nextDouble() < mhRatio) {
                numAccepts++;

                // update parent
                currentParent.removeChild(node.getIndex());
                int newIndex = proposeParent.getNextChildIndex();
                node.setIndex(newIndex);
                proposeParent.addChild(newIndex, node);
                node.setParent(proposeParent);

                // update level of nodes in the subtree
                for (int n : subtreeNodes) {
                    nodes[n].setLevel(nodes[n].getLevel()
                            - currentParent.getLevel()
                            + proposeParent.getLevel());
                }

//                if (verbose && debug) {
//                    logln("Accept. iter = " + iter
//                            + ". ratio: " + mhRatio
//                            + ". node " + node.toString()
//                            + ". " + labelVocab.get(node.id));
//                    logln("Old parent: " + currentParent.toString() + ". " + labelVocab.get(currentParent.id));
//                    logln("New parent: " + proposeParent.toString() + ". " + labelVocab.get(proposeParent.id));
//                    String[] nodeTopWords = getTopWords(node.topic, 15);
//                    String[] curParentTopWords = getTopWords(currentParent.topic, 15);
//                    String[] newParentTopWords = getTopWords(proposeParent.topic, 15);
//                    System.out.println("Node: " + MiscUtils.arrayToString(nodeTopWords));
//                    System.out.println("Cur parent: " + MiscUtils.arrayToString(curParentTopWords));
//                    System.out.println("New parent: " + MiscUtils.arrayToString(newParentTopWords));
//                    logln("Current: " + MiscUtils.formatDouble(curPhiLogprob)
//                            + ". " + MiscUtils.formatDouble(curXLogprob)
//                            + ". " + MiscUtils.formatDouble(curZLogprob)
//                            + ". " + MiscUtils.formatDouble(curLogprob));
//                    logln("Propose: " + MiscUtils.formatDouble(newPhiLogprob)
//                            + ". " + MiscUtils.formatDouble(newXLogprob)
//                            + ". " + MiscUtils.formatDouble(newZLogprob)
//                            + ". " + MiscUtils.formatDouble(newLogprob));
//                    System.out.println();
//                }

                // remove current switch assignments
                for (int d : subtreeDocs) {
                    docMaskes[d] = proposedMasks.get(d);
                    for (int n = 0; n < words[d].length; n++) {
                        docSwitches[d].decrement(x[d][n]); // decrement

                        // update
                        if (docMaskes[d].contains(z[d][n])) {
                            x[d][n] = INSIDE;
                        } else {
                            x[d][n] = OUTSIDE;
                        }
                        docSwitches[d].increment(x[d][n]); // increment
                    }
                }
            }

            if (debug) {
                validate("Update node " + ll + ". " + node.toString());
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    private double computeWordLogprob(Node node, Node parent) {
        SparseCount obs = new SparseCount();
        for (int v : node.getContent().getIndices()) {
            obs.changeCount(v, node.getContent().getCount(v));
        }
        for (int v : node.pseudoCounts.getIndices()) {
            obs.changeCount(v, node.pseudoCounts.getCount(v));
        }
        return SamplerUtils.computeLogLhood(obs, parent.topic, hyperparams.get(BETA));
    }

    /**
     * Compute the log probability of the topic assignments of a document given
     * a candidate set.
     *
     * @param docLabelCount Store the number of times tokens in this document
     * assigned to each topic
     * @param docMask The candidate set
     */
    private double computeDocLabelLogprob(SparseCount docLabelCount, Set<Integer> docMask) {
        double priorVal = hyperparams.get(ALPHA);
        double logGammaPriorVal = SamplerUtils.logGammaStirling(priorVal);

        double insideLp = 0.0;
        insideLp += SamplerUtils.logGammaStirling(priorVal * docMask.size());
        insideLp -= docMask.size() * logGammaPriorVal;


        double outsideLp = 0.0;
        outsideLp += SamplerUtils.logGammaStirling(priorVal * (L - docMask.size()));
        outsideLp -= (L - docMask.size()) * logGammaPriorVal;

        int insideCountSum = 0;
        int outsideCountSum = 0;
        for (int ll : docLabelCount.getIndices()) {
            int count = docLabelCount.getCount(ll);

            if (docMask.contains(ll)) {
                insideLp += SamplerUtils.logGammaStirling(count + priorVal);
                insideCountSum += count;
            } else {
                outsideLp += SamplerUtils.logGammaStirling(count + priorVal);
                outsideCountSum += count;
            }
        }

        insideLp -= SamplerUtils.logGammaStirling(insideCountSum + priorVal * docMask.size());
        outsideLp -= SamplerUtils.logGammaStirling(outsideCountSum + priorVal * (L - docMask.size()));

        double logprob = insideLp + outsideLp;
        return logprob;
    }

    /**
     * Return the set of documents whose label set contains any label in the
     * subtree rooted at a given node.
     *
     * @param node The root of the subtree
     */
    private Set<Integer> getSubtreeDocumentIndices(Node node) {
        Set<Integer> docIdx = new HashSet<Integer>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(node);
        while (!stack.isEmpty()) {
            Node n = stack.pop();

            for (Node c : n.getChildren()) {
                stack.add(c);
            }

            for (int d : this.labelDocIndices.get(n.id)) {
                docIdx.add(d);
            }
        }
        return docIdx;
    }

    /**
     * Return the set of mask node if the subtree root node become a child of
     * the a proposed parent node.
     *
     * @param d Document index
     * @param subtreeRoot The ID of the root of the subtree
     * @param subtree Set of nodes in the subtree
     * @param proposedParent The proposed parent node
     */
    private Set<Integer> getProposedMask(int d,
            int subtreeRoot,
            Set<Integer> subtree,
            Node proposedParent) {
        Set<Integer> ppMask = new HashSet<Integer>();
        boolean insideSubtree = false;
        for (int label : labels[d]) {
            Node n = nodes[label];

            // if this label is inside the subtree, add all nodes from the label
            // node to the subtree root to the mask
            if (subtree.contains(label)) {
                while (n.id != subtreeRoot) {
                    ppMask.add(n.id);
                    n = n.getParent();
                }
                ppMask.add(subtreeRoot);
                insideSubtree = true;
            } // if this label is outside the subtree, all all nodes from the label
            // node to the root as usual
            else {
                while (n != null) {
                    ppMask.add(n.id);
                    n = n.getParent();
                }
            }
        }

        // if there is any label inside the subtree, add nodes from the proposed
        // label to the root
        if (insideSubtree) {
            Node n = nodes[proposedParent.id];
            while (n != null) {
                ppMask.add(n.id);
                n = n.getParent();
            }
        }

        // debug
//        System.out.println("\n");
//        for (int ll : labels[d]) {
//            System.out.println("label: " + nodes[ll].toString());
//        }
//        System.out.println("Subtree root: " + subtreeRoot + ". " + nodes[subtreeRoot].toString());
//        System.out.println("Propose parent: " + proposedParent.toString());
//        for (int ii : subtree) {
//            System.out.println("--- subtree node: " + nodes[ii].toString());
//        }
//
//        System.out.println("Proposed mask");
//        for (int ii : ppMask) {
//            System.out.println(">>> ppm node: " + nodes[ii].toString());
//        }

        return ppMask;
    }

    /**
     * Propose a parent node for a given node by sampling from the prior weights
     * (the MLE conditional probabilities)
     *
     * @param node A node
     */
    private Node proposeParent(Node node) {
        // sort candidate parents
        ArrayList<RankingItem<Node>> rankCandNodes = new ArrayList<RankingItem<Node>>();
        for (int idx : inWeights[node.id].getIndices()) {
            Node candNode = nodes[idx];
            if (node.isDescendent(candNode)) {
                continue;
            }
            double weight = inWeights[node.id].get(idx);
            rankCandNodes.add(new RankingItem<Node>(nodes[idx], weight));
        }
        Collections.sort(rankCandNodes);

        // sample from a limited set
        ArrayList<Node> candNodes = new ArrayList<Node>();
        ArrayList<Double> candWeights = new ArrayList<Double>();
        int numCands = Math.min(10, rankCandNodes.size());
        for (int ii = 0; ii < numCands; ii++) {
            RankingItem<Node> rankNode = rankCandNodes.get(ii);
            candNodes.add(rankNode.getObject());
            candWeights.add(rankNode.getPrimaryValue());
        }
        int sampledIdx = SamplerUtils.scaleSample(candWeights);
        Node sampledNode = candNodes.get(sampledIdx);

        // debug
//        int ll = node.id;
//        System.out.println("ll = " + ll
//                + "\t size: " + candNodes.size()
//                + "\t" + nodes[ll].toString()
//                + "\t" + labelVocab.get(ll)
//                + "\tcurpar: " + labelVocab.get(nodes[ll].getParent().id)
//                + "\t" + inWeights[node.id].get(nodes[ll].getParent().id));
//        for (int ii = 0; ii < candNodes.size(); ii++) {
//            System.out.println("cand " + ii
//                    + "\t" + candNodes.get(ii).toString()
//                    + "\t" + candWeights.get(ii)
//                    + "\t" + labelVocab.get(candNodes.get(ii).id));
//        }
//        System.out.println(">>> sampledIdx: " + sampledIdx
//                + "\t" + sampledNode.toString()
//                + "\t" + labelVocab.get(sampledNode.id)
//                + "\t" + candWeights.get(sampledIdx));
//        System.out.println();

        return sampledNode;
    }

    /**
     * Sample x and z together for all documents.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private long sampleXZsExact(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        numTokensChange = 0;
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleXZExact(d, n, removeFromModel, addToModel, removeFromData, addToData);
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample x and z together for a token.
     *
     * @param d
     * @param n
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private void sampleXZExact(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            nodes[z[d][n]].getContent().decrement(words[d][n]);
            nodes[z[d][n]].removeToken(d, n);
        }
        if (removeFromData) {
            docSwitches[d].decrement(x[d][n]);
            docLabelCounts[d].decrement(z[d][n]);
        }

        double[] logprobs = new double[L];
        for (int ll = 0; ll < L; ll++) {
            boolean inside = docMaskes[d].contains(ll);
            double xLlh;
            double zLlh;
            double wLlh = Math.log(nodes[ll].topic[words[d][n]]);

            if (inside) {
                xLlh = Math.log(docSwitches[d].getCount(INSIDE) + hyperparams.get(A_0));
                zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * docMaskes[d].size()));
            } else {
                xLlh = Math.log(docSwitches[d].getCount(OUTSIDE) + hyperparams.get(B_0));
                zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(OUTSIDE) + hyperparams.get(ALPHA) * (L - docMaskes[d].size())));
            }
            logprobs[ll] = xLlh + zLlh + wLlh;
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledZ != z[d][n]) {
            numTokensChange++;
        }
        z[d][n] = sampledZ;
        if (docMaskes[d].contains(z[d][n])) {
            x[d][n] = INSIDE;
        } else {
            x[d][n] = OUTSIDE;
        }

        if (addToModel) {
            nodes[z[d][n]].getContent().increment(words[d][n]);
            nodes[z[d][n]].addToken(d, n);
        }

        if (addToData) {
            docSwitches[d].increment(x[d][n]);
            docLabelCounts[d].increment(z[d][n]);
        }
    }

    /**
     * Sample x and z. This is done by first sampling x, and given the value of
     * x, sample z. This is an approximation of sampleXZsExact.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private long sampleXZsMH(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        numTokensChange = 0;
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleXZMH(d, n, removeFromModel, addToModel, removeFromData, addToData);
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample x and z. This is done by first sampling x, and given the value of
     * x, sample z. This is an approximation of sampleXZsExact.
     *
     * @param d
     * @param n
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private void sampleXZMH(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            nodes[z[d][n]].getContent().decrement(words[d][n]);
            nodes[z[d][n]].removeToken(d, n);
        }
        if (removeFromData) {
            docSwitches[d].decrement(x[d][n]);
            docLabelCounts[d].decrement(z[d][n]);
        }

        // propose
        int pX;
        if (!docMaskes[d].isEmpty()) {
            double[] ioLogProbs = new double[2];
            ioLogProbs[INSIDE] = docSwitches[d].getCount(INSIDE) + hyperparams.get(A_0);
            ioLogProbs[OUTSIDE] = docSwitches[d].getCount(OUTSIDE) + hyperparams.get(B_0);
            pX = SamplerUtils.scaleSample(ioLogProbs);
        } else { // if candidate set is empty
            pX = OUTSIDE;
        }
        int pZ = proposeZ(d, n, pX);

        // compute MH ratio: accept all for now
        if (pZ != z[d][n]) {
            numTokensChange++;
        }
        z[d][n] = pZ;
        x[d][n] = pX;
        // accept or reject

        if (docMaskes[d].contains(z[d][n])) {
            x[d][n] = INSIDE;
        } else {
            x[d][n] = OUTSIDE;
        }

        if (addToModel) {
            nodes[z[d][n]].getContent().increment(words[d][n]);
            nodes[z[d][n]].addToken(d, n);
        }

        if (addToData) {
            docSwitches[d].increment(x[d][n]);
            docLabelCounts[d].increment(z[d][n]);
        }
    }

    /**
     * Sample a node for a token given the binary indicator x.
     *
     * @param d Document index
     * @param n Token index
     * @param pX Binary indicator specifying whether the given token should be
     * assigned to a node in the candidate set or not.
     */
    private int proposeZ(int d, int n, int pX) {
        ArrayList<Integer> indices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();
        if (pX == INSIDE) {
            for (int ll : docMaskes[d]) {
                double zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * docMaskes[d].size()));
                double wLlh = Math.log(nodes[ll].topic[words[d][n]]);
                logprobs.add(zLlh + wLlh);
                indices.add(ll);
            }
        } else {
            for (int ll = 0; ll < L; ll++) {
                if (docMaskes[d].contains(ll)) {
                    continue;
                }
                double zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * (L - docMaskes[d].size())));
                double wLlh = Math.log(nodes[ll].topic[words[d][n]]);
                logprobs.add(zLlh + wLlh);
                indices.add(ll);
            }
        }
        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
        return indices.get(sampledIdx);
    }

    /**
     * Sample topics (distributions over words) in the tree. This is done by (1)
     * performing a bottom-up smoothing to compute the pseudo-counts from
     * children for each node, and (2) top-down sampling to get the topics.
     */
    private long sampleTopics() {
        if (verbose && iter % REP_INTERVAL == 0) {
            logln("--- Sampling topics ...");
        }
        long sTime = System.currentTimeMillis();
        // get all leaves of the tree
        ArrayList<Node> leaves = new ArrayList<Node>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getChildren().isEmpty()) {
                leaves.add(node);
            }
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // bottom-up smoothing to compute pseudo-counts from children
        Queue<Node> queue = new LinkedList<Node>();
        for (Node leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            Node parent = node.getParent();
            if (!node.isRoot() && !queue.contains(parent)) {
                queue.add(parent);
            }
            if (node.isLeaf()) {
                continue;
            }
            if (this.pathAssumption == PathAssumption.MINIMAL) {
                node.getPseudoCountsFromChildrenMin();
            } else if (this.pathAssumption == PathAssumption.MAXIMAL) {
                node.getPseudoCountsFromChildrenMax();
            } else {
                throw new RuntimeException("Path assumption " + this.pathAssumption
                        + " is not supported.");
            }
        }

        // top-down sampling to get topics
        queue = new LinkedList<Node>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            for (Node child : node.getChildren()) {
                queue.add(child);
            }
            node.sampleTopic();
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Update the set of candidate labels for a document d. This set is defined
     * based on the set of actual document labels and the topic tree structure.
     *
     * @param d Document index
     */
    private void updateMaskes(int d) {
        if (labels[d].length > 0) {
            this.docMaskes[d] = new HashSet<Integer>();
            for (int label : labels[d]) {
                Node node = nodes[label];
                while (node != null) {
                    docMaskes[d].add(node.id);
                    node = node.getParent();
                }
            }
        }
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int ll = 0; ll < L; ll++) {
            wordLlh += nodes[ll].getWordLogLikelihood();
        }

        double docSwitchesLlh = 0.0;
        for (int dd = 0; dd < D; dd++) {
            docSwitchesLlh += docSwitches[dd].getLogLikelihood();
        }

        double docLabelLlh = 0.0;
        for (int dd = 0; dd < D; dd++) {
            // inside
            if (!docMaskes[dd].isEmpty()) {
                int[] insideCounts = new int[docMaskes[dd].size()];
                int insideCountSum = 0;
                int ii = 0;
                for (int ll : docMaskes[dd]) {
                    insideCounts[ii++] = docLabelCounts[dd].getCount(ll);
                    insideCountSum += docLabelCounts[dd].getCount(ll);
                }
                double insideLlh = SamplerUtils.computeLogLhood(insideCounts,
                        insideCountSum, hyperparams.get(ALPHA));
                docLabelLlh += insideLlh;
            }

            // outside
            int[] outsideCounts = new int[L - docMaskes[dd].size()];
            int outsideCountSum = 0;
            int ii = 0;
            for (int ll = 0; ll < L; ll++) {
                if (docMaskes[dd].contains(ll)) {
                    continue;
                }
                outsideCounts[ii++] = docLabelCounts[dd].getCount(ll);
                outsideCountSum += docLabelCounts[dd].getCount(ll);
            }
            double outsideLlh = SamplerUtils.computeLogLhood(outsideCounts,
                    outsideCountSum, hyperparams.get(ALPHA));
            docLabelLlh += outsideLlh;
        }

        double treeLp = 0.0;
        for (int ll = 0; ll < L; ll++) {
            Node node = nodes[ll];
            if (node.isRoot()) {
                continue;
            }
            treeLp += Math.log(inWeights[ll].get(node.getParent().id));
        }

        logln(">>> >>> word: " + MiscUtils.formatDouble(wordLlh)
                + ". switch: " + MiscUtils.formatDouble(docSwitchesLlh)
                + ". label: " + MiscUtils.formatDouble(docLabelLlh)
                + ". tree: " + MiscUtils.formatDouble(treeLp));
        return wordLlh + docSwitchesLlh + docLabelLlh + treeLp;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath();
    }

    @Override
    public void validate(String msg) {
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        int numNodes = 0;
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            numNodes++;
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.validate(msg);
        }

        if (numNodes != L) {
            throw new RuntimeException(msg + ". Number of connected nodes: "
                    + numNodes + ". L = " + L);
        }

        for (int d = 0; d < D; d++) {
            docSwitches[d].validate(msg);
            docLabelCounts[d].validate(msg);

            if (labels[d].length > 0) {
                HashSet<Integer> tempDocMask = new HashSet<Integer>();
                for (int label : labels[d]) {
                    Node node = nodes[label];
                    while (node != null) {
                        tempDocMask.add(node.id);
                        node = node.getParent();
                    }
                }

                if (tempDocMask.size() != docMaskes[d].size()) {
                    for (int ll : labels[d]) {
                        System.out.println("label " + ll + "\t" + nodes[ll].toString());
                    }
                    System.out.println();
                    for (int ii : tempDocMask) {
                        System.out.println("true " + ii + "\t" + nodes[ii].toString());
                    }
                    System.out.println();
                    for (int ii : docMaskes[d]) {
                        System.out.println("actu " + ii + "\t" + nodes[ii].toString());
                    }
                    throw new RuntimeException(msg + ". Mask sizes mismatch. "
                            + tempDocMask.size() + " vs. " + docMaskes[d].size()
                            + " in document " + d);
                }
            }
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
            for (int k = 0; k < nodes.length; k++) {
                Node node = nodes[k];
                modelStr.append(k)
                        .append("\t").append(node.getIndex())
                        .append("\t").append(node.getLevel())
                        .append("\n");
                for (int v = 0; v < node.topic.length; v++) {
                    modelStr.append(node.topic[v]).append("\t");
                }
                modelStr.append("\n");
            }

            for (int k = 0; k < nodes.length; k++) {
                modelStr.append(k);
                if (nodes[k].getParent() == null) {
                    modelStr.append("\t-1\n");
                } else {
                    modelStr.append("\t").append(nodes[k].getParent().id).append("\n");
                }
            }

            for (int k = 0; k < nodes.length; k++) {
                modelStr.append(k).append("\t")
                        .append(SparseVector.output(inWeights[k])).append("\n");
            }

            // data
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docSwitches[d])).append("\n");
                assignStr.append(SparseCount.output(docLabelCounts[d])).append("\n");
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(x[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing state to "
                    + filepath);
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
            throw new RuntimeException("Excepion while inputing state from "
                    + filepath);
        }
    }

    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            // initialize
            this.nodes = new Node[L];
//            this.inWeights = new SparseVector[L];
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);

            for (int k = 0; k < L; k++) {
                // node ids
                String[] sline = reader.readLine().split("\t");
                int id = Integer.parseInt(sline[0]);
                if (id != k) {
                    throw new RuntimeException("Mismatch");
                }
                int idx = Integer.parseInt(sline[1]);
                int level = Integer.parseInt(sline[2]);
                nodes[k] = new Node(id, idx, level, null, null);

                // node topic
                sline = reader.readLine().split("\t");
                double[] topic = new double[V];
                if (sline.length != V) {
                    throw new RuntimeException("Mismatch: " + sline.length + " vs. " + V);
                }
                for (int v = 0; v < V; v++) {
                    topic[v] = Double.parseDouble(sline[v]);
                }
                nodes[k].topic = topic;
            }

            // tree structure
            for (int k = 0; k < L; k++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != k) {
                    throw new RuntimeException("Mismatch");
                }
                int parentId = Integer.parseInt(sline[1]);
                if (parentId == -1) {
                    root = nodes[k];
                } else {
                    nodes[k].setParent(nodes[parentId]);
                    nodes[parentId].addChild(nodes[k].getIndex(), nodes[k]);
                }
            }
            for (int k = 0; k < L; k++) {
                nodes[k].fillInactiveChildIndices();
            }

            // edge weight
//            String line;
//            int count = 0;
//            while((line = reader.readLine()) != null) {
//                String[] sline = line.split("\t");
//                if (Integer.parseInt(sline[0]) != count) {
//                    throw new RuntimeException("Mismatch");
//                }
//                this.inWeights[count] = SparseVector.input(sline[1]);
//                count ++;
//            }

//            for (int k = 0; k < L; k++) {
//                String[] sline = reader.readLine().split("\t");
//                if (Integer.parseInt(sline[0]) != k) {
//                    throw new RuntimeException("Mismatch");
//                }
//                this.inWeights[k] = SparseVector.input(sline[1]);
//            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }

        if (verbose) {
            logln("--- Model loaded.\n" + printTreeStructure());
        }
    }

    private void inputAssignments(String zipFilepath) {
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
                docSwitches[d] = DirMult.input(reader.readLine());
                docLabelCounts[d] = SparseCount.input(reader.readLine());
                String[] sline = reader.readLine().trim().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("Mismatch");
                }
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
                sline = reader.readLine().trim().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("Mismatch");
                }
                for (int n = 0; n < words[d].length; n++) {
                    x[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Print out the structure (number of observations at each level) of the
     * global tree.
     */
    public String printTreeStructure() {
        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        SparseCount nodeCountPerLevel = new SparseCount();      // nodes
        SparseCount obsCountPerLevel = new SparseCount();       // observation

        while (!stack.isEmpty()) {
            Node node = stack.pop();
            // node
            nodeCountPerLevel.increment(node.getLevel());
            // observation
            if (node.getContent() != null) {
                obsCountPerLevel.changeCount(node.getLevel(),
                        node.getContent().getCountSum());
            }
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int level : nodeCountPerLevel.getSortedIndices()) {
            str.append(">>> level ").append(level)
                    .append(". n: ").append(nodeCountPerLevel.getCount(level))
                    .append(". o: ").append(obsCountPerLevel.getCount(level))
                    .append(" (").append((double) obsCountPerLevel.getCount(level)
                    / nodeCountPerLevel.getCount(level)).append(")")
                    .append("\n");
        }
        str.append(">>> >>> # nodes: ").append(nodeCountPerLevel.getCountSum()).append("\n");
        str.append(">>> >>> # obs  : ").append(obsCountPerLevel.getCountSum()).append("\n");
        return str.toString();
    }

    /**
     * Print out the global tree.
     *
     * @param numTopWords Number of top words shown for each topic
     */
    public String printTree(int numTopWords) {
        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        int totalObs = 0;
        int numNodes = 0;
        SparseCount nodeCountPerLevel = new SparseCount(); // nodes
        SparseCount obsCountPerLevel = new SparseCount();  // observations

        while (!stack.isEmpty()) {
            Node node = stack.pop();
            numNodes++;
            // node
            nodeCountPerLevel.increment(node.getLevel());

            // observation
            if (node.getContent() != null) {
                obsCountPerLevel.changeCount(node.getLevel(),
                        node.getContent().getCountSum());
                totalObs += node.getContent().getCountSum();
            }

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }

            String[] topWords = null;
            if (node.topic != null) {
                topWords = getTopWords(node.topic, numTopWords);
            }

            str.append(node.toString()).append(", ")
                    .append(getLabelString(node.id))
                    .append(" ").append(node.getContent() == null ? "" : node.getContent().getCountSum())
                    .append(" ").append(topWords == null ? "" : Arrays.toString(topWords))
                    .append("\n\n");

            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # nodes = ").append(numNodes)
                .append("\n");
        for (int level : nodeCountPerLevel.getSortedIndices()) {
            str.append(">>> level ").append(level)
                    .append(". n: ").append(nodeCountPerLevel.getCount(level))
                    .append(". o: ").append(obsCountPerLevel.getCount(level))
                    .append("\n");
        }
        return str.toString();
    }

    public void outputGlobalTree(File outputFile, int numTopWords) throws Exception {
        if (verbose) {
            logln("Outputing global tree to " + outputFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write(this.printTree(numTopWords));
        writer.close();
    }

    public void outputLexicalCorrelations(File outputFile) throws Exception {
        if (verbose) {
            logln("Outputing lexical correlations to " + outputFile);
        }

        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        double[][] avgTopics = new double[L - 1][V];
        int numModels = 0;
        try {
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }
                inputState(new File(reportFolder, filename).getAbsolutePath());
                for (int ll = 0; ll < L - 1; ll++) {
                    double[] topic = nodes[ll].topic;
                    for (int v = 0; v < V; v++) {
                        avgTopics[ll][v] += topic[v];
                    }
                }
                numModels++;
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing label "
                    + "correlation to " + outputFile);
        }

        for (int ll = 0; ll < avgTopics.length; ll++) {
            for (int v = 0; v < V; v++) {
                avgTopics[ll][v] /= numModels;
            }
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int v = 0; v < V; v++) {
            writer.write(v + "," + v + ":1 ");
            for (int u = 0; u < v; u++) {
                double score = 0.0;
                for (int ll = 0; ll < avgTopics.length; ll++) {
                    score += avgTopics[ll][v] * avgTopics[ll][u];
                }
                writer.write((u + L) + "," + (v + L) + ":" + score + " ");
                writer.write((v + L) + "," + (u + L) + ":" + score + " ");
            }
        }
        writer.write("\n");
        writer.close();
    }

    public void outputLabelCorrelations(File outputFile) throws Exception {
        if (verbose) {
            logln("Outputing label correlations to " + outputFile);
        }
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        double[][] avgTopics = new double[L - 1][V];
        int numModels = 0;
        try {
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }
                inputState(new File(reportFolder, filename).getAbsolutePath());
                for (int ll = 0; ll < L - 1; ll++) {
                    double[] topic = nodes[ll].topic;
                    for (int v = 0; v < V; v++) {
                        avgTopics[ll][v] += topic[v];
                    }
                }
                numModels++;
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing label "
                    + "correlation to " + outputFile);
        }

        for (int ll = 0; ll < avgTopics.length; ll++) {
            for (int v = 0; v < V; v++) {
                avgTopics[ll][v] /= numModels;
            }
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int ii = 0; ii < L - 1; ii++) {
            writer.write(ii + "," + ii + ":1 ");
            for (int jj = 0; jj < ii; jj++) {
                double cosine = StatUtils.cosineSimilarity(avgTopics[ii], avgTopics[jj]);
                if (cosine > 0.1) {
                    writer.write(ii + "," + jj + ":" + cosine + " ");
                    writer.write(jj + "," + ii + ":" + cosine + " ");
                }
//                double klDivij = StatisticsUtils.KLDivergenceAsymmetric(nodes[ii].topic, nodes[jj].topic);
//                double klDivji = StatisticsUtils.KLDivergenceAsymmetric(nodes[jj].topic, nodes[ii].topic);
//                writer.write(ii + "," + jj + ":" + klDivij + " ");
//                writer.write(jj + "," + ii + ":" + klDivji + " ");
            }
        }
        writer.write("\n");
        writer.close();
    }

    // ******************* Start prediction ************************************
    /**
     * Sample test documents in parallel.
     *
     * @param newWords Test document
     * @param numLabels Number of labels used for prediction
     * @param iterPredFolder Folder to store predictions using different models
     * @param sampler The configured sampler
     * @param initPredictions Initial predictions from TF-IDF
     * @param topK The number of nearest neighbors to be initially included in
     * the candidate set
     */
    public static void parallelTest(int[][] newWords, File iterPredFolder, L2H sampler,
            double[][] initPredictions, int topK) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                // folder contains multiple samples during test using a learned model
                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");

                L2HTestRunner runner = new L2HTestRunner(sampler,
                        newWords, stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath(),
                        initPredictions, topK);
                Thread thread = new Thread(runner);
                threads.add(thread);
            }
            runThreads(threads); // run MAX_NUM_PARALLEL_THREADS threads at a time
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
    }

    /**
     * Perform sampling for test documents to predict labels.
     *
     * @param stateFile File containing learned model
     * @param newWords Test document
     * @param outputSamplingFolder Output file
     */
    public void sampleNewDocuments(String stateFile,
            int[][] newWords,
            String outputResultFile,
            double[][] initPredictions,
            int topK) {
        if (verbose) {
            System.out.println();
            logln("Perform prediction using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        // test data
        this.words = newWords;
        this.labels = null;
        this.D = this.words.length;

        // initialize data structure
        this.z = new int[D][];
        this.x = new int[D][];
        this.docSwitches = new DirMult[D];
        this.docLabelCounts = new SparseCount[D];
        this.docMaskes = new Set[D];

        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
            this.x[d] = new int[words[d].length];
            this.docSwitches[d] = new DirMult(
                    new double[]{hyperparams.get(A_0), hyperparams.get(B_0)});
            this.docLabelCounts[d] = new SparseCount();
            this.docMaskes[d] = new HashSet<Integer>();

            Set<Integer> cands = getCandidates(initPredictions[d], topK);
            for (int label : cands) {
                Node node = nodes[label];
                while (node != null) {
                    docMaskes[d].add(node.id);
                    node = node.getParent();
                }
            }
        }

        // initialize: sampling using global distribution over labels
        if (verbose) {
            logln("--- Sampling on test data ...");
        }
        double[][] predictedScores = new double[D][L - 1]; // exclude root
        int count = 0;
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId()
                        + "\t" + getSamplerFolderPath());
            }
            if (iter == 0) {
                sampleXZsMH(!REMOVE, !ADD, !REMOVE, ADD);
            } else {
                sampleXZsMH(!REMOVE, !ADD, REMOVE, ADD);
            }

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                for (int dd = 0; dd < D; dd++) {
                    for (int ll = 0; ll < L - 1; ll++) {
                        predictedScores[dd][ll] +=
                                (double) docLabelCounts[dd].getCount(ll) / words[dd].length;
                    }
                }
                count++;
            }
        }

        // output result during test time
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        for (int dd = 0; dd < D; dd++) {
            for (int ll = 0; ll < L - 1; ll++) {
                predictedScores[dd][ll] /= count;
            }
        }
        PredictionUtils.outputSingleModelClassifications(new File(outputResultFile),
                predictedScores);
    }

    /**
     * TODO: add other ways to get the candidate set.
     */
    private Set<Integer> getCandidates(double[] scores, int topK) {
        Set<Integer> cands = new HashSet<Integer>();
        ArrayList<RankingItem<Integer>> docRankLabels = MiscUtils.getRankingList(scores);
        for (int ii = 0; ii < topK; ii++) {
            cands.add(docRankLabels.get(ii).getObject());
        }
        return cands;
    }

    private Set<Integer> getCandidatesNew(double[] scores) {
        Set<Integer> cands = new HashSet<Integer>();
        ArrayList<RankingItem<Integer>> docRankLabels = MiscUtils.getRankingList(scores);
        int maxNum = 10;
        double minSimScore = 0.25;
        int ii = 0;
        while (true) {
            RankingItem<Integer> item = docRankLabels.get(ii++);
            if (item.getPrimaryValue() < minSimScore) {
                break;
            }
            cands.add(item.getObject());
            if (cands.size() == maxNum) {
                break;
            }
        }
        return cands;
    }

    public double[][] getTrainingExtendedLexiconFeatureVectors(int nTopWords) {
        double[][] featVecs = null;
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        double[][] thetas = new double[D][L - 1];
        double[][] phis = new double[L - 1][V];
        double[][] thetasSubtree = new double[D][L - 1];

        int numModels = 0;
        try {
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }
                numModels++;
                inputState(new File(reportFolder, filename).getAbsolutePath());

                for (int d = 0; d < D; d++) {
                    for (int ll = 0; ll < L - 1; ll++) {
                        double val = (double) docLabelCounts[d].getCount(ll) / words[d].length;
                        if (val < 0.01) {
                            val = 0.0;
                        }
                        thetas[d][ll] += val;
                    }
                }

                for (int ll = 0; ll < L - 1; ll++) {
                    for (int v = 0; v < V; v++) {
                        phis[ll][v] += nodes[ll].topic[v];
                    }
                }
            }

            // average
            for (int d = 0; d < D; d++) {
                for (int ll = 0; ll < thetas[d].length; ll++) {
                    thetas[d][ll] /= numModels;
                }
            }

            for (int ll = 0; ll < phis.length; ll++) {
                for (int v = 0; v < V; v++) {
                    phis[ll][v] /= numModels;
                }
            }

            ArrayList<RankingItem<Integer>>[] labelRankWords = new ArrayList[phis.length];
            Set<Integer>[] labelTopWords = new Set[phis.length];
            for (int ll = 0; ll < phis.length; ll++) {
                labelRankWords[ll] = MiscUtils.getRankingList(phis[ll]);

                labelTopWords[ll] = new HashSet<Integer>();
                for (int ii = 0; ii < nTopWords; ii++) {
                    labelTopWords[ll].add(labelRankWords[ll].get(ii).getObject());
                }
            }

            Set<Integer> labelSet = new HashSet<Integer>();
            for (int ll = 0; ll < L - 1; ll++) {
                labelSet.add(ll);
            }

//            featVecs = new double[thetas.length][V + thetas[0].length];
            featVecs = new double[thetas.length][thetas[0].length * 2];
            for (int dd = 0; dd < featVecs.length; dd++) {
                int count = 0;
//                double[] extFeatVec = getFeature(thetas[dd], labelRankWords, nTopWords, labelSet);
//                for (int ii = 0; ii < extFeatVec.length; ii++) {
//                    featVecs[dd][count++] = extFeatVec[ii];
//                }
                for (int ll = 0; ll < thetas[dd].length; ll++) {
                    featVecs[dd][count++] = thetas[dd][ll];
                }

                SparseCount sc = new SparseCount();
                for (int ll = 0; ll < thetas[dd].length; ll++) {
                    for (int n = 0; n < words[dd].length; n++) {
                        if (labelTopWords[ll].contains(words[dd][n])) {
                            sc.increment(ll);
                        }
                    }
                }
                int sum = sc.getCountSum();
                for (int ll = 0; ll < thetas[dd].length; ll++) {
                    int cc = sc.getCount(ll);
                    if (cc == 0) {
                        featVecs[dd][count] = 0.0;
                    } else {
                        featVecs[dd][count] = (double) sc.getCount(ll) / sum;
                    }
                    count++;
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting training feature vectors.");
        }
        return featVecs;
    }

    public double[][] getTestExtendedLexiconFeatureVectors(File iterPredFolder,
            File reportFolder, int nTopWords) {
        double[][] featVecs = null;
        String[] filenames = iterPredFolder.list();
        try {
            double[][] thetas = new double[D][L - 1];
            double[][] phis = new double[L - 1][V];
            double[][] thetasSubtree = new double[D][L - 1];

            int numModels = 0;
            for (String filename : filenames) {
                double[][] singlePreds = PredictionUtils.inputSingleModelClassifications(
                        new File(iterPredFolder, filename));
                if (singlePreds[0].length != L - 1) {
                    throw new RuntimeException("Mismatch");
                }

                numModels++;
                inputModel(new File(reportFolder, filename.replaceAll("txt", "zip")).getAbsolutePath());

                for (int d = 0; d < D; d++) {
                    for (int ll = 0; ll < L - 1; ll++) {
                        double val = singlePreds[d][ll];
                        if (val < 0.01) {
                            val = 0.0;
                        }
                        thetas[d][ll] += val;
                    }
                }

                for (int ll = 0; ll < L - 1; ll++) {
                    for (int v = 0; v < V; v++) {
                        phis[ll][v] += nodes[ll].topic[v];
                    }
                }
            }

            // average
            for (int d = 0; d < D; d++) {
                for (int ll = 0; ll < thetas[d].length; ll++) {
                    thetas[d][ll] /= numModels;
                }
            }

            for (int ll = 0; ll < phis.length; ll++) {
                for (int v = 0; v < V; v++) {
                    phis[ll][v] /= numModels;
                }
            }

            ArrayList<RankingItem<Integer>>[] labelRankWords = new ArrayList[phis.length];
            Set<Integer>[] labelTopWords = new Set[phis.length];
            for (int ll = 0; ll < phis.length; ll++) {
                labelRankWords[ll] = MiscUtils.getRankingList(phis[ll]);

                labelTopWords[ll] = new HashSet<Integer>();
                for (int ii = 0; ii < nTopWords; ii++) {
                    labelTopWords[ll].add(labelRankWords[ll].get(ii).getObject());
                }
            }

            Set<Integer> labelSet = new HashSet<Integer>();
            for (int ll = 0; ll < L - 1; ll++) {
                labelSet.add(ll);
            }

            Set<Node>[] subtrees = new Set[L - 1];
            for (int ll = 0; ll < L - 1; ll++) {
                subtrees[ll] = getSubtree(nodes[ll]);
            }

//            featVecs = new double[thetas.length][V + thetas[0].length];
            featVecs = new double[thetas.length][thetas[0].length * 2];
            for (int dd = 0; dd < featVecs.length; dd++) {
                int count = 0;
//                double[] extFeatVec = getFeature(thetas[dd], labelRankWords, nTopWords, labelSet);
//                for (int ii = 0; ii < extFeatVec.length; ii++) {
//                    featVecs[dd][count++] = extFeatVec[ii];
//                }
                for (int ii = 0; ii < thetas[dd].length; ii++) {
                    featVecs[dd][count++] = thetas[dd][ii];
                }

                // appear in top words
                SparseCount sc = new SparseCount();
                for (int ll = 0; ll < thetas[dd].length; ll++) {
                    for (int n = 0; n < words[dd].length; n++) {
                        if (labelTopWords[ll].contains(words[dd][n])) {
                            sc.increment(ll);
                        }
                    }
                }
                int sum = sc.getCountSum();
                for (int ll = 0; ll < thetas[dd].length; ll++) {
                    int cc = sc.getCount(ll);
                    if (cc == 0) {
                        featVecs[dd][count] = 0.0;
                    } else {
                        featVecs[dd][count] = (double) sc.getCount(ll) / sum;
                    }
                    count++;
                }

                // appear in subtree
                SparseCount subtreeCount = new SparseCount();
                for (int ll = 0; ll < thetas[dd].length; ll++) {
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting test feature vectors.");
        }
        return featVecs;
    }

//    private boolean isInSubtreeTopWords(Set<Integer> subtree, Set<Integer>[] labelTopWords, )
    private double[] getFeature(double[] docThetas,
            ArrayList<RankingItem<Integer>>[] labelRankWords,
            int nTopWords,
            Set<Integer> labelSet) {
        int numWords = labelRankWords[0].size();

        double[] vec = new double[numWords];
        for (int ll : labelSet) {
            for (int ii = 0; ii < nTopWords; ii++) {
                RankingItem<Integer> item = labelRankWords[ll].get(ii);
                int wid = item.getObject();
                double val = item.getPrimaryValue() * docThetas[ll];
                vec[wid] += val;
            }
        }
        return vec;
    }

    private double[][] getFeatures(double[][] thetas,
            double[][] phis,
            int nTopWords,
            Set<Integer> labelSet) {
        int numDocs = thetas.length;
        int numLabels = phis.length;
        int numWords = phis[0].length;

        ArrayList<RankingItem<Integer>>[] labelRankWords = new ArrayList[numLabels];
        for (int ll = 0; ll < numLabels; ll++) {
            labelRankWords[ll] = MiscUtils.getRankingList(phis[ll]);
        }

        double[][] vec = new double[thetas.length][numWords];
        for (int d = 0; d < numDocs; d++) {
            for (int ll : labelSet) {
                for (int ii = 0; ii < nTopWords; ii++) {
                    RankingItem<Integer> item = labelRankWords[ll].get(ii);
                    int wid = item.getObject();
                    double val = item.getPrimaryValue() * thetas[d][ll];
                    vec[d][wid] += val;
                }
            }
        }
        return vec;
    }

    public double[][] getTrainingHierarchicalFeatureVectors() {
        double[][] featVecs = new double[D][L * 3 - 3];
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        double[][] phis = new double[L - 1][V];
        try {
            int numModels = 0;
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }
                numModels++;
                inputState(new File(reportFolder, filename).getAbsolutePath());


                Set<Node>[] subtrees = new Set[L - 1];
                for (int ll = 0; ll < L - 1; ll++) {
                    subtrees[ll] = getSubtree(nodes[ll]);
                }

                for (int d = 0; d < D; d++) {
                    double[] empDist = new double[L - 1];
                    for (int ll = 0; ll < L - 1; ll++) {
                        empDist[ll] = (double) docLabelCounts[d].getCount(ll) / words[d].length;
                        if (empDist[ll] < 0.05) {
                            empDist[ll] = 0.0;
                        }
                    }

                    for (int ll = 0; ll < L - 1; ll++) {
                        Node node = nodes[ll];
                        // per node 
                        featVecs[d][ll] += empDist[ll];

                        // per edge 
//                        for (Node child : node.getChildren()) {
//                            featVecs[d][L - 1 + ll] += (empDist[child.id]);
//                        }

                        // per subtree
//                        for (Node desc : subtrees[ll]) {
//                            if (desc.id == ll) {
//                                continue;
//                            }
//                            featVecs[d][1 * (L - 1) + ll] += empDist[desc.id];
//                        }
                    }
                }
            }

            for (int d = 0; d < D; d++) {
                for (int ii = 0; ii < featVecs[d].length; ii++) {
                    featVecs[d][ii] /= numModels;
                }
            }

//            for (int d = 0; d < D; d++) {
//                for (int ii = 0; ii < featVecs[d].length; ii++) {
//                    featVecs[d][ii] = Math.log(featVecs[d][ii] * words[d].length / numModels + 1);
//                }
//                for (int xx = 0; xx < 3; xx++) {
//                    for (int ll = 0; ll < L - 1; ll++) {
//                        featVecs[d][xx * (L - 1) + ll] *= labelIDFs[ll];
//                    }
//                }
//            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting training feature vectors.");
        }
        return featVecs;
    }

    public double[][] getTestHierarchicalFeatureVectors(File iterPredFolder,
            File reportFolder) {
        double[][] featVecs = new double[D][3 * L - 3];
        String[] filenames = iterPredFolder.list();
        try {
            int numModels = 0;
            for (String filename : filenames) {
                double[][] singlePreds = PredictionUtils.inputSingleModelClassifications(
                        new File(iterPredFolder, filename));
                if (singlePreds[0].length != L - 1) {
                    throw new RuntimeException("Mismatch");
                }

                numModels++;
                inputModel(new File(reportFolder, filename.replaceAll("txt", "zip")).getAbsolutePath());
                Set<Node>[] subtrees = new Set[L - 1];
                for (int ll = 0; ll < L - 1; ll++) {
                    subtrees[ll] = getSubtree(nodes[ll]);
                }

                for (int d = 0; d < D; d++) {
                    double[] empDist = singlePreds[d];
                    for (int ll = 0; ll < empDist.length; ll++) {
                        if (empDist[ll] < 0.05) {
                            empDist[ll] = 0.0;
                        }
                    }

                    for (int ll = 0; ll < L - 1; ll++) {
                        Node node = nodes[ll];
                        // per node 
                        featVecs[d][ll] += empDist[ll];

                        // per edge 
//                        for (Node child : node.getChildren()) {
//                            featVecs[d][L - 1 + ll] += (empDist[child.id]);
//                        }

                        // per subtree
//                        for (Node desc : subtrees[ll]) {
//                            if (desc.id == ll) {
//                                continue;
//                            }
//                            featVecs[d][1 * (L - 1) + ll] += empDist[desc.id];
//                        }
                    }
                }
            }

            for (int d = 0; d < D; d++) {
                for (int ii = 0; ii < featVecs[d].length; ii++) {
                    featVecs[d][ii] /= numModels;
                }
            }
//            for (int d = 0; d < D; d++) {
//                for (int ii = 0; ii < featVecs[d].length; ii++) {
//                    featVecs[d][ii] = Math.log(featVecs[d][ii] * words[d].length / numModels + 1);
//                }
//                for (int xx = 0; xx < 3; xx++) {
//                    for (int ll = 0; ll < L - 1; ll++) {
//                        featVecs[d][xx * (L - 1) + ll] *= labelIDFs[ll];
//                    }
//                }
//            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting test feature vectors.");
        }
        return featVecs;
    }

    private Set<Node> getSubtree(Node node) {
        Set<Node> subtree = new HashSet<Node>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(node);
        while (!stack.isEmpty()) {
            Node n = stack.pop();
            subtree.add(n);
            for (Node c : n.getChildren()) {
                stack.add(c);
            }
        }
        return subtree;
    }

    public SparseVector[] getTrainingFeatureVectors() {
        SparseVector[] featVecs = new SparseVector[D];
        for (int d = 0; d < D; d++) {
            featVecs[d] = new SparseVector();
        }

        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            int numModels = 0;
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                inputState(new File(reportFolder, filename).getAbsolutePath());
                for (int d = 0; d < D; d++) {
                    for (int ll : docLabelCounts[d].getIndices()) {
                        if (ll == L - 1) { // skip the root node
                            continue;
                        }
                        int featIdx = ll + 1;
                        featVecs[d].change(featIdx,
                                (double) docLabelCounts[d].getCount(ll) / words[d].length);
                    }
                }
                numModels++;
            }

            // average
            for (int d = 0; d < D; d++) {
                for (int featIdx : featVecs[d].getIndices()) {
                    double avgVal = featVecs[d].get(featIdx) / numModels;
                    featVecs[d].set(featIdx, avgVal);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting training feature vectors.");
        }
        return featVecs;
    }

    public SparseVector[] getTestFeatureVectors(File iterPredFolder) {
        SparseVector[] featVecs = new SparseVector[D];
        for (int d = 0; d < D; d++) {
            featVecs[d] = new SparseVector();
        }
        double[][] sumDists = new double[D][L - 1];

        String[] filenames = iterPredFolder.list();
        try {
            for (String filename : filenames) {
                double[][] singlePreds = PredictionUtils.inputSingleModelClassifications(
                        new File(iterPredFolder, filename));
                if (singlePreds[0].length != L - 1) {
                    throw new RuntimeException("Mismatch");
                }
                for (int d = 0; d < D; d++) {
                    for (int ll = 0; ll < L - 1; ll++) {
                        sumDists[d][ll] += singlePreds[d][ll];
                    }
                }
            }

            // average
            for (int d = 0; d < D; d++) {
                for (int ll = 0; ll < L - 1; ll++) {
                    double val = sumDists[d][ll] / filenames.length;
                    featVecs[d].set(ll + 1, val);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting test feature vectors.");
        }

        return featVecs;
    }

    // ================= Debug ================
    public double[][] debugPrediction(String stateFile,
            int[][] newWords,
            int[][] newLabels,
            String outputResultFile,
            double[][] initPredictions, double scale) {
        // input model
        inputModel(stateFile);

        // test data
        this.words = newWords;
        this.labels = newLabels;
        this.D = this.words.length;

        // normalize initial predicted scores
        for (int d = 0; d < D; d++) {
            double sum = StatUtils.sum(initPredictions[d]);
            for (int ii = 0; ii < initPredictions[d].length; ii++) {
                initPredictions[d][ii] /= sum;
            }
        }

        double[][] predictedScores = new double[D][];
        for (int d = 0; d < D; d++) {
            predictedScores[d] = debugPredict(d, initPredictions[d], scale);
        }
        return predictedScores;
    }

    private double[] debugPredict(int d, double[] initPreds, double scale) {
        double[] newPreds = new double[L - 1];
        ArrayList<Node> leaves = new ArrayList<Node>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getChildren().isEmpty()) {
                leaves.add(node);
            }
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // bottom-up smoothing to compute pseudo-counts from children
        Queue<Node> queue = new LinkedList<Node>();
        for (Node leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            Node parent = node.getParent();
            if (!node.isRoot() && !queue.contains(parent)) {
                queue.add(parent);
            }

            if (!node.isRoot()) {
                double currentScore = initPreds[node.id];

                double addedScore = 0.0;
                // from parent
                if (!node.getParent().isRoot()) {
                    double scoreFromParent = inWeights[node.id].get(node.getParent().id);
                    addedScore += scoreFromParent * initPreds[node.getParent().id];
                }

                // from children
                double maxChildScore = -1;
                Node maxChild = null;
                for (Node child : node.getChildren()) {
                    double childScore = initPreds[child.id];
                    if (maxChildScore < childScore) {
                        maxChildScore = childScore;
                        maxChild = child;
                    }
                }
                if (maxChild != null) {
                    double weight = inWeights[maxChild.id].get(node.id);
                    addedScore += maxChildScore * weight;
                }

                // current score
                newPreds[node.id] = scale * currentScore + (1 - scale) * addedScore;
            }
        }

        if (d < 50) {
            System.out.println("d = " + d);
            for (int ll : labels[d]) {
                System.out.println("TRUE: " + ll
                        + ". " + nodes[ll].getPathString()
                        + ". " + labelVocab.get(ll));
            }
            System.out.println();

            ArrayList<RankingItem<Integer>> rankTFIDF = MiscUtils.getRankingList(initPreds);
            for (int ii = 0; ii < Math.max(labels[d].length, 15); ii++) {
                RankingItem<Integer> item = rankTFIDF.get(ii);
                System.out.println("TFIDF: " + truth(labels[d], item.getObject())
                        + ". " + item.getObject()
                        + ". " + nodes[item.getObject()].getPathString()
                        + ". " + labelVocab.get(item.getObject())
                        + ". " + item.getPrimaryValue());
            }
            System.out.println();

            ArrayList<RankingItem<Integer>> predList = MiscUtils.getRankingList(newPreds);
            for (int ii = 0; ii < Math.max(labels[d].length, 15); ii++) {
                RankingItem<Integer> item = predList.get(ii);
                System.out.println("Pred: " + truth(labels[d], item.getObject())
                        + ". " + item.getObject()
                        + ". " + nodes[item.getObject()].getPathString()
                        + ". " + labelVocab.get(item.getObject())
                        + ". " + item.getPrimaryValue());
            }
            System.out.println();
            System.out.println();
        }

        return newPreds;
    }

    public void sampleNewDocumentsDebug(String stateFile,
            int[][] newWords, int[][] newLabels,
            String outputResultFile,
            double[][] initPredictions) {
        if (verbose) {
            System.out.println();
            logln("Perform prediction using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        // test data
        this.words = newWords;
        this.labels = newLabels;
        this.D = this.words.length;

        // initialize data structure
        this.z = new int[D][];
        this.x = new int[D][];
        this.docSwitches = new DirMult[D];
        this.docLabelCounts = new SparseCount[D];
        this.docMaskes = new Set[D];

        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
            this.x[d] = new int[words[d].length];
            this.docSwitches[d] = new DirMult(
                    new double[]{hyperparams.get(A_0), hyperparams.get(B_0)});
            this.docLabelCounts[d] = new SparseCount();
            this.docMaskes[d] = new HashSet<Integer>();

            Set<Integer> cands = getCandidatesNew(initPredictions[d]);
            for (int label : cands) {
                Node node = nodes[label];
                while (node != null) {
                    docMaskes[d].add(node.id);
                    node = node.getParent();
                }
            }
        }

        // debug
        for (int d = 0; d < 20; d++) {
            System.out.println("\n\nd = " + d);
            for (int ll : labels[d]) {
                System.out.println("TRUE: " + ll
                        + ". " + nodes[ll].getPathString()
                        + ". " + labelVocab.get(ll));
            }
            System.out.println();

            ArrayList<RankingItem<Integer>> rankTFIDF = MiscUtils.getRankingList(initPredictions[d]);
            for (int ii = 0; ii < Math.max(labels[d].length, 15); ii++) {
                RankingItem<Integer> item = rankTFIDF.get(ii);
                System.out.println("TFIDF: " + truth(labels[d], item.getObject())
                        + ". " + item.getObject()
                        + ". " + nodes[item.getObject()].getPathString()
                        + ". " + labelVocab.get(item.getObject())
                        + ". " + item.getPrimaryValue());
            }
            System.out.println();

//        for (int ll : docMaskes[d]) {
//            System.out.println("Mask: " + ll
//                    + ". " + nodes[ll].getPathString()
//                    + ". " + labelVocab.get(ll));
//        }
//        System.out.println();

            double[] predictedScores = new double[L - 1]; // exclude root
            int count = 0;
            for (iter = 0; iter < 5; iter++) {
                for (int n = 0; n < words[d].length; n++) {
                    if (iter == 0) {
                        sampleXZMH(d, n, !REMOVE, !ADD, !REMOVE, ADD);
                    } else {
                        sampleXZMH(d, n, !REMOVE, !ADD, REMOVE, ADD);
                    }
                }
                for (int ll = 0; ll < L - 1; ll++) {
                    predictedScores[ll] +=
                            (double) docLabelCounts[d].getCount(ll) / words[d].length;
                }
                count++;

//            System.out.println();
//            System.out.println("iter = " + iter
//                    + ". " + docSwitches[d].getCount(INSIDE)
//                    + ". " + docSwitches[d].getCount(OUTSIDE));
//            for (int k = 0; k < L; k++) {
//                if (docLabelCounts[d].getCount(k) > 0) {
//                    System.out.println("--- " + k
//                            + ". " + nodes[k].getPathString()
//                            + ". " + labelVocab.get(k)
//                            + ". " + docLabelCounts[d].getCount(k));
//                }
//            }
            }

            for (int ll = 0; ll < L - 1; ll++) {
                predictedScores[ll] /= count;
            }
            ArrayList<RankingItem<Integer>> predList = MiscUtils.getRankingList(predictedScores);
            for (int ii = 0; ii < Math.max(labels[d].length, 15); ii++) {
                RankingItem<Integer> item = predList.get(ii);
                System.out.println("Pred: " + truth(labels[d], item.getObject())
                        + ". " + item.getObject()
                        + ". " + nodes[item.getObject()].getPathString()
                        + ". " + labelVocab.get(item.getObject())
                        + ". " + item.getPrimaryValue());
            }
        }
    }

    private boolean truth(int[] ls, int l) {
        for (int ii = 0; ii < ls.length; ii++) {
            if (ls[ii] == l) {
                return true;
            }
        }
        return false;
    }
    // ******************* End prediction **************************************

    // ******************* Start perplexity ************************************
    private long sampleXZsExact_test(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            ArrayList<Integer>[] trainIndices) {
        numTokensChange = 0;
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            for (int ii = 0; ii < trainIndices[d].size(); ii++) {
                int n = trainIndices[d].get(ii);
                sampleXZExact_test(d, ii, n, removeFromModel, addToModel, removeFromData, addToData);
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample x and z together for a token.
     *
     * @param d
     * @param n
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private void sampleXZExact_test(int d, int ii, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            nodes[z[d][ii]].getContent().decrement(words[d][n]);
            nodes[z[d][ii]].removeToken(d, n);
        }
        if (removeFromData) {
            docSwitches[d].decrement(x[d][ii]);
            docLabelCounts[d].decrement(z[d][ii]);
        }

        double[] logprobs = new double[L];
        for (int ll = 0; ll < L; ll++) {
            boolean inside = docMaskes[d].contains(ll);
            double xLlh;
            double zLlh;
            double wLlh = Math.log(nodes[ll].topic[words[d][n]]);

            if (inside) {
                xLlh = Math.log(docSwitches[d].getCount(INSIDE) + hyperparams.get(A_0));
                zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * docMaskes[d].size()));
            } else {
                xLlh = Math.log(docSwitches[d].getCount(OUTSIDE) + hyperparams.get(B_0));
                zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(OUTSIDE) + hyperparams.get(ALPHA) * (L - docMaskes[d].size())));
            }
            logprobs[ll] = xLlh + zLlh + wLlh;
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledZ != z[d][ii]) {
            numTokensChange++;
        }
        z[d][ii] = sampledZ;
        if (docMaskes[d].contains(z[d][ii])) {
            x[d][ii] = INSIDE;
        } else {
            x[d][ii] = OUTSIDE;
        }

        if (addToModel) {
            nodes[z[d][ii]].getContent().increment(words[d][n]);
            nodes[z[d][ii]].addToken(d, n);
        }

        if (addToData) {
            docSwitches[d].increment(x[d][ii]);
            docLabelCounts[d].increment(z[d][ii]);
        }
    }

    /**
     * Perform sampling for test documents to compute perplexity.
     *
     * @param stateFile File storing the state of a learned model
     * @param newWords Words of test documents
     * @param newLabels Labels of test documents
     */
    public double computePerplexity(String stateFile,
            int[][] newWords, int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices) {
        if (verbose) {
            System.out.println();
            logln("Computing perplexity using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        words = newWords;
        labels = newLabels;
        D = words.length;
        docLabelCounts = new SparseCount[D];
        int numTrainTokens = 0;
        int numTestTokens = 0;

        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
            numTrainTokens += trainIndices[d].size();
            numTestTokens += testIndices[d].size();
        }

        if (verbose) {
            logln("Test data:");
            logln("--- D = " + D);
            logln("--- # tokens = " + numTokens);
            logln("--- # train tokens = " + numTrainTokens);
            logln("--- # test tokens = " + numTestTokens);
        }

        this.z = new int[D][];
        this.x = new int[D][];
        this.docSwitches = new DirMult[D];
        this.docLabelCounts = new SparseCount[D];
        this.docMaskes = new Set[D];
        for (int d = 0; d < D; d++) {
            this.z[d] = new int[trainIndices[d].size()];
            this.x[d] = new int[trainIndices[d].size()];
            this.docSwitches[d] = new DirMult(new double[]{hyperparams.get(A_0),
                        hyperparams.get(B_0)});
            this.docLabelCounts[d] = new SparseCount();
            this.docMaskes[d] = new HashSet<Integer>();
            if (labels != null) { // if labels are given during training time
                updateMaskes(d);
            }
        }

        ArrayList<Double> perplexities = new ArrayList<Double>();
        if (verbose) {
            logln("--- Sampling on test data ...");
        }
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId());
            }
            if (iter == 0) {
                sampleXZsExact_test(!REMOVE, !ADD, !REMOVE, ADD, trainIndices);
            } else {
                sampleXZsExact_test(!REMOVE, !ADD, REMOVE, ADD, trainIndices);
            }

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                perplexities.add(computePerplexity(testIndices, stateFile + ".perp"));
            }
        }
        double avgPerplexity = StatUtils.mean(perplexities);
        return avgPerplexity;
    }

    /**
     * Compute the perplexity of the current assignments.
     */
    private double computePerplexity(ArrayList<Integer>[] testIndices, String outFile) {
        double totalLogprob = 0.0;
        int numTestTokens = 0;
        for (int d = 0; d < D; d++) {
            numTestTokens += testIndices[d].size();
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outFile);
            for (int d = 0; d < D; d++) {
                double docLogProb = 0.0;
                for (int n : testIndices[d]) {
                    double val = 0.0;
                    for (int ll = 0; ll < L; ll++) {
                        boolean inside = docMaskes[d].contains(ll);
                        double pw = nodes[ll].topic[words[d][n]];
                        double pz;
                        double px;
                        if (inside) {
                            px = (docSwitches[d].getCount(INSIDE) + hyperparams.get(A_0))
                                    / (docSwitches[d].getCountSum() + hyperparams.get(A_0) + hyperparams.get(B_0));
                            pz = (docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                                    / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * docMaskes[d].size());
                        } else {
                            px = (docSwitches[d].getCount(OUTSIDE) + hyperparams.get(B_0))
                                    / (docSwitches[d].getCountSum() + hyperparams.get(A_0) + hyperparams.get(B_0));
                            pz = (docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                                    / (docSwitches[d].getCount(OUTSIDE) + hyperparams.get(ALPHA) * (L - docMaskes[d].size()));
                        }
                        val += pw * pz * px;
                    }
                    docLogProb += Math.log(val);
                }
                totalLogprob += docLogProb;
                writer.write(d
                        + "\t" + words[d].length
                        + "\t" + testIndices[d].size()
                        + "\t" + docLogProb + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
        double perplexity = Math.exp(-totalLogprob / numTestTokens);
        return perplexity;
    }

    /**
     * Static method to compute perplexity using multiple chains, each
     * corresponds to using a learned model during training.
     *
     * @param newWords Words of test documents
     * @param newLabels Labels of test documents
     * @param iterPerplexityFolder Perplexity folder
     * @param sampler The sample to propagate its configurations to multiple
     * parallel samplers
     */
    public static void parallelPerplexity(int[][] newWords,
            int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            File iterPerplexityFolder,
            File resultFolder,
            L2H sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPerplexityFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPerplexityFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                L2HPerplexityRunner runner = new L2HPerplexityRunner(sampler,
                        newWords, newLabels, trainIndices, testIndices,
                        stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // summarize multiple perplexities
            String[] ppxFiles = iterPerplexityFolder.list();
            ArrayList<Double> ppxs = new ArrayList<Double>();
            for (String ppxFile : ppxFiles) {
                double ppx = IOUtils.inputPerplexity(new File(iterPerplexityFolder, ppxFile));
                ppxs.add(ppx);
            }

            // averaging
            File ppxResultFile = new File(resultFolder, PerplexityFile);
            IOUtils.outputPerplexities(ppxResultFile, ppxs);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while computing perplexity parallel test.");
        }
    }
    // ******************* End perplexity **************************************

    /**
     * Compute topic coherence.
     */
    public double[][] computeAvgTopicCoherence(File file,
            MimnoTopicCoherence topicCoherence) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing averaged topic coherence to file " + file);
        }

        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        double[][] avgTopics = new double[L][V];
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file.getAbsolutePath() + ".iter");
            writer.write("Iteration");
            for (int k = 0; k < L; k++) {
                writer.write("\tTopic_" + k);
            }
            writer.write("\n");

            // partial score
            ArrayList<double[][]> aggTopics = new ArrayList<double[][]>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }
                inputModel(new File(reportFolder, filename).getAbsolutePath());
                double[][] pointTopics = new double[L][V];

                writer.write(filename);
                for (int k = 0; k < L; k++) {
                    pointTopics[k] = nodes[k].topic;
                    int[] topic = SamplerUtils.getSortedTopic(pointTopics[k]);
                    double score = topicCoherence.getCoherenceScore(topic);

                    writer.write("\t" + score);
                }
                writer.write("\n");
                aggTopics.add(pointTopics);
            }

            // averaging
            writer.write("Average");
            ArrayList<Double> scores = new ArrayList<Double>();
            for (int k = 0; k < L; k++) {
                double[] avgTopic = new double[V];
                for (int v = 0; v < V; v++) {
                    for (int ii = 0; ii < aggTopics.size(); ii++) {
                        avgTopic[v] += aggTopics.get(ii)[k][v] / aggTopics.size();
                    }
                }
                int[] topic = SamplerUtils.getSortedTopic(avgTopic);
                double score = topicCoherence.getCoherenceScore(topic);
                writer.write("\t" + score);
                if (!nodes[k].isRoot()) {
                    scores.add(score);
                }
                avgTopics[k] = avgTopic;
            }
            writer.write("\n");
            writer.close();

            // output aggregated topic coherence scores
            IOUtils.outputTopicCoherences(file, scores);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
        return avgTopics;
    }

    class Node extends TreeNode<Node, SparseCount> {

        final int id;
        double[] topic;
        SparseCount pseudoCounts;
        HashMap<Integer, ArrayList<Integer>> assignedTokens;

        Node(int id, int index, int level,
                SparseCount content,
                Node parent) {
            super(index, level, content, parent);
            this.id = id;
            this.pseudoCounts = new SparseCount();
            this.assignedTokens = new HashMap<Integer, ArrayList<Integer>>();
        }

        public Set<Integer> getSubtree() {
            Set<Integer> subtree = new HashSet<Integer>();
            Stack<Node> stack = new Stack<Node>();
            stack.add(this);
            while (!stack.isEmpty()) {
                Node n = stack.pop();
                for (Node c : n.getChildren()) {
                    stack.add(c);
                }
                subtree.add(n.id);
            }
            return subtree;
        }

        public Set<Integer> getAssignedDocuments() {
            return this.assignedTokens.keySet();
        }

        public HashMap<Integer, ArrayList<Integer>> getAssignedTokens() {
            return this.assignedTokens;
        }

        public void addToken(int d, int n) {
            ArrayList<Integer> docAssignedTokens = this.assignedTokens.get(d);
            if (docAssignedTokens == null) {
                docAssignedTokens = new ArrayList<Integer>();
            }
            docAssignedTokens.add(n);
            this.assignedTokens.put(d, docAssignedTokens);
        }

        public void removeToken(int d, int n) {
            this.assignedTokens.get(d).remove(Integer.valueOf(n));
            if (this.assignedTokens.get(d).isEmpty()) {
                this.assignedTokens.remove(d);
            }
        }

        public double[] getTopic() {
            return this.topic;
        }

        public void setTopic(double[] t) {
            this.topic = t;
        }

        /**
         * Return true if the given node is in the subtree rooted at this node
         *
         * @param node The given node to check
         */
        public boolean isDescendent(Node node) {
            Node temp = node;
            while (temp != null) {
                if (temp.equals(this)) {
                    return true;
                }
                temp = temp.parent;
            }
            return false;
        }

        public void getPseudoCountsFromChildren() {
            if (pathAssumption == PathAssumption.MINIMAL) {
                this.getPseudoCountsFromChildrenMin();
            } else if (pathAssumption == PathAssumption.MAXIMAL) {
                this.getPseudoCountsFromChildrenMax();
            } else {
                throw new RuntimeException("Path assumption " + pathAssumption
                        + " is not supported.");
            }
        }

        /**
         * Propagate the observations from all children nodes to this node using
         * minimal path assumption, which means for each observation type v, a
         * child node will propagate a value of 1 if it contains v, and 0
         * otherwise.
         */
        public void getPseudoCountsFromChildrenMin() {
            this.pseudoCounts = new SparseCount();
            for (Node child : this.getChildren()) {
                SparseCount childObs = child.getContent();
                for (int obs : childObs.getIndices()) {
                    this.pseudoCounts.increment(obs);
                }
            }
        }

        /**
         * Propagate the observations from all children nodes to this node using
         * maximal path assumption, which means that each child node will
         * propagate its full observation vector.
         */
        public void getPseudoCountsFromChildrenMax() {
            this.pseudoCounts = new SparseCount();
            for (Node child : this.getChildren()) {
                SparseCount childObs = child.getContent();
                for (int obs : childObs.getIndices()) {
                    this.pseudoCounts.changeCount(obs, childObs.getCount(obs));
                }
            }
        }

        /**
         * Sample topic. This applies since the topic of a node is modeled as a
         * drawn from a Dirichlet distribution with the mean vector is the topic
         * of the node's parent and scaling factor gamma.
         *
         * @param beta Topic smoothing parameter
         * @param gamma Dirichlet-Multinomial chain parameter
         */
        public void sampleTopic() {
            double[] meanVector = new double[V];
            Arrays.fill(meanVector, hyperparams.get(BETA) / V); // to prevent zero count

            // from parent
            if (!this.isRoot()) {
                double[] parentTopic = parent.topic;
                for (int v = 0; v < V; v++) {
                    meanVector[v] += parentTopic[v] * hyperparams.get(BETA);
                }
            } else {
                for (int v = 0; v < V; v++) {
                    meanVector[v] += hyperparams.get(BETA) / V;
                }
            }

            // current observations
            SparseCount observations = this.content;
            for (int obs : observations.getIndices()) {
                meanVector[obs] += observations.getCount(obs);
            }

            // from children
            for (int obs : this.pseudoCounts.getIndices()) {
                meanVector[obs] += this.pseudoCounts.getCount(obs);
            }

            Dirichlet dir = new Dirichlet(meanVector);
            double[] ts = dir.nextDistribution();

            if (debug) {
                for (int v = 0; v < V; v++) {
                    if (ts[v] == 0) {
                        throw new RuntimeException("Zero probability");
                    }
                }
            }
            this.setTopic(ts);
        }

        public double getWordLogLikelihood() {
            double llh = 0.0;
            for (int w : getContent().getIndices()) {
                llh += getContent().getCount(w) * Math.log(topic[w]);
            }
            return llh;
        }

        public void validate(String msg) {
            int numTokens = 0;
            for (int dd : this.assignedTokens.keySet()) {
                numTokens += this.assignedTokens.get(dd).size();
            }

            if (numTokens != this.getContent().getCountSum()) {
                throw new RuntimeException(msg + ". Mismatch: " + numTokens
                        + " vs. " + this.getContent().getCountSum());
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(id).append(", ")
                    .append(getPathString())
                    .append(", #c = ").append(getChildren().size())
                    .append(", #o = ").append(getContent().getCountSum())
                    .append("]");
            return str.toString();
        }
    }
}
class L2HPerplexityRunner implements Runnable {

    L2H sampler;
    int[][] newWords;
    int[][] newLabels;
    ArrayList<Integer>[] trainIndices;
    ArrayList<Integer>[] testIndices;
    String stateFile;
    String outputFile;

    public L2HPerplexityRunner(L2H sampler,
            int[][] newWords,
            int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newLabels = newLabels;
        this.trainIndices = trainIndices;
        this.testIndices = testIndices;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        L2H testSampler = new L2H();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            double perplexity = testSampler.computePerplexity(stateFile,
                    newWords, newLabels, trainIndices, testIndices);
            IOUtils.outputPerplexity(outputFile, perplexity);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}

class L2HTestRunner implements Runnable {

    L2H sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;
    double[][] initPredidctions;
    int topK;

    public L2HTestRunner(L2H sampler,
            int[][] newWords,
            String stateFile,
            String outputFile,
            double[][] initPreds,
            int topK) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
        this.initPredidctions = initPreds;
        this.topK = topK;
    }

    @Override
    public void run() {
        L2H testSampler = new L2H();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.sampleNewDocuments(stateFile, newWords, outputFile,
                    initPredidctions, topK);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
