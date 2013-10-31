package sampler.backup;

import cc.mallet.types.Dirichlet;
import core.AbstractSampler;
import data.LabelTextData;
import gnu.trove.TIntIntHashMap;
import graph.DirectedGraph;
import graph.EdmondsMST;
import graph.GraphEdge;
import graph.GraphNode;
import graph.PrimMST;
import graph.UndirectedGraph;
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
import java.util.concurrent.LinkedBlockingDeque;
import util.CLIUtils;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.labeled.LabeledLDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class HierarchicalLabeledLDABackup extends AbstractSampler {

    public static enum PathAssumption {

        MINIMAL, MAXIMAL
    }
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int GAMMA = 2;
    public static final int MEAN = 3;
    public static final int SCALE = 4;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int L;
    protected int V;
    protected int D;
    private ArrayList<String> labelVocab;
    private PathAssumption pathAssumption;
    // tree-based hierarchy
    private HLLDANode root;
    private ArrayList<HLLDANode> leaves;
    private HLLDANode[] nodeList;
    // label graph
    private GraphNode<Integer> graphRoot;
    private int[][] z;
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
            double mean,
            double scale,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
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
        this.hyperparams.add(mean);
        this.hyperparams.add(scale);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInterval;

        this.initState = initState;
        this.pathAssumption = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        int[] labelFreqs = this.getLabelFrequencies();
        for (int ll = 0; ll < L; ll++) {
            labelVocab.set(ll, labelVocab.get(ll) + " (" + labelFreqs[ll] + ")");
        }

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- label vocab:\t" + L);
            logln("--- word vocab:\t" + V);
            logln("--- # tokens:\t" + numTokens);
            logln("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            logln("--- beta:\t" + MiscUtils.formatDouble(beta));
            logln("--- gamma:\t" + MiscUtils.formatDouble(gamma));
            logln("--- mean:\t" + MiscUtils.formatDouble(mean));
            logln("--- scale:\t" + MiscUtils.formatDouble(scale));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- path assumption:\t" + pathAssumption);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_HLLDA")
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(MiscUtils.formatDouble(hyperparams.get(ALPHA)))
                .append("_b-").append(MiscUtils.formatDouble(hyperparams.get(BETA)))
                .append("_g-").append(MiscUtils.formatDouble(hyperparams.get(GAMMA)))
                .append("_m-").append(MiscUtils.formatDouble(hyperparams.get(MEAN)))
                .append("_s-").append(MiscUtils.formatDouble(hyperparams.get(SCALE)))
                .append("_opt-").append(this.paramOptimized)
                .append("_path-").append(this.pathAssumption);
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

        // create list of labels
        this.nodeList = new HLLDANode[L + 1];
        this.labelVocab.add("Background");

        // create tree root
        this.root = new HLLDANode(-1, 0, 0, L, null);
        this.leaves = new ArrayList<HLLDANode>();

        int[] labelFreqs = getLabelFrequencies();
        HashMap<String, Integer> labelPairFreqs = getLabelPairFrequencies();
        TreeInitializer treeInit = new TreeInitializer(
                labelFreqs,
                labelPairFreqs);
        treeInit.initializeUsingEntropies();
//        treeInit.initializeUsingConditionalProbabilities();
        DirectedGraph<Integer> tree = treeInit.getTree();
        this.graphRoot = treeInit.getGraphRoot();
        this.setTree(tree, graphRoot);

        if (verbose) {
            logln("--- --- Initializing topics using Labeled LDA ...");
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
        llda.configure(null, words, labels,
                V, L, lda_alpha, lda_beta, initState, false,
                lda_burnin, lda_maxiter, lda_samplelag, lda_repInterval);

        try {
            File lldaZFile = new File(this.folder, "labeled-lda-init-" + L + ".zip");
            if (lldaZFile.exists()) {
                llda.inputState(lldaZFile);
            } else {
                llda.sample();
                llda.outputState(lldaZFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while initializing topics with Labeled LDA");
        }
        setLog(true);

        // topics
        DirMult[] labelWordDists = llda.getTopicWordDistributions();
        for (int ll = 0; ll < L; ll++) {
            double[] topic = labelWordDists[ll].getDistribution();
            this.nodeList[ll].setTopic(topic);
        }

        // use the Laplace-smoothed background distribution for the root's topics
        double[] topic = new double[V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                topic[words[d][n]]++;
            }
        }
        for (int v = 0; v < V; v++) {
            topic[v] = (topic[v] + 1) / (this.numTokens + V);
        }
        this.nodeList[L].setTopic(topic);
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }
        this.z = new int[D][];
        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
        }

        if (verbose) {
            logln("--- --- Data structure initialized\n"
                    + this.printGlobalTree());
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }
        this.sampleTopicAssignments(!REMOVE, ADD);

        if (verbose) {
            logln("--- --- Assignments initialized\n" + this.printGlobalTree());
        }
    }

    /**
     * Get a set of candidate labels for a document, which contains all nodes on
     * the paths from root to the document's observed labels.
     *
     * @param d Document index
     */
    private Set<Integer> getPathCandidateLabels(int d) {
        Set<Integer> candLabels = new HashSet<Integer>();
        for (int ii = 0; ii < labels[d].length; ii++) {
            int l = labels[d][ii];
            HLLDANode node = this.nodeList[l];
            while (node.getParent() != null) {
                int labelId = node.getContent();
                candLabels.add(labelId);

                node = node.getParent();
            }
        }
        return candLabels;
    }

    /**
     * Get a set of candidate labels for a document, which contains all nodes
     * under the subtree rooted at any first-level nodes on the paths from root
     * to the document's observed labels.
     *
     * @param d The document index
     */
    private Set<Integer> getSubtreeCandidateLabels(int d) {
        Set<Integer> candLabels = new HashSet<Integer>();
        Set<Integer> firstLevelLabels = new HashSet<Integer>();
        for (int ii = 0; ii < labels[d].length; ii++) {
            int l = labels[d][ii];
            HLLDANode node = this.nodeList[l];
            while (node.getParent() != null) {
                int labelId = node.getContent();
                candLabels.add(labelId);

                if (node.getLevel() == 1) {
                    firstLevelLabels.add(labelId);
                }

                node = node.getParent();
            }
        }

        for (int firstLevelLabel : firstLevelLabels) {
            Set<Integer> subtreeLabelIds = getSubtreeLabelIds(firstLevelLabel);
            for (int id : subtreeLabelIds) {
                candLabels.add(id);
            }
        }

        return candLabels;
    }

    /**
     * Get the set of labels in a subtree rooted at a given label
     *
     * @param labelId The root node's label ID
     */
    private Set<Integer> getSubtreeLabelIds(int labelId) {
        Set<Integer> subtreeLabelIds = new HashSet<Integer>();
        HLLDANode rootNode = this.nodeList[labelId];
        Stack<HLLDANode> stack = new Stack<HLLDANode>();
        stack.add(rootNode);
        while (!stack.isEmpty()) {
            HLLDANode node = stack.pop();
            subtreeLabelIds.add(node.getContent());
            for (HLLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return subtreeLabelIds;
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
                        + "\t # tokens changed: " + numTokensChange
                        + "\t (" + ((double) numTokensChange / numTokens) + ")"
                        + "\t" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            sampleTopicAssignments(REMOVE, ADD);

            sampleTopics();

            updateTree();

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
            BufferedWriter writer = IOUtils.getBufferedWriter(
                    new File(getSamplerFolderPath(), "likelihoods.txt"));
            for (int i = 0; i < logLikelihoods.size(); i++) {
                writer.write(i + "\t" + logLikelihoods.get(i) + "\n");
            }
            writer.close();

            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                        "hyperparameters.txt"));
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void addObservationToNode(HLLDANode node, int obs) {
        node.changeNumStayingCustomers(1);
        node.incrementObservation(obs);
        HLLDANode tmpNode = node.getParent();
        while (tmpNode != null) {
            tmpNode.changeNumPassingCustomers(1);
            tmpNode = tmpNode.getParent();
        }
    }

    private void removeObservationFromNode(HLLDANode node, int obs) {
        node.changeNumStayingCustomers(-1);
        node.decrementObservation(obs);
        HLLDANode tmpNode = node.getParent();
        while (tmpNode != null) {
            tmpNode.changeNumPassingCustomers(-1);
            tmpNode = tmpNode.getParent();
        }
    }

    /**
     * Sample topic assignments for all tokens in all documents.
     *
     * @param remove Whether the existing assignments should be removed
     * @param add Whether the new sampled assignments should be added
     */
    protected void sampleTopicAssignments(boolean remove, boolean add) {
        int docStepSize = MiscUtils.getRoundStepSize(D, 5);
        numTokensChange = 0;
        for (int d = 0; d < D; d++) {
            // only allow this document to be assigned to a subset of the tree
            // depending on the document's set of labels
//            Set<Integer> candidateLabels = this.getSubtreeCandidateLabels(d);
            Set<Integer> candidateLabels = this.getPathCandidateLabels(d);

            // debug
            if (verbose && debug && d % docStepSize == 0) {
                logln("iter = " + iter
                        + ". d = " + d + " / " + D);
            }

            for (int n = 0; n < words[d].length; n++) {
                if (remove) {
                    this.removeObservationFromNode(nodeList[z[d][n]], words[d][n]);
                }

                HashMap<HLLDANode, Double> nodeLogPriors = new HashMap<HLLDANode, Double>();
                this.getNodeLogPrior(root, 0.0, nodeLogPriors, candidateLabels);

                ArrayList<HLLDANode> nlist = new ArrayList<HLLDANode>();
                ArrayList<Double> logprobs = new ArrayList<Double>();
                for (HLLDANode node : nodeLogPriors.keySet()) {
                    double logprob = nodeLogPriors.get(node)
                            + Math.log(node.getTopic()[words[d][n]]);

                    nlist.add(node);
                    logprobs.add(logprob);
                }
                int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
                if (sampledIdx == logprobs.size()) {
                    logln("iter = " + iter
                            + ". d = " + d
                            + ". n = " + n
                            + ". sampled idx: " + sampledIdx);

                    for (HLLDANode node : nodeLogPriors.keySet()) {
                        double nodeLlh = Math.log(node.getTopic()[words[d][n]]);
                        logln("--- node " + node.toString()
                                + ". " + MiscUtils.formatDouble(nodeLogPriors.get(node))
                                + ". " + MiscUtils.formatDouble(nodeLlh));
                    }
                    throw new RuntimeException("Sampling-out-of-bound Exception");
                }

                int labelIdx = nlist.get(sampledIdx).getContent();

                if (labelIdx != z[d][n]) {
                    numTokensChange++;
                }

                z[d][n] = labelIdx;
                if (add) {
                    this.addObservationToNode(nodeList[z[d][n]], words[d][n]);
                }
            }
        }
    }

    /**
     * Recursively compute the log prior of assigning to each node in the tree.
     *
     * @param node Current node
     * @param val Value passed from the parent node
     * @param vals Hash map to store the results
     * @param candNodes Set of candidate nodes
     */
    private void getNodeLogPrior(
            HLLDANode node,
            double val,
            HashMap<HLLDANode, Double> vals,
            Set<Integer> candNodes) {
        int numStays = node.getNumStayingCustomers();
        int numPasses = node.getNumPassingCustomers();

        // compute the probability of staying at the current node
        double stayingProb =
                (numStays + hyperparams.get(MEAN) * hyperparams.get(SCALE))
                / (numStays + numPasses + hyperparams.get(SCALE));
        vals.put(node, val + Math.log(stayingProb));

        // compute the probability of moving pass the current node
        double passingLogProb = Math.log(1.0 - stayingProb);

        // debug
//        System.out.println(node.toString()
//                + "\t" + numStays
//                + "\t" + numPasses
//                + "\tstaying: " + stayingProb
//                + "\tlog passing: " + passingLogProb
//                + "\tpre-val = " + val
//                + "\tval = " + (val + Math.log(stayingProb))
//                );

        // compute the probability of choosing a child node
        double totalChildRawScore = 0;
        HashMap<HLLDANode, Double> childRawScores = new HashMap<HLLDANode, Double>();
        for (HLLDANode child : node.getChildren()) {
            if (candNodes != null && !candNodes.contains(child.getContent())) {
                continue;
            }

            double childRawScore =
                    child.getNumStayingCustomers()
                    + child.getNumPassingCustomers()
                    + hyperparams.get(ALPHA);
            totalChildRawScore += childRawScore;
            childRawScores.put(child, childRawScore);

            // debug
//            System.out.println("--- --- child " + child.toString()
//                    + "\t" + child.getNumStayingCustomers()
//                    + "\t" + child.getNumPassingCustomers()
//                    + "\t" + childRawScore);
        }

        for (HLLDANode vChild : childRawScores.keySet()) {
            double childLogProb =
                    passingLogProb
                    + Math.log(childRawScores.get(vChild) / totalChildRawScore);

            // debug
//            System.out.println("+++ +++ child " + vChild.toString()
//                    + "\t" + passingLogProb
//                    + "\t" + childRawScores.get(vChild)
//                    + "\t" + totalChildRawScore
//                    + "\t" + Math.log(childRawScores.get(vChild) / totalChildRawScore)
//                    + "\t" + childLogProb);

            this.getNodeLogPrior(vChild, val + childLogProb, vals, candNodes);
        }
    }

    protected void sampleTopics() {
        // bottom-up smoothing to compute pseudo-counts from children
        Queue<HLLDANode> queue = new LinkedList<HLLDANode>();
        for (HLLDANode leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            HLLDANode node = queue.poll();
            if (node.equals(root)) {
                break;
            }

            HLLDANode parent = node.getParent();
            if (!queue.contains(parent)) {
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
        queue = new LinkedBlockingDeque<HLLDANode>();
        queue.add(root);
        while (!queue.isEmpty()) {
            HLLDANode node = queue.poll();
            for (HLLDANode child : node.getChildren()) {
                queue.add(child);
            }

            node.sampleTopic();
        }
    }

    protected void updateTree() {
        // remember to update this.leaves
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int kk = 0; kk < L + 1; kk++) {
            wordLlh += nodeList[kk].getLogLikelihood();
        }

        double llh = wordLlh;
        if (verbose) {
            logln("--- word: " + MiscUtils.formatDouble(wordLlh));
        }

        return llh;
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
        StringBuilder str = new StringBuilder();
        Stack<HLLDANode> stack = new Stack<HLLDANode>();
        stack.add(root);

        int totalObs = 0;
        int numNodes = 0;
        SparseCount tokenCountPerLevel = new SparseCount();
        SparseCount nodeCountPerLevel = new SparseCount();

        while (!stack.isEmpty()) {
            HLLDANode node = stack.pop();
            numNodes++;
            tokenCountPerLevel.changeCount(node.getLevel(), node.getNumStayingCustomers());
            nodeCountPerLevel.increment(node.getLevel());
            totalObs += node.getTotalObservationCount();
            for (HLLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # nodes = ").append(numNodes)
                .append("\n");
        for (int level : tokenCountPerLevel.getSortedIndices()) {
            str.append(">>> level ").append(level)
                    .append(". ").append(nodeCountPerLevel.getCount(level))
                    .append(": ").append(tokenCountPerLevel.getCount(level))
                    .append("\n");
        }
        return str.toString();
    }

    @Override
    public void validate(String msg) {
        int numNodes = 0;

        Stack<HLLDANode> stack = new Stack<HLLDANode>();
        stack.add(root);
        while (!stack.isEmpty()) {
            HLLDANode node = stack.pop();

            numNodes++;
            node.validate(msg);

            int numPassesFromChildren = 0;
            for (HLLDANode child : node.getChildren()) {
                stack.add(child);
                numPassesFromChildren +=
                        child.getNumPassingCustomers()
                        + child.getNumStayingCustomers();
            }

            if (numPassesFromChildren != node.getNumPassingCustomers()) {
                throw new RuntimeException(msg
                        + ". Number of passing customers mismatches. "
                        + " Node " + node.toString()
                        + ". " + numPassesFromChildren
                        + " vs. " + node.getNumPassingCustomers());
            }
        }

        if (numNodes != L + 1) {
            throw new RuntimeException(msg + ". Number of nodes mismatches. "
                    + numNodes + " vs. " + (L + 1));
        }

        logln("Passed validation. " + msg);
    }

    @Override
    public void outputState(String filepath) {
    }

    @Override
    public void inputState(String filepath) {
    }

    /**
     * Get the document frequency of each label
     *
     * @return An array of label's document frequency
     */
    private int[] getLabelFrequencies() {
        int[] labelFreqs = new int[L];
        for (int dd = 0; dd < D; dd++) {
            for (int ii = 0; ii < labels[dd].length; ii++) {
                labelFreqs[labels[dd][ii]]++;
            }
        }
        return labelFreqs;
    }

    /**
     * Get the document frequency of each label pair
     *
     * @return Document frequency of each label pair
     */
    private HashMap<String, Integer> getLabelPairFrequencies() {
        HashMap<String, Integer> pairFreqs = new HashMap<String, Integer>();
        for (int d = 0; d < D; d++) {
            int[] docLabels = labels[d];
            for (int ii = 0; ii < docLabels.length; ii++) {
                for (int jj = 0; jj < docLabels.length; jj++) {
                    if (ii == jj) {
                        continue;
                    }
                    String pair = docLabels[ii] + "-" + docLabels[jj];
                    Integer count = pairFreqs.get(pair);
                    if (count == null) {
                        pairFreqs.put(pair, 1);
                    } else {
                        pairFreqs.put(pair, count + 1);
                    }
                }
            }
        }
        return pairFreqs;
    }

    /**
     * Convert a minimum spanning tree to the label hierarchy
     *
     * @param tree The minimum spanning tree, represented by adjacency list
     * @param graphRoot The root of the minimum spanning tree
     */
    private void setTree(DirectedGraph<Integer> tree, GraphNode<Integer> graphRoot) {
        this.leaves = new ArrayList<HLLDANode>();
        Queue<GraphNode<Integer>> queue = new LinkedList<GraphNode<Integer>>();
        queue.add(graphRoot);
        this.nodeList[L] = this.root;
        while (!queue.isEmpty()) {
            GraphNode<Integer> mstNode = queue.poll();
            HLLDANode node = this.root;
            if (mstNode.getId() < L) {
                node = this.nodeList[mstNode.getId()];
            }

            if (node.getChildren().isEmpty()) {
                this.leaves.add(node);
            }

            if (tree.hasOutEdges(mstNode)) {
                for (GraphEdge edge : tree.getOutEdges(mstNode)) {
                    GraphNode<Integer> mstChild = edge.getTarget();
                    int labelIdx = mstChild.getId();
                    HLLDANode childNode = new HLLDANode(-1,
                            node.getNextChildIndex(),
                            node.getLevel() + 1, labelIdx, node);
                    node.addChild(childNode.getIndex(), childNode);
                    this.nodeList[labelIdx] = childNode;
                    queue.add(mstChild);
                }
            }
        }
    }

    /**
     * Print out the structure (number of observations at each level) of the
     * global tree.
     */
    public String printTreeStructure() {
        StringBuilder str = new StringBuilder();
        Stack<HLLDANode> stack = new Stack<HLLDANode>();
        stack.add(root);

        SparseCount levelNodeCounts = new SparseCount();
        int maxLevel = -1;

        while (!stack.isEmpty()) {
            HLLDANode node = stack.pop();
            levelNodeCounts.increment(node.getLevel());
            if (node.getLevel() > maxLevel) {
                maxLevel = node.getLevel();
            }
            for (HLLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int ii = 0; ii < maxLevel; ii++) {
            str.append(ii).append("\t").append(levelNodeCounts.getCount(ii)).append("\n");
        }
        return str.toString();
    }

    /**
     * Print out the global tree.
     */
    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        Stack<HLLDANode> stack = new Stack<HLLDANode>();
        stack.add(root);

        int totalObs = 0;
        int numNodes = 0;
        SparseCount tokenCountPerLevel = new SparseCount();
        SparseCount nodeCountPerLevel = new SparseCount();

        while (!stack.isEmpty()) {
            HLLDANode node = stack.pop();
            numNodes++;
            tokenCountPerLevel.changeCount(node.getLevel(), node.getNumStayingCustomers());
            nodeCountPerLevel.increment(node.getLevel());

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }

            int numTopWords = 10;
            String[] topWords = getTopWords(node.topic, numTopWords);
            str.append(labelVocab.get(node.getContent()))
                    .append(", ").append(node.toString())
                    .append("\t").append(node.getTotalObservationCount())
                    .append(" ").append(Arrays.toString(topWords))
                    .append("\n");

            totalObs += node.getTotalObservationCount();

            for (HLLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # nodes = ").append(numNodes)
                .append("\n");
        for (int level : tokenCountPerLevel.getSortedIndices()) {
            str.append(">>> level ").append(level)
                    .append(". ").append(nodeCountPerLevel.getCount(level))
                    .append(": ").append(tokenCountPerLevel.getCount(level))
                    .append("\n");
        }
        return str.toString();
    }

    public void outputGlobalTree(File outputFile) throws Exception {
        if (verbose) {
            logln("Outputing global tree to " + outputFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write(this.printGlobalTree());
        writer.close();
    }

    public void outputTopicCoherence(
            File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            logln("Outputing topic coherence to file " + file);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int kk = 0; kk < L; kk++) {
            double[] distribution = nodeList[kk].topic;
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(kk
                    + "\t" + nodeList[kk].getNumStayingCustomers()
                    + "\t" + nodeList[kk].getNumPassingCustomers()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    class HLLDANode extends TreeNode<HLLDANode, Integer> {

        private final int born;
        private int numStayingCustomers; // N_j
        private int numPassingCustomers; // N_{>j}
        private TIntIntHashMap pseudoCounts; // pseudo-counts from children
        private TIntIntHashMap observations; // actual counts
        private double[] topic;

        HLLDANode(int iter,
                int index,
                int level,
                int labelId,
                HLLDANode parent) {
            super(index, level, labelId, parent);
            this.born = iter;
            this.numStayingCustomers = 0;
            this.numPassingCustomers = 0;
            this.observations = new TIntIntHashMap();
            this.pseudoCounts = new TIntIntHashMap();
        }

        public double getLogLikelihood() {
            double llh = 0.0;
            for (int obs : this.observations.keys()) {
                llh += this.observations.get(obs) * Math.log(topic[obs]);
            }
            return llh;
        }

        public void sampleTopic() {
            double[] meanVector = new double[V];
            Arrays.fill(meanVector, hyperparams.get(BETA));
            for (int obs : this.observations.keys()) {
                meanVector[obs] += this.observations.get(obs);
            }
            for (int obs : this.pseudoCounts.keys()) {
                meanVector[obs] += this.pseudoCounts.get(obs);
            }
            if (this.parent != null) {
                for (int v = 0; v < V; v++) {
                    meanVector[v] += this.parent.topic[v] * hyperparams.get(GAMMA);
                }
            }
            Dirichlet dir = new Dirichlet(meanVector);
            this.topic = dir.nextDistribution();
        }

        public void getPseudoCountsFromChildrenMin() {
            this.pseudoCounts = new TIntIntHashMap();
            for (HLLDANode child : this.getChildren()) {
                TIntIntHashMap childObs = child.getObservations();
                for (int o : childObs.keys()) {
                    this.pseudoCounts.adjustOrPutValue(o, 1, 1);
                }
            }
        }

        public void getPseudoCountsFromChildrenMax() {
            this.pseudoCounts = new TIntIntHashMap();
            for (HLLDANode child : this.getChildren()) {
                TIntIntHashMap childObs = child.getObservations();
                for (int o : childObs.keys()) {
                    int v = childObs.get(o);
                    this.pseudoCounts.adjustOrPutValue(o, v, v);
                }
            }
        }

        public void setTopic(double[] t) {
            this.topic = t;
        }

        public double[] getTopic() {
            return this.topic;
        }

        public TIntIntHashMap getPseudoCounts() {
            return this.pseudoCounts;
        }

        public TIntIntHashMap getObservations() {
            return this.observations;
        }

        public int getTotalObservationCount() {
            return StatisticsUtils.sum(observations.getValues());
        }

        public void incrementPseudoCount(int obs) {
            this.pseudoCounts.adjustOrPutValue(obs, 1, 1);
        }

        public void decrementPseudoCount(int obs) {
            boolean found = this.pseudoCounts.adjustValue(obs, -1);
            if (!found) {
                throw new RuntimeException("Removing non-existing key in pseudo-counts.");
            }
            if (this.pseudoCounts.get(obs) == 0) {
                this.pseudoCounts.remove(obs);
            }
        }

        public void incrementObservation(int obs) {
            this.observations.adjustOrPutValue(obs, 1, 1);
        }

        public void decrementObservation(int obs) {
            boolean found = this.observations.adjustValue(obs, -1);
            if (!found) {
                throw new RuntimeException("Removing non-existing key in observations");
            }
            if (this.observations.get(obs) == 0) {
                this.observations.remove(obs);
            }
        }

        public void setLevel(int level) {
            this.level = level;
        }

        public void setIndex(int index) {
            this.index = index;
        }

        public int getIterationCreated() {
            return this.born;
        }

        public int getNumStayingCustomers() {
            return this.numStayingCustomers;
        }

        public void changeNumStayingCustomers(int delta) {
            this.numStayingCustomers += delta;
        }

        public int getNumPassingCustomers() {
            return this.numPassingCustomers;
        }

        public void changeNumPassingCustomers(int delta) {
            this.numPassingCustomers += delta;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append(level)
                    .append(", ").append(index)
                    .append(", ").append(content.toString());
            return str.toString();
        }

        public void validate(String msg) {
            for (int obs : this.observations.keys()) {
                if (this.observations.get(obs) < 0) {
                    throw new RuntimeException(msg + ". Negative count");
                }
            }
        }
    }

    class TreeInitializer {
        // inputs

        private int[] labelFreqs;
        private HashMap<String, Integer> pairFreqs;
        // output
        private DirectedGraph<Integer> tree;
        private GraphNode<Integer> graphRoot;
        // internal
        private GraphNode<Integer>[] nodes;
        private ArrayList<RankingItem<String>> rankPairs;

        public TreeInitializer(int[] lFreqs,
                HashMap<String, Integer> pFreqs) {
            this.labelFreqs = lFreqs;
            this.pairFreqs = pFreqs;

            // initialize label nodes
            nodes = new GraphNode[L + 1];
            for (int ll = 0; ll < L; ll++) {
                nodes[ll] = new GraphNode<Integer>(ll);
            }
            nodes[L] = new GraphNode<Integer>(L);

            // rank edges
            rankPairs = new ArrayList<RankingItem<String>>();
            for (String pair : pairFreqs.keySet()) {
                rankPairs.add(new RankingItem<String>(pair, pairFreqs.get(pair)));
            }
            Collections.sort(rankPairs);
        }

        public GraphNode<Integer> getGraphRoot() {
            return this.graphRoot;
        }

        public DirectedGraph<Integer> getTree() {
            return this.tree;
        }

        public void initializeUsingEntropies() {
            if (verbose) {
                logln("--- --- Creating label graph using entropies ...");
            }

            DirectedGraph<Integer> labelGraph = new DirectedGraph<Integer>();
            int E = rankPairs.size();
            for (int e = 0; e < E; e++) {
                RankingItem<String> pair = rankPairs.get(e);
                int source = Integer.parseInt(pair.getObject().split("-")[0]);
                int target = Integer.parseInt(pair.getObject().split("-")[1]);
                double pairFreq = pair.getPrimaryValue();
                if (pairFreq < 50) {
                    logln("--- --- --- # raw edges: " + e);
                    break;
                }
                int sourceFreq = labelFreqs[source];
                int targetFreq = labelFreqs[target];

                // create edge
                GraphNode<Integer> sourceNode = nodes[source];
                GraphNode<Integer> targetNode = nodes[target];
                double weight = -getConditionalEntropy(
                        targetFreq,
                        sourceFreq,
                        (int) pairFreq, D);
                GraphEdge edge = new GraphEdge(sourceNode, targetNode, weight);
                labelGraph.addEdge(edge);
            }

            // create pseudo-edges from root to all nodes
            graphRoot = nodes[L];
            for (int ll = 0; ll < L; ll++) {
                double weight = -getEntropy(labelFreqs[ll], D);
                GraphEdge edge = new GraphEdge(graphRoot, nodes[ll], weight);
                labelGraph.addEdge(edge);
            }

            if (verbose) {
                logln("--- --- Running directed minimum spanning tree ...");
            }

            EdmondsMST<Integer> dmst = new EdmondsMST<Integer>(graphRoot, labelGraph);
            tree = dmst.getMinimumSpanningTree();
        }

        /**
         * Initialize the tree by generating the label graph using document
         * co-occurrences and extracting minimum spanning tree from the graph.
         */
        public void initializeUsingConditionalProbabilities() {
            if (verbose) {
                logln("--- --- Creating label graph using conditional probabilities ...");
            }

            // create raw label graph
            DirectedGraph<Integer> labelGraph = new DirectedGraph<Integer>();
            int E = 500;
            for (int e = 0; e < E; e++) {
                RankingItem<String> pair = rankPairs.get(e);
                int source = Integer.parseInt(pair.getObject().split("-")[0]);
                int target = Integer.parseInt(pair.getObject().split("-")[1]);
                double pairFreq = pair.getPrimaryValue();
                int sourceFreq = labelFreqs[source];

                // create edge
                GraphNode<Integer> sourceNode = nodes[source];
                GraphNode<Integer> targetNode = nodes[target];
                double weight = 1.0 - (pairFreq / sourceFreq);
                labelGraph.addEdge(sourceNode, targetNode, weight);
            }

            // create pseudo-edges from root to all nodes
            graphRoot = nodes[L];
            for (int ll = 0; ll < L; ll++) {
                double weight = 1.0 - (double) labelFreqs[ll] / D;
                labelGraph.addEdge(graphRoot, nodes[ll], weight);
            }

            if (verbose) {
                logln("--- --- Running Edmonds' directed minimum spanning tree ...");
            }
            EdmondsMST<Integer> dmst = new EdmondsMST<Integer>(graphRoot, labelGraph);
            tree = dmst.getMinimumSpanningTree();
        }

        public void initializeUsingMutualInformation() {
            if (verbose) {
                logln("--- --- Creating label graph using mutual information ...");
            }

            int E = rankPairs.size();
            Set<String> processedEdges = new HashSet<String>();
            UndirectedGraph<Integer> graph = new UndirectedGraph<Integer>();
            for (int e = 0; e < E; e++) {
                RankingItem<String> pair = rankPairs.get(e);
                String[] splitPair = pair.getObject().split("-");
                String reverse = splitPair[1] + "-" + splitPair[0];
                if (processedEdges.contains(reverse)) {
                    continue;
                }
                processedEdges.add(pair.getObject());

                int source = Integer.parseInt(splitPair[0]);
                int target = Integer.parseInt(splitPair[1]);
                double pairFreq = pair.getPrimaryValue();
                int sourceFreq = labelFreqs[source];
                int targetFreq = labelFreqs[target];

                // create edge
                GraphNode<Integer> sourceNode = nodes[source];
                GraphNode<Integer> targetNode = nodes[target];
                double weight = getMutualInformation(sourceFreq, targetFreq, (int) pairFreq, D);

                System.out.println(
                        ">>> " + pair
                        + ". " + sourceFreq
                        + ". " + targetFreq
                        + ". " + pairFreq
                        + ". " + weight);
                graph.addEdge(new GraphEdge(sourceNode, targetNode, -weight));
            }
            graphRoot = nodes[L];
            for (int ll = 0; ll < L; ll++) {
                double weight = getMutualInformation(D, labelFreqs[ll], labelFreqs[ll], D);
//                double weight = labelFreqs[ll];
//                double weight = getEntropy(labelFreqs[ll], D);

                // debug
                System.out.println("<<< root-" + ll
                        + ". " + labelFreqs[ll]
                        + ". " + weight);

                graph.addEdge(new GraphEdge(graphRoot, nodes[ll], -weight));
            }

            PrimMST<Integer> prim = new PrimMST<Integer>(graphRoot, graph);
            tree = prim.getMinimumSpanningTree();
        }
    }

    private static double getEntropy(int nx0, int nx) {
        double entropy = 0.0;
        double px0 = (double) nx0 / nx;
        entropy += px0 * Math.log(px0) / Math.log(2);
        double px1 = 1.0 - px0;
        entropy += px1 * Math.log(px1) / Math.log(2);
        return -entropy;
    }

    // H(Y | X)
    private static double getConditionalEntropy(int nx, int ny, int nxy, int n) {
        double[] px = new double[2];
        px[0] = (double) nx / n;
        px[1] = 1 - px[0];

        double[][] pxy = new double[2][2];
        pxy[0][0] = (double) nxy / n;
        pxy[0][1] = (double) (nx - nxy) / n;
        pxy[1][0] = (double) (ny - nxy) / n;
        pxy[1][1] = 1.0 - pxy[0][0] - pxy[0][1] - pxy[1][0];

        double condEnt = 0.0;
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                if (pxy[ii][jj] != 0) {
                    condEnt += pxy[ii][jj]
                            * Math.log(pxy[ii][jj] / (px[ii])) / Math.log(2);
                }
            }
        }
        return -condEnt;
    }

    private static double getMutualInformation(int nx, int ny, int nxy, int n) {
        double[] px = new double[2];
        px[0] = (double) nx / n;
        px[1] = 1 - px[0];

        double[] py = new double[2];
        py[0] = (double) ny / n;
        py[1] = 1 - py[0];

        double[][] pxy = new double[2][2];
        pxy[0][0] = (double) nxy / n;
        pxy[0][1] = (double) (nx - nxy) / n;
        pxy[1][0] = (double) (ny - nxy) / n;
        pxy[1][1] = 1.0 - pxy[0][0] - pxy[0][1] - pxy[1][0];

        double mi = 0.0;
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                if (pxy[ii][jj] != 0) {
                    mi += pxy[ii][jj] * Math.log(pxy[ii][jj] / (px[ii] * py[jj])) / Math.log(2);
                }
            }
        }
        return mi;
    }

    public static void run(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("dataset", "Dataset");
            addOption("data-folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("format-file", "Formatted file name");
            addOption("output", "Output folder");

            // sampling configurations
            addOption("burnIn", "Burn-in");
            addOption("maxIter", "Maximum number of iterations");
            addOption("sampleLag", "Sample lag");
            addOption("report", "Report interval");

            // model parameters
            addOption("numTopwords", "Number of top words per topic");
            addOption("min-label-freq", "Minimum label frequency");

            addOption("run-mode", "Run mode");

            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("s", false, "standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp dist/segan.jar main.RunHLLDA -help", options);
                return;
            }

//            boolean verbose = cmd.hasOption("verbose");
//            boolean debug = cmd.hasOption("debug");
            boolean verbose = true;
            boolean debug = true;

            if (verbose) {
                System.out.println("\nRunning model");
            }
            String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "112");
            String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "../data");
            String outputFolder = CLIUtils.getStringArgument(cmd, "output", "../data/112/hllda");
            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format-label");
            String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
            int minLabelFreq = CLIUtils.getIntegerArgument(cmd, "min-label-freq", 300);
            int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
            double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
            double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
            double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 0.5);
            double mean = CLIUtils.getDoubleArgument(cmd, "mean", 0.3);
            double scale = CLIUtils.getDoubleArgument(cmd, "scale", 50);

            int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 10);
            int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 20);
            int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
            int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
            boolean paramOpt = cmd.hasOption("paramOpt");

            if (verbose) {
                System.out.println("--- Loading data from " + datasetFolder);
            }
            LabelTextData data = new LabelTextData(datasetName, datasetFolder);
            data.setFormatFilename(formatFile);
            data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            data.filterLabelsByFrequency(minLabelFreq);
//            data.filterDocumentWithoutLabels();
            data.prepareTopicCoherence(numTopWords);
            if (verbose) {
                System.out.println("--- Data loaded.");
                System.out.println("--- --- min-label-freq: " + minLabelFreq);
                System.out.println("--- --- label vocab size: " + data.getLabelVocab().size());
                System.out.println("--- --- # documents: " + data.getWords().length
                        + ". " + data.getLabels().length);
            }

            if (verbose) {
                System.out.println("\nSampling hierarchical labeled LDA ...");
            }
            HierarchicalLabeledLDABackup sampler = new HierarchicalLabeledLDABackup();
            sampler.setVerbose(verbose);
            sampler.setDebug(debug);
            sampler.setWordVocab(data.getWordVocab());
            sampler.setLabelVocab(data.getLabelVocab());

            sampler.configure(outputFolder, data.getWords(), data.getLabels(),
                    data.getWordVocab().size(), data.getLabelVocab().size(),
                    alpha, beta, gamma, mean, scale,
                    InitialState.RANDOM,
                    PathAssumption.MINIMAL,
                    paramOpt,
                    burnIn, maxIters, sampleLag, repInterval);

            File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
            IOUtils.createFolder(samplerFolder);
            sampler.initialize();
            sampler.outputGlobalTree(new File(samplerFolder, "init-global-tree.txt"));
            sampler.outputTopicCoherence(new File(samplerFolder, "init-topic-coherence.txt"),
                    data.getTopicCoherence());

            sampler.iterate();
            sampler.outputGlobalTree(new File(samplerFolder, "final-global-tree.txt"));
            sampler.outputTopicCoherence(new File(samplerFolder, "final-topic-coherence.txt"),
                    data.getTopicCoherence());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running in " + Class.class.getName());
        }
    }

    public static void main(String[] args) {
        run(args);

//        TIntIntHashMap obs = new TIntIntHashMap();
//        obs.adjustOrPutValue(1, 3, 3);
//        System.out.println(obs.size());
//        for (int o : obs.keys()) {
//            System.out.println(o + "\t" + obs.get(o));
//        }
    }

    private static void test() {
        HierarchicalLabeledLDABackup sampler = new HierarchicalLabeledLDABackup();
    }
}
