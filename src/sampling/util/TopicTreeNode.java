package sampling.util;

import cc.mallet.types.Dirichlet;
import java.util.Arrays;
import sampling.likelihood.DirMult;

/**
 * Implementation of tree node which stores a topic (i.e., a multinomial
 * distribution over the vocabulary)
 *
 * @author vietan
 */
public class TopicTreeNode<N extends TopicTreeNode, C extends DirMult> extends TreeNode<N, C> {

    public static enum PathAssumption {

        MINIMAL, MAXIMAL
    }
    private SparseCount pseudoCounts; // pseudo observations from children

    public TopicTreeNode(int index, int level, C content, N parent) {
        super(index, level, content, parent);
    }

    /**
     * Get the sampling topic (without collapsed).
     */
    public double[] getTopic() {
        return this.content.getSamplingDistribution();
    }

    /**
     * Propagate the observations from all children nodes to this node using
     * minimal path assumption, which means for each observation type v, a child
     * node will propagate a value of 1 if it contains v, and 0 otherwise.
     */
    public void getPseudoCountsFromChildrenMin() {
        this.pseudoCounts = new SparseCount();
        for (TopicTreeNode child : this.getChildren()) {
            SparseCount childObs = ((C) child.getContent()).getSparseCounts();
            for (int obs : childObs.getIndices()) {
                this.pseudoCounts.increment(obs);
            }
        }
    }

    /**
     * Propagate the observations from all children nodes to this node using
     * maximal path assumption, which means that each child node will propagate
     * its full observation vector.
     */
    public void getPseudoCountsFromChildrenMax() {
        this.pseudoCounts = new SparseCount();
        for (TopicTreeNode child : this.getChildren()) {
            SparseCount childObs = ((C) child.getContent()).getSparseCounts();
            for (int obs : childObs.getIndices()) {
                this.pseudoCounts.changeCount(obs, childObs.getCount(obs));
            }
        }
    }

    /**
     * Sample topic. This applies since the topic of a node is modeled as a
     * drawn from a Dirichlet distribution with the mean vector is the topic of
     * the node's parent and scaling factor gamma.
     *
     * @param beta Topic smoothing parameter
     * @param gamma Dirichlet-Multinomial chain parameter
     */
    public void sampleTopic(double beta, double gamma) {
        int V = content.getDimension();
        double[] meanVector = new double[V];
        Arrays.fill(meanVector, beta);
        SparseCount observations = this.content.getSparseCounts();
        for (int obs : observations.getIndices()) {
            meanVector[obs] += observations.getCount(obs);
        }

        for (int obs : this.pseudoCounts.getIndices()) {
            meanVector[obs] += this.pseudoCounts.getCount(obs);
        }
        if (this.parent != null) {
            double[] parentTopic = ((C) parent.getContent()).getSamplingDistribution();
            for (int v = 0; v < V; v++) {
                meanVector[v] += parentTopic[v] * gamma;
            }
        }
        Dirichlet dir = new Dirichlet(meanVector);
        double[] topic = dir.nextDistribution();
        this.content.setSamplingDistribution(topic);
    }
}
