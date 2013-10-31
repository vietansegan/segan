package sampler.labeled;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import sampling.util.SparseCount;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class TFIDF {

    private int[][] words;
    private int[][] labels;
    private int L;
    private int V;
    private int D;
    protected double[] idfs; // V-dim vector
    protected SparseVector[] labelVectors; // L x V;
    private int minWordTypeCount = 5;

    public TFIDF(
            int[][] docWords,
            int[][] labels,
            int L,
            int V) {
        this.words = docWords;
        this.labels = labels;
        this.L = L;
        this.V = V;
        this.D = this.words.length;
    }

    public TFIDF(
            ArrayList<int[]> docWords,
            ArrayList<int[]> docLabels,
            int L,
            int V) {
        this.L = L;
        this.V = V;
        this.D = docWords.size();
        this.words = new int[docWords.size()][];
        this.labels = new int[docLabels.size()][];
        for (int d = 0; d < D; d++) {
            this.words[d] = docWords.get(d);
            this.labels[d] = docLabels.get(d);
        }
    }

    public SparseVector[] getLabelVectors() {
        return this.labelVectors;
    }

    public void setMinWordTypeCount(int minTypeCount) {
        this.minWordTypeCount = minTypeCount;
    }

    public double[] getIdfs() {
        return this.idfs;
    }

    public void learn() {
        // estimate doc-frequencies of each word type
        int[] dfs = new int[V];
        for (int d = 0; d < D; d++) {
            Set<Integer> uniqueWords = new HashSet<Integer>();
            for (int n = 0; n < words[d].length; n++) {
                uniqueWords.add(words[d][n]);
            }
            for (int uw : uniqueWords) {
                dfs[uw]++;
            }
        }

        idfs = new double[V];
        for (int v = 0; v < V; v++) {
            idfs[v] = Math.log(this.D) - Math.log(dfs[v] + 1);
        }

        this.labelVectors = new SparseVector[L];
        for (int ll = 0; ll < L; ll++) {
            this.labelVectors[ll] = new SparseVector();
        }
        int[] labelDocCounts = new int[L];
        for (int d = 0; d < D; d++) {
            int[] docTopics = this.labels[d];
            // skip unlabeled document or very short (after filtered) documents
            if (docTopics == null
                    || docTopics.length == 0
                    || words[d].length < minWordTypeCount) {
                continue;
            }

            SparseCount typeCount = new SparseCount();
            for (int n = 0; n < words[d].length; n++) {
                typeCount.increment(words[d][n]);
            }

            // max tf
            int maxTf = -1;
            for (int idx : typeCount.getIndices()) {
                int tf = typeCount.getCount(idx);
                if (maxTf < tf) {
                    maxTf = tf;
                }
            }

            SparseVector docVector = new SparseVector();
            for (int idx : typeCount.getIndices()) {
                double tf = 0.5 + 0.5 * typeCount.getCount(idx) / maxTf;
                double idf = idfs[idx];
                double tfidf = tf * idf;
                docVector.set(idx, tfidf);
            }

            for (int ll : docTopics) {
                labelDocCounts[ll]++;
                this.labelVectors[ll].add(docVector);
            }
        }

        // average
        for (int ll = 0; ll < L; ll++) {
            int docCount = labelDocCounts[ll];
            if (docCount > 0) {
                this.labelVectors[ll].divide(docCount);
            }
        }
    }
}
