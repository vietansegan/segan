package sampler.labeled;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampling.util.SparseCount;
import util.IOUtils;
import util.RankingItem;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class TFIDF {

    protected int[][] words;
    protected int[][] labels;
    protected int L;
    protected int V;
    protected int D;
    protected double[] idfs; // V-dim vector
    protected SparseVector[] labelVectors; // L x V;
    protected int minWordTypeCount = 5;
    protected double[] labelL2Norms;

    public TFIDF() {
    }

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

    public String getName() {
        return "tf-idf";
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
        System.out.println("Estimating DFs ...");
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
        System.out.println("Aggregate label vectors ...");
        for (int d = 0; d < D; d++) {
            if (d % 100000 == 0) {
                System.out.println("--- d = " + d);
            }
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
        System.out.println("Averaging ...");
        for (int ll = 0; ll < L; ll++) {
            int docCount = labelDocCounts[ll];
            if (docCount > 0) {
                this.labelVectors[ll].divide(docCount);
            }
        }

        computeLabelL2Norms();
    }

    protected void computeLabelL2Norms() {
        labelL2Norms = new double[L];
        for (int ll = 0; ll < L; ll++) {
            labelL2Norms[ll] = labelVectors[ll].getL2Norm();
        }
    }

    /**
     * Predict topics for a given document
     *
     * @param newWords The token vector of the test document
     * @return A vector of length L (i.e., number of labels) specifying the
     * score of each label predicted for the given document. The scores are in
     * [0, 1].
     */
    public double[] predict(int[] newWords) {
        double[] scores = new double[L];
        if (newWords.length == 0) {
            return scores;
        }

        SparseCount typeCount = new SparseCount();
        for (int n = 0; n < newWords.length; n++) {
            typeCount.increment(newWords[n]);
        }

        SparseVector docVector = new SparseVector();
        for (int idx : typeCount.getIndices()) {
            double tfidf = typeCount.getCount(idx) * idfs[idx];
            docVector.set(idx, tfidf);
        }

        double newDocL2Norm = docVector.getL2Norm();

        for (int l = 0; l < L; l++) {
            if (labelVectors[l].size() > 0) { // skip topics that didn't have enough training data for
                scores[l] = labelVectors[l].dotProduct(docVector)
                        / (labelL2Norms[l] * newDocL2Norm);
            }
        }
        return scores;
    }

    public ArrayList<Integer> predictLabel(int[] newWords, int topK) {
        double[] scores = predict(newWords);
        ArrayList<RankingItem<Integer>> rank = new ArrayList<RankingItem<Integer>>();
        for (int ii = 0; ii < scores.length; ii++) {
            rank.add(new RankingItem<Integer>(ii, scores[ii]));
        }
        Collections.sort(rank);
        ArrayList<Integer> rankLabels = new ArrayList<Integer>();
        for (int ii = 0; ii < topK; ii++) {
            RankingItem<Integer> item = rank.get(ii);
            int label = item.getObject();
            double score = item.getPrimaryValue();
            if (score == 0.0) {
                break;
            }
            rankLabels.add(label);
        }
        return rankLabels;
    }

    public void outputPredictorTextFile(File predFile) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(predFile);
            writer.write("num-labels\t" + L + "\n");
            writer.write("num-dimension\t" + V + "\n");
            for (int l = 0; l < this.labelVectors.length; l++) {
                writer.write(labelVectors[l].toString() + "\n");
            }
            for (int v = 0; v < V; v++) {
                writer.write(this.idfs[v] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing model to "
                    + predFile);
        }
    }

    public void inputPredictorTextFile(File predFile) {
        try {
            BufferedReader reader = IOUtils.getBufferedReader(predFile);
            L = Integer.parseInt(reader.readLine().split("\t")[1]);
            V = Integer.parseInt(reader.readLine().split("\t")[1]);
            this.labelVectors = new SparseVector[L];
            for (int l = 0; l < L; l++) {
                labelVectors[l] = SparseVector.parseString(reader.readLine());
            }
            this.idfs = new double[V];
            for (int v = 0; v < V; v++) {
                this.idfs[v] = Double.parseDouble(reader.readLine());
            }
            reader.close();

            labelL2Norms = new double[L];
            for (int ll = 0; ll < L; ll++) {
                labelL2Norms[ll] = labelVectors[ll].getL2Norm();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + predFile);
        }
    }

    public void outputPredictor(File predictorFile) {
        System.out.println("Outputing learned model to " + predictorFile);
        try {
            StringBuilder labelVecStr = new StringBuilder();
            labelVecStr.append("num-labels\t").append(L).append("\n");
            labelVecStr.append("num-dimensions\t").append(V).append("\n");
            for (int l = 0; l < this.labelVectors.length; l++) {
                labelVecStr.append(this.labelVectors[l].toString()).append("\n");
            }

            StringBuilder docFreqStr = new StringBuilder();
            docFreqStr.append("V\t").append(V).append("\n");
            for (int v = 0; v < V; v++) {
                docFreqStr.append(this.idfs[v]).append("\n");
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(predictorFile.getName());
            ZipOutputStream writer = IOUtils.getZipOutputStream(predictorFile.getAbsolutePath());

            ZipEntry modelEntry = new ZipEntry(filename + ".label");
            writer.putNextEntry(modelEntry);
            byte[] data = labelVecStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            ZipEntry assignEntry = new ZipEntry(filename + ".docfreq");
            writer.putNextEntry(assignEntry);
            data = docFreqStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing model to "
                    + predictorFile);
        }
    }

    public void inputPredictor(File predictorFile) {
        System.out.println("Inputing learned model from " + predictorFile);
        try {
            String filename = IOUtils.removeExtension(predictorFile.getName());

            ZipFile zipFile = new ZipFile(predictorFile);
            BufferedReader reader = IOUtils.getBufferedReader(zipFile,
                    zipFile.getEntry(filename + ".label"));
            L = Integer.parseInt(reader.readLine().split("\t")[1]);
            V = Integer.parseInt(reader.readLine().split("\t")[1]);
            this.labelVectors = new SparseVector[L];
            for (int l = 0; l < L; l++) {
                labelVectors[l] = SparseVector.parseString(reader.readLine());
            }
            reader.close();

            reader = IOUtils.getBufferedReader(zipFile,
                    zipFile.getEntry(filename + ".docfreq"));
            if (V != Integer.parseInt(reader.readLine().split("\t")[1])) {
                throw new RuntimeException("Mismatch");
            }
            this.idfs = new double[V];
            for (int v = 0; v < V; v++) {
                this.idfs[v] = Double.parseDouble(reader.readLine());
            }
            reader.close();

            labelL2Norms = new double[L];
            for (int ll = 0; ll < L; ll++) {
                labelL2Norms[ll] = labelVectors[ll].getL2Norm();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
