package sampler.labeled;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
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
public class TFNN {
    protected int[][] words;
    protected int[][] labels;
    protected int L;
    protected int V;
    protected int D;
    protected SparseVector[] labelVectors; // L x V;
    protected int minWordTypeCount = 0;
    protected double[] labelL2Norms;
    
    public TFNN(
            int[][] docWords,
            int[][] labels,
            int L,
            int V,
            int minWordTypeCount) {
        this.words = docWords;
        this.labels = labels;
        this.L = L;
        this.V = V;
        this.D = this.words.length;
        this.minWordTypeCount = minWordTypeCount;
    }
    
    public String getName() {
        return "tf-nn-" + minWordTypeCount;
    }
    
    public SparseVector[] getLabelVectors() {
        return this.labelVectors;
    }

    public void setMinWordTypeCount(int minTypeCount) {
        this.minWordTypeCount = minTypeCount;
    }
    
    public void learn() {
        this.labelVectors = new SparseVector[L];
        for (int ll = 0; ll < L; ll++) {
            this.labelVectors[ll] = new SparseVector();
        }
        int[] labelDocCounts = new int[L];
        System.out.println("Aggregate label vectors ...");
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

            SparseVector docVector = new SparseVector();
            for (int idx : typeCount.getIndices()) {
                double score = (double)typeCount.getCount(idx) / words[d].length;
                docVector.set(idx, score);
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
//            double tfidf = typeCount.getCount(idx) * idfs[idx];
            double val = (double) typeCount.getCount(idx) / newWords.length;
            docVector.set(idx, val);
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
    
    public void outputPredictor(File predictorFile) {
        System.out.println("Outputing learned model to " + predictorFile);
        try {
            StringBuilder labelVecStr = new StringBuilder();
            labelVecStr.append("num-labels\t").append(L).append("\n");
            labelVecStr.append("num-dimensions\t").append(V).append("\n");
            for (int l = 0; l < this.labelVectors.length; l++) {
                labelVecStr.append(this.labelVectors[l].toString()).append("\n");
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(predictorFile.getName());
            ZipOutputStream writer = IOUtils.getZipOutputStream(predictorFile.getAbsolutePath());

            ZipEntry modelEntry = new ZipEntry(filename + ".label");
            writer.putNextEntry(modelEntry);
            byte[] data = labelVecStr.toString().getBytes();
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

            labelL2Norms = new double[L];
            for (int ll = 0; ll < L; ll++) {
                labelL2Norms[ll] = labelVectors[ll].getL2Norm();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictor from "
                    + predictorFile);
        }
    }
    
    public void outputTopWords(File outputFile,
            ArrayList<String> labelVocab,
            ArrayList<String> wordVocab,
            int numTopWords) {
        System.out.println("Outputing top words to " + outputFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int ll = 0; ll < L; ll++) {
                ArrayList<RankingItem<Integer>> rankWords = new ArrayList<RankingItem<Integer>>();
                for (int v : labelVectors[ll].getIndices()) {
                    rankWords.add(new RankingItem<Integer>(v, labelVectors[ll].get(v)));
                }
                Collections.sort(rankWords);

                String topicStr = "Label-" + ll;
                if (labelVocab != null) {
                    topicStr = labelVocab.get(ll);
                }
                writer.write(topicStr);

                for (int ii = 0; ii < numTopWords; ii++) {
                    RankingItem<Integer> item = rankWords.get(ii);
                    writer.write("\t" + wordVocab.get(item.getObject()));
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + outputFile);
        }
    }
}
