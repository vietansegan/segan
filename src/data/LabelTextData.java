package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import util.DataUtils;
import util.IOUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public class LabelTextData extends TextDataset {

    public static final String labelVocabExt = ".lvoc";
    protected ArrayList<String>[] labelList;
    protected ArrayList<String> labelVocab;
    protected int[][] labels;
    protected int maxLabelVocSize = Integer.MAX_VALUE;
    protected int minLabelDocFreq = 1;

    public LabelTextData(String name, String folder) {
        super(name, folder);
    }

    public LabelTextData(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public void setMaxLabelVocabSize(int L) {
        this.maxLabelVocSize = L;
    }

    public void setMinLabelDocFreq(int f) {
        this.minLabelDocFreq = f;
    }

    public int[][] getLabels() {
        return this.labels;
    }

    public ArrayList<String> getLabelVocab() {
        return this.labelVocab;
    }

    public void setLabelVocab(ArrayList<String> lVoc) {
        this.labelVocab = lVoc;
    }

    public ArrayList<String>[] getLabelList() {
        return this.labelList;
    }

    /**
     * Filter document labels. The remaining labels only come from a given set
     * of labels
     *
     * @param labVoc The given set of labels
     */
    public void filterLabels(ArrayList<String> labVoc) {
        int D = words.length;
        this.labelVocab = labVoc;

        int[][] filterLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            ArrayList<Integer> docFilterLabels = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String label = labelVocab.get(labels[d][ii]);
                int filterLabelIndex = this.labelVocab.indexOf(label);
                if (filterLabelIndex >= 0) {
                    docFilterLabels.add(filterLabelIndex);
                }
            }

            filterLabels[d] = new int[docFilterLabels.size()];
            for (int ii = 0; ii < docFilterLabels.size(); ii++) {
                filterLabels[d][ii] = docFilterLabels.get(ii);
            }
        }

        this.labels = filterLabels;
    }

    /**
     * Filter labels that do not meet the minimum frequency requirement.
     *
     * @param minLabelFreq Minimum frequency
     */
    public void filterLabelsByFrequency(int minLabelFreq) {
        int D = words.length;
        int L = labelVocab.size();
        int[] labelFreqs = new int[L];
        for (int dd = 0; dd < D; dd++) {
            for (int ii = 0; ii < labels[dd].length; ii++) {
                labelFreqs[labels[dd][ii]]++;
            }
        }

        ArrayList<String> filterLabelVocab = new ArrayList<String>();
        for (int ll = 0; ll < L; ll++) {
            if (labelFreqs[ll] > minLabelFreq) {
                filterLabelVocab.add(labelVocab.get(ll));
            }
        }
        Collections.sort(filterLabelVocab);

        int[][] filterLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            ArrayList<Integer> docFilterLabels = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String label = labelVocab.get(labels[d][ii]);
                int filterLabelIndex = filterLabelVocab.indexOf(label);
                if (filterLabelIndex >= 0) {
                    docFilterLabels.add(filterLabelIndex);
                }
            }

            filterLabels[d] = new int[docFilterLabels.size()];
            for (int ii = 0; ii < docFilterLabels.size(); ii++) {
                filterLabels[d][ii] = docFilterLabels.get(ii);
            }
        }

        this.labels = filterLabels;
        this.labelVocab = filterLabelVocab;
    }

//    public void filterDocumentWithoutLabels() {
//        logln("Filtering out documents without labels");
//        int count = 0;
//        for (int d = 0; d < labels.length; d++) {
//            if (labels[d].length == 0) {
//                // TODO
//                count++;
//            }
//        }
//        logln("--- # documents without labels: " + count);
//    }

    public void loadLabels(File labelFile) throws Exception {
        loadLabels(labelFile.getAbsolutePath());
    }

    public void loadLabels(String labelFile) throws Exception {
        logln("--- Loading labels from " + labelFile);

        if (this.docIdList == null) {
            throw new RuntimeException("docIdList is null. Load text data first.");
        }

        HashMap<String, ArrayList<String>> docLabelMap = new HashMap<String, ArrayList<String>>();
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(labelFile);
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            String docId = sline[0];

            ArrayList<String> docLabels = new ArrayList<String>();
            for (int ii = 1; ii < sline.length; ii++) {
                docLabels.add(sline[ii]);
            }
            docLabelMap.put(docId, docLabels);
        }
        reader.close();

        this.labelList = new ArrayList[docIdList.size()];
        for (int ii = 0; ii < docIdList.size(); ii++) {
            ArrayList<String> docLabels = docLabelMap.get(docIdList.get(ii));
            this.labelList[ii] = docLabels;
        }
    }

    @Override
    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);

        formatLabels(outputFolder);

        // perform normal processing
        super.format(outputFolder);
    }

    public void formatLabels(String outputFolder) throws Exception {
        logln("Formatting labels ...");
        if (this.labelVocab == null) {
            createLabelVocab();
        }

        // output label vocab
        outputLabelVocab(outputFolder);

        // get label indices
        this.labels = new int[this.labelList.length][];
        for (int ii = 0; ii < labels.length; ii++) {
            ArrayList<Integer> docLabels = new ArrayList<Integer>();
            for (int jj = 0; jj < labelList[ii].size(); jj++) {
                int labelIndex = labelVocab.indexOf(labelList[ii].get(jj));
                if (labelIndex >= 0) { // filter out labels not in label vocab
                    docLabels.add(labelIndex);
                }
            }

            this.labels[ii] = new int[docLabels.size()];
            for (int jj = 0; jj < labels[ii].length; jj++) {
                this.labels[ii][jj] = docLabels.get(jj);
            }
        }
    }

    /**
     * Output the list of unique labels
     *
     * @param outputFolder Output folder
     */
    protected void outputLabelVocab(String outputFolder) throws Exception {
        File labelVocFile = new File(outputFolder, formatFilename + labelVocabExt);
        logln("--- Outputing label vocab ... " + labelVocFile.getAbsolutePath());
        DataUtils.outputVocab(labelVocFile.getAbsolutePath(),
                this.labelVocab);
    }

    /**
     * Create label vocabulary
     */
    public void createLabelVocab() throws Exception {
        logln("--- Creating label vocab ...");
        createLabelVocabByFrequency();
    }

    protected void createLabelVocabByFrequency() throws Exception {
        HashMap<String, Integer> labelFreqs = new HashMap<String, Integer>();
        for (int ii = 0; ii < this.labelList.length; ii++) {
            for (String label : this.labelList[ii]) {
                Integer count = labelFreqs.get(label);
                if (count == null) {
                    labelFreqs.put(label, 1);
                } else {
                    labelFreqs.put(label, count + 1);
                }
            }
        }

        ArrayList<RankingItem<String>> rankLabels = new ArrayList<RankingItem<String>>();
        for (String label : labelFreqs.keySet()) {
            int freq = labelFreqs.get(label);
            if (freq >= this.minLabelDocFreq) {
                rankLabels.add(new RankingItem<String>(label, labelFreqs.get(label)));
            }
        }
        Collections.sort(rankLabels);

        this.labelVocab = new ArrayList<String>();
        for (int k = 0; k < Math.min(this.maxLabelVocSize, rankLabels.size()); k++) {
            this.labelVocab.add(rankLabels.get(k).getObject());
        }
        Collections.sort(this.labelVocab);
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        logln("--- Outputing document info ... " + outputFile);

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex));
            for (int label : labels[docIndex]) {
                infoWriter.write("\t" + label);
            }
            infoWriter.write("\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File file) throws Exception {
        logln("--- Reading document info from " + file);

        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<int[]> labelIndexList = new ArrayList<int[]>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            int[] labelIndices = new int[sline.length - 1];
            for (int ii = 0; ii < sline.length - 1; ii++) {
                labelIndices[ii] = Integer.parseInt(sline[ii + 1]);
            }
            labelIndexList.add(labelIndices);
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.labels = new int[labelIndexList.size()][];
        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii] = labelIndexList.get(ii);
        }
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            this.inputLabelVocab(new File(fFolder, formatFilename + labelVocabExt));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    protected void inputLabelVocab(File file) throws Exception {
        labelVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        while ((line = reader.readLine()) != null) {
            labelVocab.add(line);
        }
        reader.close();
    }
}
