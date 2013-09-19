/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
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
public class MultiLabelTextData extends TextDataset {

    public static final String labelVocabExt = ".lvoc";
    public static final int ALL_LABELS = -1;
    protected ArrayList<String>[] labelList;
    protected ArrayList<String> labelVocab;
    protected int[][] labels;

    public MultiLabelTextData(String name, String folder) {
        super(name, folder);
    }

    public MultiLabelTextData(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public int[][] getLabels() {
        return this.labels;
    }

    public ArrayList<String> getLabelVocab() {
        return this.labelVocab;
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
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);

        logln("--- Creating label vocab ...");
        createLabelVocab();

        File labelVocFile = new File(outputFolder, name + labelVocabExt);
        logln("--- Outputing label vocab ... " + labelVocFile.getAbsolutePath());
        DataUtils.outputVocab(labelVocFile.getAbsolutePath(),
                this.labelVocab);

        // get label indices
        this.labels = new int[this.labelList.length][];
        for (int ii = 0; ii < labels.length; ii++) {
            this.labels[ii] = new int[labelList[ii].size()];
            for (int jj = 0; jj < labels[ii].length; jj++) {
                int labelIndex = Collections.binarySearch(labelVocab, labelList[ii].get(jj));
                this.labels[ii][jj] = labelIndex;
            }
        }

        // perform normal processing
        super.format(outputFolder);
    }

    public void createLabelVocab() throws Exception {
        createLabelVocabByFrequency(ALL_LABELS);
    }

    private void createLabelVocabByFrequency(int topK) throws Exception {
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
            rankLabels.add(new RankingItem<String>(label, labelFreqs.get(label)));
        }
        Collections.sort(rankLabels);

        if (topK == ALL_LABELS) {
            topK = rankLabels.size();
        }
        this.labelVocab = new ArrayList<String>();
        for (int k = 0; k < Math.min(topK, rankLabels.size()); k++) {
            this.labelVocab.add(rankLabels.get(k).getObject());
        }
        Collections.sort(this.labelVocab);
    }

    public void outputLabelVocab(String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (String label : labelVocab) {
            writer.write(label + "\n");
        }
        writer.close();
    }

    @Override
    protected void outputInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, name + docInfoExt);
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
            this.inputLabelVocab(new File(fFolder, name + labelVocabExt));
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
