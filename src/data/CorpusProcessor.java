/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import main.GlobalConstants;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import util.IOUtils;
import util.RankingItem;
import util.Stemmer;
import util.StopwordRemoval;
import weka.core.stemmers.SnowballStemmer;

/**
 *
 * @author vietan
 */
public class CorpusProcessor {

    private ChiSquareTest chiSquareTest;
    private boolean verbose = true;
    // inputs
    private String[] rawTexts;
    private Set<String> excludeFromBigrams; // set of bigrams that should not be considered
    // settings
    protected int unigramCountCutoff; // minimum count of raw unigrams
    protected int bigramCountCutoff; // minimum count of raw bigrams
    protected double bigramScoreCutoff; // minimum bigram score
    protected int maxVocabSize; // maximum vocab size
    protected int vocabTermFreqMinCutoff; // minimum term frequency (for all items in the vocab, including unigrams and bigrams)
    protected int vocabTermFreqMaxCutoff; // maximum term frequency
    protected int vocabDocFreqMinCutoff; // minimum document frequency (for all items in the vocab, including unigrams and bigrams)
    protected int vocabDocFreqMaxCutoff; // maximum document frequency 
    protected int docTypeCountCutoff;  // minumum number of types in a document
    protected boolean filterStopwords = true; // whether stopwords are filtered
    protected boolean lemmatization = false; // whether lemmatization should be performed
    // tools
    protected Tokenizer tokenizer;
    protected SentenceDetector sentenceDetector;
    private StopwordRemoval stopwordRemoval;
    private Stemmer stemmer;
    private HashMap<String, Integer> termFreq;
    private HashMap<String, Integer> docFreq;
    private HashMap<String, Integer> leftFreq;
    private HashMap<String, Integer> rightFreq;
    private HashMap<String, Integer> bigramFreq;
    private int totalBigram;
    // output data after processing
    private ArrayList<String> vocabulary;
    private int[][] numericDocs;
    private int[][][] numericSentences;
    private String[][] rawSentences;
    private Pattern p = Pattern.compile("\\p{Punct}");

    public CorpusProcessor(
            int unigramCountCutoff, // 
            int bigramCountCutoff, // 
            double bigramScoreCutoff, //
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            boolean filterStopwords,
            boolean lemmatization) {
        this.chiSquareTest = new ChiSquareTest();
        this.excludeFromBigrams = new HashSet<String>();

        // settings
        this.unigramCountCutoff = unigramCountCutoff;
        this.bigramCountCutoff = bigramCountCutoff;
        this.bigramScoreCutoff = bigramScoreCutoff;
        this.maxVocabSize = maxVocabSize;

        this.vocabTermFreqMinCutoff = vocTermFreqMinCutoff;
        this.vocabTermFreqMaxCutoff = vocTermFreqMaxCutoff;

        this.vocabDocFreqMinCutoff = vocDocFreqMinCutoff;
        this.vocabDocFreqMaxCutoff = vocDocFreqMaxCutoff;

        this.docTypeCountCutoff = docTypeCountCutoff;
        this.filterStopwords = filterStopwords;
        this.lemmatization = lemmatization;

        this.termFreq = new HashMap<String, Integer>();
        this.docFreq = new HashMap<String, Integer>();

        this.leftFreq = new HashMap<String, Integer>();
        this.rightFreq = new HashMap<String, Integer>();
        this.bigramFreq = new HashMap<String, Integer>();
        this.totalBigram = 0;

        try {
            this.stemmer = new Stemmer();
            if (lemmatization) {
                this.stopwordRemoval = new StopwordRemoval(stemmer);
            } else {
                this.stopwordRemoval = new StopwordRemoval();
            }

            // initiate tokenizer
            InputStream tokenizeIn = new FileInputStream(GlobalConstants.tokenizerFilePath);
            TokenizerModel tokenizeModel = new TokenizerModel(tokenizeIn);
            this.tokenizer = new TokenizerME(tokenizeModel);
            tokenizeIn.close();

            InputStream tokenizeSent = new FileInputStream(GlobalConstants.sentDetectorFilePath);
            SentenceModel sentenceModel = new SentenceModel(tokenizeSent);
            this.sentenceDetector = new SentenceDetectorME(sentenceModel);
            tokenizeSent.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void setRawTexts(String[] rawTexts) {
        this.rawTexts = rawTexts;
    }

    public void setBigramExcluded(Set<String> exclude) {
        this.excludeFromBigrams = exclude;
    }

    public String getSettings() {
        StringBuilder str = new StringBuilder();
        str.append("Raw unigram count cut-off:\t").append(unigramCountCutoff).append("\n");
        str.append("Raw bigram count cut-off:\t").append(bigramCountCutoff).append("\n");
        str.append("Bigram score cut-off:\t").append(bigramScoreCutoff).append("\n");
        str.append("Vocab term frequency cut-off:\t").append(vocabTermFreqMinCutoff).append("\n");
        str.append("Vocab doc frequency cut-off:\t").append(vocabDocFreqMinCutoff).append("\n");
        str.append("Max vocab size:\t").append(maxVocabSize).append("\n");
        str.append("Filter stopwords:\t").append(filterStopwords).append("\n");
        str.append("Lemmatization:\t").append(lemmatization);
        return str.toString();
    }

    public void setVerbose(boolean v) {
        this.verbose = v;
    }

    public void setMaxVocabSize(int vocsize) {
        this.maxVocabSize = vocsize;
    }

    public void setStopwordFilter(boolean filter) {
        this.filterStopwords = filter;
    }

    public void setLemmatize(boolean stem) {
        this.lemmatization = stem;
    }

    public ArrayList<String> getVocab() {
        return this.vocabulary;
    }

    public int[][] getNumerics() {
        return this.numericDocs;
    }

    public int[][][] getNumericSentences() {
        return this.numericSentences;
    }

    public String[][] getRawSentences() {
        return this.rawSentences;
    }

    public void process() {
        if (rawTexts == null) {
            throw new RuntimeException("Raw texts have not been initialized yet");
        }

        // tokenize and normalize texts
        if (verbose) {
            System.out.println("Tokenizing and counting ...");
        }
        String[][][] normTexts = new String[rawTexts.length][][];
        this.rawSentences = new String[rawTexts.length][];
        this.numericSentences = new int[rawTexts.length][][];

        for (int d = 0; d < rawTexts.length; d++) {
            if (verbose && d % 500 == 0) {
                System.out.println("--- Tokenizing doc # " + d + " / " + rawTexts.length);
            }

            Set<String> uniqueDocTokens = new HashSet<String>();

            String rawText = rawTexts[d];
            this.rawSentences[d] = sentenceDetector.sentDetect(rawText);

            normTexts[d] = new String[this.rawSentences[d].length][];

            for (int s = 0; s < this.rawSentences[d].length; s++) {
                String[] sentTokens = tokenizer.tokenize(this.rawSentences[d][s].toLowerCase());
                normTexts[d][s] = new String[sentTokens.length];

                for (int t = 0; t < sentTokens.length; t++) {
                    String normToken = normalize(sentTokens[t]);
                    normTexts[d][s][t] = normToken;

                    if (!normToken.isEmpty()) {
                        incrementMap(termFreq, normToken);
                        uniqueDocTokens.add(normToken);

                        if (t - 1 >= 0 && !normTexts[d][s][t - 1].isEmpty()) {
                            String preToken = normTexts[d][s][t - 1];
                            incrementMap(leftFreq, preToken);
                            incrementMap(rightFreq, normToken);
                            incrementMap(bigramFreq, getBigramString(preToken, normToken));
                            totalBigram++;
                        }
                    }
                }
            }

            for (String uniToken : uniqueDocTokens) {
                incrementMap(docFreq, uniToken);
            }
        }

        // debug
        if (verbose) {
            System.out.println("--- # raw unique unigrams: " + termFreq.size() + ". " + docFreq.size());
            System.out.println("--- # raw unique bigrams: " + bigramFreq.size());
            System.out.println("--- # left: " + leftFreq.size() + ". # right: " + rightFreq.size() + ". total: " + totalBigram);
        }

        // score bigrams
        if (verbose) {
            System.out.println("Scoring bigram ...");
        }
        Set<String> vocab = new HashSet<String>();
        for (String bigram : bigramFreq.keySet()) {
            if (bigramFreq.get(bigram) < this.bigramCountCutoff) {
                continue;
            }

            double score = scoreBigram(getTokensFromBigram(bigram));
            if (score < this.bigramScoreCutoff) {
                continue;
            }

            vocab.add(bigram);
        }

        // debug
        if (verbose) {
            System.out.println("--- # bigrams after being scored: " + vocab.size());
        }

        // merge bigrams
        if (verbose) {
            System.out.println("Merging unigrams to create bigram ...");
        }

        HashMap<String, Integer> finalTermFreq = new HashMap<String, Integer>();
        HashMap<String, Integer> finalDocFreq = new HashMap<String, Integer>();

        for (int d = 0; d < normTexts.length; d++) {
            for (int s = 0; s < normTexts[d].length; s++) {
                ArrayList<String> tokens = new ArrayList<String>();
                for (int i = 0; i < normTexts[d][s].length; i++) {
                    String curToken = normTexts[d][s][i];
                    if (curToken.isEmpty()) {
                        continue;
                    }

                    if (i + 1 < normTexts[d][s].length
                            && !normTexts[d][s][i + 1].isEmpty()) {
                        String bigram = getBigramString(normTexts[d][s][i], normTexts[d][s][i + 1]);
                        if (!vocab.contains(bigram)) {
                            continue;
                        }
                        tokens.add(bigram);
                        incrementMap(finalTermFreq, bigram);
                        i++;
                    } else {
                        if (termFreq.get(curToken) < this.unigramCountCutoff) {
                            continue;
                        }
                        tokens.add(curToken);
                        vocab.add(curToken);

                        incrementMap(finalTermFreq, curToken);
                    }
                }
                normTexts[d][s] = tokens.toArray(new String[tokens.size()]);

                Set<String> uniqueTerms = new HashSet<String>();
                uniqueTerms.addAll(Arrays.asList(normTexts[d][s]));
                for (String ut : uniqueTerms) {
                    incrementMap(finalDocFreq, ut);
                }
            }
        }

        // finalize
        ArrayList<RankingItem<String>> rankVocab = new ArrayList<RankingItem<String>>();
        for (String term : finalTermFreq.keySet()) {
            int tf = finalTermFreq.get(term);
            int df = finalDocFreq.get(term);

            if (tf < this.vocabTermFreqMinCutoff
                    || tf > this.vocabTermFreqMaxCutoff
                    || df < this.vocabDocFreqMinCutoff
                    || df > this.vocabDocFreqMaxCutoff) {
                continue;
            }
            double tfidf = finalTermFreq.get(term) * (Math.log(rawTexts.length) - Math.log(finalDocFreq.get(term)));
            rankVocab.add(new RankingItem<String>(term, tfidf));
        }
        Collections.sort(rankVocab);

        int vocabSize = Math.min(maxVocabSize, rankVocab.size());
        this.vocabulary = new ArrayList<String>();
        for (int i = 0; i < vocabSize; i++) {
            this.vocabulary.add(rankVocab.get(i).getObject());
        }
        Collections.sort(this.vocabulary);

        this.numericDocs = new int[rawTexts.length][];
        for (int d = 0; d < this.numericDocs.length; d++) { // for each document
            ArrayList<Integer> numericDoc = new ArrayList<Integer>();
            this.numericSentences[d] = new int[normTexts[d].length][];
            for (int s = 0; s < normTexts[d].length; s++) { // for each sentence
                ArrayList<Integer> numericSent = new ArrayList<Integer>();
                for (int w = 0; w < normTexts[d][s].length; w++) {
                    int numericTerm = Collections.binarySearch(this.vocabulary, normTexts[d][s][w]);
                    if (numericTerm < 0) // this term is out-of-vocab
                    {
                        continue;
                    }
                    numericDoc.add(numericTerm);
                    numericSent.add(numericTerm);
                }

                this.numericSentences[d][s] = new int[numericSent.size()];
                for (int i = 0; i < numericSent.size(); i++) {
                    this.numericSentences[d][s][i] = numericSent.get(i);
                }
            }
            this.numericDocs[d] = new int[numericDoc.size()];
            for (int i = 0; i < numericDoc.size(); i++) {
                this.numericDocs[d][i] = numericDoc.get(i);
            }
        }
    }

    private double scoreBigram(String[] bigramTokens) {
        String left = bigramTokens[0];
        String right = bigramTokens[1];
        if (excludeFromBigrams.contains(left) || excludeFromBigrams.contains(right)) {
            return 0.0;
        }
        long[][] counts = new long[2][2];
        counts[0][0] = this.bigramFreq.get(getBigramString(left, right));
        counts[1][0] = this.leftFreq.get(left) - counts[0][0];
        counts[0][1] = this.rightFreq.get(right) - counts[0][0];
        counts[1][1] = this.totalBigram - counts[0][0] - counts[0][1] - counts[1][0];
        double chisquareValue = this.chiSquareTest.chiSquare(counts);

        // debug
//        System.out.println(counts[0][0] + ". " + counts[0][1] + ". " + counts[1][0] + ". " + counts[1][1] 
//                + " ---> " + chisquareValue
//                + ". " + this.chiSquareTest.chiSquareTest(counts));

        return chisquareValue;
    }

    public void outputDetailedBigrams(String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (String bigram : bigramFreq.keySet()) {
            String[] bigramTokens = getTokensFromBigram(bigram);
            String left = bigramTokens[0];
            String right = bigramTokens[1];
            if (excludeFromBigrams.contains(left) || excludeFromBigrams.contains(right)) {
                continue;
            }
            long[][] counts = new long[2][2];
            counts[0][0] = this.bigramFreq.get(getBigramString(left, right));
            counts[1][0] = this.leftFreq.get(left) - counts[0][0];
            counts[0][1] = this.rightFreq.get(right) - counts[0][0];
            counts[1][1] = this.totalBigram - counts[0][0] - counts[0][1] - counts[1][0];
            double chisquareValue = this.chiSquareTest.chiSquare(counts);
            double pValue = this.chiSquareTest.chiSquareTest(counts);

            writer.write(bigram
                    + "\t" + counts[0][0]
                    + "\t" + counts[0][1]
                    + "\t" + counts[1][0]
                    + "\t" + counts[1][1]
                    + "\t" + chisquareValue
                    + "\t" + pValue);

            if (chisquareValue >= this.bigramScoreCutoff) {
                writer.write("\t1\n");
            } else {
                writer.write("\t0\n");
            }
        }
        writer.close();
    }

    public void outputDetailedVocab(String filepath) throws Exception {
        HashMap<Integer, Integer> numericTermFreq = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> numericDocFreq = new HashMap<Integer, Integer>();
        for (int d = 0; d < this.numericDocs.length; d++) {
            Set<Integer> docTerms = new HashSet<Integer>();
            for (int i = 0; i < this.numericDocs[d].length; i++) {
                int term = this.numericDocs[d][i];
                docTerms.add(term);

                Integer count = numericTermFreq.get(term);
                if (count == null) {
                    numericTermFreq.put(term, 1);
                } else {
                    numericTermFreq.put(term, count + 1);
                }
            }

            for (int term : docTerms) {
                Integer count = numericDocFreq.get(term);
                if (count == null) {
                    numericDocFreq.put(term, 1);
                } else {
                    numericDocFreq.put(term, count + 1);
                }
            }
        }


        // debug
        for (int i = 0; i < vocabulary.size(); i++) {
            if (numericTermFreq.get(i) == null) {
                System.out.println(i + "\t" + vocabulary.get(i));
            }
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < this.vocabulary.size(); i++) {
            int tf = numericTermFreq.get(i);
            int df = numericDocFreq.get(i);
            double tfidf = tf * (Math.log(numericDocs.length) - Math.log(df));
            writer.write(vocabulary.get(i) + "\t" + tf + "\t" + df + "\t" + tfidf + "\n");
        }
        writer.close();
    }

    private String[] getTokensFromBigram(String bigram) {
        return bigram.split("_");
    }

    private String getBigramString(String left, String right) {
        return left + "_" + right;
    }

    /**
     * Normalize a token. This includes - turn the token into lowercase - remove
     * punctuation - discard the reduced token if is stopword
     */
    public String normalize(String token) {
        StringBuilder normToken = new StringBuilder();
        token = token.toLowerCase();
        for (int i = 0; i < token.length(); i++) {
            Matcher m = p.matcher(token.substring(i, i + 1));
            if (!m.matches()) {
                normToken.append(token.substring(i, i + 1));
            }
        }
        String reduced = normToken.toString();
        if (lemmatization) {
            reduced = this.stemmer.stem(reduced);
        }

        if (reduced.length() < 3
                || token.matches("[^A-Za-z]+")
                || Character.isDigit(token.charAt(0))) {
            return "";
        }

        for (int i = 0; i < token.length(); i++) {
            if (!Character.isLetterOrDigit(token.charAt(i))) {
                return "";
            }
        }

        if (filterStopwords && stopwordRemoval.isStopword(reduced)) {
            return "";
        }
        return reduced;
    }

    private static void incrementMap(HashMap<String, Integer> map, String key) {
        Integer count = map.get(key);
        if (count == null) {
            map.put(key, 1);
        } else {
            map.put(key, count + 1);
        }
    }

    public static void main(String[] args) {
        long[][] array = new long[2][2];
        array[0][0] = 20;
        array[0][1] = 20;
        array[1][0] = 0;
        array[1][1] = 0;
        ChiSquareTest cst = new ChiSquareTest();
        double stat = cst.chiSquare(array);
        double p = cst.chiSquareTest(array);
        System.out.println(p);
        System.out.println(stat);
//        
//        try{
//            String folder = "L:/temp/";
//            File file = new File(folder);
//            String[] filenames = file.list();
//            String[] rawTexts = new String[filenames.length];
//            for(int i=0; i<filenames.length; i++){
//                StringBuilder str = new StringBuilder();
//                
//                BufferedReader reader = IOUtils.getBufferedReader(folder + filenames[i]);
//                String line = reader.readLine();
//                str.append(line);
//                while((line = reader.readLine()) != null)
//                    str.append(" ").append(line);
//                reader.close();
//                
//                rawTexts[i] = str.toString();
//            }
//            
//            CorpusProcessor cp = new CorpusProcessor(rawTexts);
//            cp.count();
//        }
//        catch(Exception e){
//            e.printStackTrace();
//            System.exit(1);
//        }

        SnowballStemmer stemmer = new SnowballStemmer();
        System.out.println(stemmer.stem("companies"));
        System.out.println(stemmer.stem("computers"));
    }
}