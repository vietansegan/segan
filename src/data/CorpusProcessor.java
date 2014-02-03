package data;

import java.io.BufferedReader;
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
import util.MiscUtils;
import util.RankingItem;
import util.Stemmer;
import util.StopwordRemoval;

/**
 * Process text data
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
    protected int minWordLength = 3; // minimum length of a word type 
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

    public CorpusProcessor(CorpusProcessor corp) {
        this(corp.unigramCountCutoff,
                corp.bigramCountCutoff,
                corp.bigramScoreCutoff,
                corp.maxVocabSize,
                corp.vocabTermFreqMinCutoff,
                corp.vocabTermFreqMaxCutoff,
                corp.vocabDocFreqMinCutoff,
                corp.vocabDocFreqMaxCutoff,
                corp.docTypeCountCutoff,
                corp.filterStopwords,
                corp.lemmatization);
    }

    public CorpusProcessor(
            int unigramCountCutoff,
            int bigramCountCutoff,
            double bigramScoreCutoff,
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            boolean filterStopwords,
            boolean lemmatization) {
        this(unigramCountCutoff,
                bigramCountCutoff,
                bigramScoreCutoff,
                maxVocabSize,
                vocTermFreqMinCutoff,
                vocTermFreqMaxCutoff,
                vocDocFreqMinCutoff,
                vocDocFreqMaxCutoff,
                docTypeCountCutoff,
                3,
                filterStopwords,
                lemmatization);
    }

    public CorpusProcessor(
            int unigramCountCutoff,
            int bigramCountCutoff,
            double bigramScoreCutoff,
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            int minWordLength,
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
        this.minWordLength = minWordLength;

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
            if (this.lemmatization) {
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

    public String[][] getRawSentences() {
        return this.rawSentences;
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

    public void setVocab(ArrayList<String> voc) {
        this.vocabulary = voc;
    }

    public void loadVocab(String filepath) {
        try {
            this.vocabulary = new ArrayList<String>();
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            this.vocabulary.add(reader.readLine());
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading vocab from " + filepath);
        }
    }

    public int[][] getNumerics() {
        return this.numericDocs;
    }

    public int[][][] getNumericSentences() {
        return this.numericSentences;
    }

    /**
     * Segment sentences
     */
    private String[][] segmentSentences(String[] rawDocuments) {
        if (verbose) {
            System.out.println("Segmenting sentences ...");
        }
        if (sentenceDetector == null) {
            throw new RuntimeException("Sentence detector is not initialized.");
        }
        int D = rawDocuments.length;
        String[][] rawSents = new String[D][];
        int step = MiscUtils.getRoundStepSize(D, 10);
        for (int d = 0; d < D; d++) {
            if (verbose && d % step == 0) {
                System.out.println("--- Segmenting sentences " + d + " / " + D);
            }

            rawSents[d] = sentenceDetector.sentDetect(rawDocuments[d]);
        }
        return rawSents;
    }

    /**
     * Tokenize sentences and filter tokens
     *
     * @param rawSentences
     */
    private String[][][] normalizeTokens(String[][] rawSentences) {
        if (verbose) {
            System.out.println("Normalizing tokens ...");
        }

        String[][][] normTexts = new String[rawTexts.length][][];
        int D = rawSentences.length;
        int step = MiscUtils.getRoundStepSize(D, 10);
        for (int d = 0; d < D; d++) {
            if (verbose && d % step == 0) {
                System.out.println("--- Normalizing tokens d = " + d + " / " + D);
            }
            normTexts[d] = new String[rawSentences[d].length][];
            for (int s = 0; s < rawSentences[d].length; s++) {
                String[] sentTokens = tokenizer.tokenize(rawSentences[d][s].toLowerCase());
                normTexts[d][s] = new String[sentTokens.length];
                for (int t = 0; t < sentTokens.length; t++) {
                    String normToken = normalize(sentTokens[t]);
                    normTexts[d][s][t] = normToken;
                }
            }
        }

        return normTexts;
    }

    /**
     * Process a set of documents with an existing vocabulary
     *
     * @param voc An existing vocabulary
     */
    public void process(ArrayList<String> voc) {
        if (rawTexts == null) {
            throw new RuntimeException("Raw texts have not been initialized yet");
        }

        // tokenize and normalize texts
        if (verbose) {
            System.out.println("Tokenizing and counting ...");
        }

        // segment sentences
        rawSentences = this.segmentSentences(this.rawTexts);

        // tokenize sentences and normalize tokens
        String[][][] normTexts = this.normalizeTokens(rawSentences);

        // keep only unigrams and bigrams in the given vocab
        if (verbose) {
            System.out.println("Building numeric representations ...");
        }
        int step = MiscUtils.getRoundStepSize(normTexts.length, 10);
        for (int d = 0; d < normTexts.length; d++) {
            if (verbose && d % step == 0) {
                System.out.println("--- Normalizing tokens d = " + d + " / " + normTexts.length);
            }
            for (int s = 0; s < normTexts[d].length; s++) {
                ArrayList<String> tokens = new ArrayList<String>();
                for (int i = 0; i < normTexts[d][s].length; i++) {
                    String curToken = normTexts[d][s][i];
                    if (curToken.isEmpty()) {
                        continue;
                    }

                    // consider a bigram
                    if (i + 1 < normTexts[d][s].length
                            && !normTexts[d][s][i + 1].isEmpty()) {
                        String bigram = getBigramString(normTexts[d][s][i], normTexts[d][s][i + 1]);

                        // if the bigram is not in the vocab, add the current
                        // unigram and move on
                        if (!voc.contains(bigram)) {
                            if (voc.contains(curToken)) {
                                tokens.add(curToken);
                            }
                            continue;
                        }

                        // if the bigram is in the vocab, add the bigram
                        tokens.add(bigram);
                        i++;
                    } else {
                        if (voc.contains(curToken)) {
                            tokens.add(curToken);
                        }
                    }
                }
                normTexts[d][s] = tokens.toArray(new String[tokens.size()]);
            }
        }

        this.vocabulary = voc;
        this.numericDocs = new int[rawTexts.length][];
        this.numericSentences = new int[rawTexts.length][][];
        for (int d = 0; d < this.numericDocs.length; d++) { // for each document
            ArrayList<Integer> numericDoc = new ArrayList<Integer>();
            this.numericSentences[d] = new int[normTexts[d].length][];
            for (int s = 0; s < normTexts[d].length; s++) { // for each sentence
                ArrayList<Integer> numericSent = new ArrayList<Integer>();
                for (int w = 0; w < normTexts[d][s].length; w++) {
                    int numericTerm = Collections.binarySearch(this.vocabulary, normTexts[d][s][w]);
                    if (numericTerm < 0) { // this term is out-of-vocab
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

    /**
     * Process a set of documents
     */
    public void process() {
        if (rawTexts == null) {
            throw new RuntimeException("Raw texts have not been initialized yet");
        }

        if (vocabulary != null) {
            if (verbose) {
                System.out.println("Using exisitng vocabulary ...");
            }
            this.process(vocabulary);
            return;
        }

        // tokenize and normalize texts
        if (verbose) {
            System.out.println("Tokenizing and counting ...");
        }
        String[][][] normTexts = new String[rawTexts.length][][];
        rawSentences = new String[rawTexts.length][];
        int stepsize = MiscUtils.getRoundStepSize(rawTexts.length, 10);
        for (int d = 0; d < rawTexts.length; d++) {
            if (verbose && d % stepsize == 0) {
                System.out.println("--- Tokenizing doc # " + d + " / " + rawTexts.length);
            }

            Set<String> uniqueDocTokens = new HashSet<String>();
            String rawText = rawTexts[d];
            rawSentences[d] = sentenceDetector.sentDetect(rawText);
            normTexts[d] = new String[rawSentences[d].length][];

            for (int s = 0; s < rawSentences[d].length; s++) {
                String[] sentTokens = tokenizer.tokenize(rawSentences[d][s].toLowerCase());
                normTexts[d][s] = new String[sentTokens.length];

                for (int t = 0; t < sentTokens.length; t++) {
                    String normToken = normalize(sentTokens[t]);
                    normTexts[d][s][t] = normToken;

                    if (!normToken.isEmpty()) {
                        MiscUtils.incrementMap(termFreq, normToken);
                        uniqueDocTokens.add(normToken);

                        if (t - 1 >= 0 && !normTexts[d][s][t - 1].isEmpty()) {
                            String preToken = normTexts[d][s][t - 1];
                            MiscUtils.incrementMap(leftFreq, preToken);
                            MiscUtils.incrementMap(rightFreq, normToken);
                            MiscUtils.incrementMap(bigramFreq, getBigramString(preToken, normToken));
                            totalBigram++;
                        }
                    }
                }
            }

            for (String uniToken : uniqueDocTokens) {
                MiscUtils.incrementMap(docFreq, uniToken);
            }
        }

        // debug
        if (verbose) {
            System.out.println("--- # raw unique unigrams: " + termFreq.size()
                    + ". " + docFreq.size());
            System.out.println("--- # raw unique bigrams: " + bigramFreq.size());
            System.out.println("--- # left: " + leftFreq.size()
                    + ". # right: " + rightFreq.size()
                    + ". total: " + totalBigram);
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
            Set<String> docUniqueTerms = new HashSet<String>();
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
                            // if the bigram is not in the vocab, add the current
                            // unigram and move on to the next unigram
                            if (termFreq.get(curToken) < this.unigramCountCutoff) {
                                continue;
                            }
                            tokens.add(curToken);
                            vocab.add(curToken);
                            MiscUtils.incrementMap(finalTermFreq, curToken);
                            continue;
                        }
                        tokens.add(bigram);
                        MiscUtils.incrementMap(finalTermFreq, bigram);
                        i++;
                    } else {
                        if (termFreq.get(curToken) < this.unigramCountCutoff) {
                            continue;
                        }
                        tokens.add(curToken);
                        vocab.add(curToken);
                        MiscUtils.incrementMap(finalTermFreq, curToken);
                    }
                }
                normTexts[d][s] = tokens.toArray(new String[tokens.size()]);

                // union
                docUniqueTerms.addAll(Arrays.asList(normTexts[d][s]));
            }

            // update document frequencies
            for (String ut : docUniqueTerms) {
                MiscUtils.incrementMap(finalDocFreq, ut);
            }
        }

        // finalize
        ArrayList<RankingItem<String>> rankVocab = new ArrayList<RankingItem<String>>();
        for (String term : finalTermFreq.keySet()) {
            int rawTf = finalTermFreq.get(term);
            int df = finalDocFreq.get(term);

            if (rawTf < this.vocabTermFreqMinCutoff
                    || rawTf > this.vocabTermFreqMaxCutoff
                    || df < this.vocabDocFreqMinCutoff
                    || df > this.vocabDocFreqMaxCutoff) {
                continue;
            }

            double tf = Math.log(rawTf + 1);
            double idf = Math.log(rawTexts.length) - Math.log(df);
            double tfidf = tf * idf;
            rankVocab.add(new RankingItem<String>(term, tfidf));
        }
        Collections.sort(rankVocab);

        if (verbose) {
            System.out.println("Raw vocab size: " + rankVocab.size());
        }

        int vocabSize = Math.min(maxVocabSize, rankVocab.size());
        this.vocabulary = new ArrayList<String>();
        for (int i = 0; i < vocabSize; i++) {
            this.vocabulary.add(rankVocab.get(i).getObject());
        }
        Collections.sort(this.vocabulary);

        this.numericDocs = new int[rawTexts.length][];
        this.numericSentences = new int[rawTexts.length][][];
        for (int d = 0; d < this.numericDocs.length; d++) { // for each document
            ArrayList<Integer> numericDoc = new ArrayList<Integer>();
            this.numericSentences[d] = new int[normTexts[d].length][];
            for (int s = 0; s < normTexts[d].length; s++) { // for each sentence
                ArrayList<Integer> numericSent = new ArrayList<Integer>();
                for (int w = 0; w < normTexts[d][s].length; w++) {
                    int numericTerm =
                            Collections.binarySearch(this.vocabulary, normTexts[d][s][w]);
                    if (numericTerm < 0) { // this term is out-of-vocab
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
        return chisquareValue;
    }

    public void outputDetailedBigrams(String filepath) throws Exception {
        if (verbose) {
            System.out.println("--- Outputing detailed bigrams to " + filepath);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        writer.write("bigram-score-cutoff:\t" + this.bigramCountCutoff + "\n");
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
                    + "\t" + MiscUtils.formatDouble(chisquareValue)
                    + "\t" + MiscUtils.formatDouble(pValue));

            if (chisquareValue >= this.bigramScoreCutoff) {
                writer.write("\t1\n");
            } else {
                writer.write("\t0\n");
            }
        }
        writer.close();
    }

    public void outputDetailedVocab(String filepath) throws Exception {
        if (verbose) {
            System.out.println("--- Outputing detailed vocab to " + filepath);
        }

        HashMap<Integer, Integer> numericTermFreq = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> numericDocFreq = new HashMap<Integer, Integer>();
        for (int d = 0; d < this.numericDocs.length; d++) {
            Set<Integer> docTerms = new HashSet<Integer>();
            for (int i = 0; i < this.numericDocs[d].length; i++) {
                int term = this.numericDocs[d][i];
                docTerms.add(term);
                MiscUtils.incrementMap(numericTermFreq, term);
            }

            for (int term : docTerms) {
                MiscUtils.incrementMap(numericDocFreq, term);
            }
        }

        // debug
        for (int i = 0; i < vocabulary.size(); i++) {
            if (numericTermFreq.get(i) == null) {
                System.out.println(i + "\t" + vocabulary.get(i)
                        + "\tnull");
            }
        }

        // sort terms according to TF-IDF
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int ii = 0; ii < this.vocabulary.size(); ii++) {
            int rawTf = numericTermFreq.get(ii);
            double tf = Math.log(rawTf + 1);
            int df = numericDocFreq.get(ii);
            double idf = (Math.log(numericDocs.length) - Math.log(df));
            double tfidf = tf * idf;
            rankItems.add(new RankingItem<Integer>(ii, tfidf));
        }
        Collections.sort(rankItems);

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        writer.write("Type\tRawTF\tTF\tDF\tIDF\tTFIDF\n");
        for (int i = 0; i < this.vocabulary.size(); i++) {
            RankingItem<Integer> item = rankItems.get(i);
            int rawTf = numericTermFreq.get(item.getObject());
            double tf = Math.log(rawTf + 1);
            int df = numericDocFreq.get(item.getObject());
            double idf = (Math.log(numericDocs.length) - Math.log(df));
            double tfidf = tf * idf;
            writer.write(vocabulary.get(item.getObject())
                    + "\t" + rawTf
                    + "\t" + MiscUtils.formatDouble(tf)
                    + "\t" + df
                    + "\t" + MiscUtils.formatDouble(idf)
                    + "\t" + MiscUtils.formatDouble(tfidf)
                    + " = " + MiscUtils.formatDouble(item.getPrimaryValue())
                    + "\n");
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
     * Normalize a token. This includes: (1) turn the token into lowercase, (2)
     * remove punctuation, (3) discard the reduced token if is stop-word.
     *
     * @param token The raw token
     * @return The normalize token
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

        if (reduced.length() < minWordLength
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
}