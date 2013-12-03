package data;

import core.AbstractObject;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class AuthorResponseTextDataset extends ResponseTextDataset {

    // header in author vocab file
    private ArrayList<String> authorProperties;
    // raw author list
    protected ArrayList<String> authorList;
    // processed authors
    protected ArrayList<String> authorVocab;
    protected int[] authors;
    private HashMap<String, Author> authorTable;
    // author specific data
    private int[][] authorWords;
    private int[][][] authorSentWords;
    private String[][] authorRawSents;
    private double[] authorResponses;

    public AuthorResponseTextDataset(String name) {
        super(name);
        this.authorTable = new HashMap<String, Author>();
    }

    public AuthorResponseTextDataset(String name, String folder) {
        super(name, folder);
        this.authorTable = new HashMap<String, Author>();
    }

    public AuthorResponseTextDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
        this.authorTable = new HashMap<String, Author>();
    }

    public void setAuthorPropertyNames(ArrayList<String> authorProps) {
        this.authorProperties = authorProps;
    }

    public ArrayList<String> getAuthorVocab() {
        return this.authorVocab;
    }

    public int[] getAuthors() {
        return this.authors;
    }

    public void setAuthorProperty(String authorId, String propName, String propVal) {
        Author author = authorTable.get(authorId);
        if (author == null) {
            author = new Author(authorId);
        }
        author.addProperty(propName, propVal);
        authorTable.put(authorId, author);
    }

    public String getAuthorProperty(String authorId, String propName) {
        Author author = authorTable.get(authorId);
        if (author == null) {
            return null;
        }
        return author.getProperty(propName);
    }

    public void setAuthorList(ArrayList<String> authorList) {
        this.authorList = authorList;
    }

    public String[] getAuthorIds() {
        return this.authorVocab.toArray(new String[authorVocab.size()]);
    }

    public String[][] getAuthorRawSentences() {
        return this.authorRawSents;
    }

    public int[][][] getAuthorSentWords() {
        return this.authorSentWords;
    }

    public int[][] getAuthorWords() {
        return this.authorWords;
    }

    public double[] getAuthorResponses() {
        return this.authorResponses;
    }

    public void setAuthorResponses(double[] ar) {
        this.authorResponses = ar;
    }

    public void formatAuthorData(String responseName) {
        this.formatAuthorRawSentences();
        this.formatAuthorSentWords();
        this.formatAuthorWords();
        this.formatAuthorResponses(responseName);
    }

    private void formatAuthorRawSentences() {
        HashMap<Integer, ArrayList<String>> authorRawSentMap =
                new HashMap<Integer, ArrayList<String>>();
        for (int d = 0; d < sentRawWords.length; d++) {
            int author = authors[d];
            ArrayList<String> authorRawSentList = authorRawSentMap.get(author);
            if (authorRawSentList == null) {
                authorRawSentList = new ArrayList<String>();
            }
            authorRawSentList.addAll(Arrays.asList(sentRawWords[d]));
            authorRawSentMap.put(author, authorRawSentList);
        }

        authorRawSents = new String[authorVocab.size()][];
        for (int a = 0; a < authorVocab.size(); a++) {
            ArrayList<String> authorSents = authorRawSentMap.get(a);
            if (authorRawSents == null) {
                authorRawSents[a] = new String[0];
                continue;
            }
            authorRawSents[a] = new String[authorSents.size()];
            for (int ii = 0; ii < authorSents.size(); ii++) {
                authorRawSents[a][ii] = authorSents.get(ii);
            }
        }
    }

    private void formatAuthorSentWords() {
        HashMap<Integer, ArrayList<int[]>> authorSentWordListMap =
                new HashMap<Integer, ArrayList<int[]>>();
        for (int d = 0; d < sentWords.length; d++) {
            int author = authors[d];
            ArrayList<int[]> authorSentWordList = authorSentWordListMap.get(author);
            if (authorSentWordList == null) {
                authorSentWordList = new ArrayList<int[]>();
            }
            authorSentWordList.addAll(Arrays.asList(sentWords[d]));
            authorSentWordListMap.put(author, authorSentWordList);
        }

        authorSentWords = new int[authorVocab.size()][][];
        for (int a = 0; a < authorVocab.size(); a++) {
            ArrayList<int[]> authorSentWordList = authorSentWordListMap.get(a);
            if (authorSentWordList == null) {
                authorSentWords[a] = new int[0][];
                continue;
            }
            int[][] sws = new int[authorSentWordList.size()][];
            for (int ii = 0; ii < sws.length; ii++) {
                sws[ii] = authorSentWordList.get(ii);
            }
            authorSentWords[a] = sws;
        }
    }

    public void formatAuthorWords() {
        HashMap<Integer, ArrayList<Integer>> authorWordListMap = new HashMap<Integer, ArrayList<Integer>>();
        for (int d = 0; d < words.length; d++) {
            int author = authors[d];
            ArrayList<Integer> authorWordList = authorWordListMap.get(author);
            if (authorWordList == null) {
                authorWordList = new ArrayList<Integer>();
            }

            for (int n = 0; n < words[d].length; n++) {
                authorWordList.add(words[d][n]);
            }
            authorWordListMap.put(author, authorWordList);
        }

        authorWords = new int[authorVocab.size()][];
        for (int a = 0; a < authorWords.length; a++) {
            ArrayList<Integer> authorWordList = authorWordListMap.get(a);
            if (authorWordList == null) {
                authorWords[a] = new int[0];
                continue;
            }
            int[] ws = new int[authorWordList.size()];
            for (int ii = 0; ii < ws.length; ii++) {
                ws[ii] = authorWordList.get(ii);
            }
            authorWords[a] = ws;
        }
    }

    public void formatAuthorResponses(String prop) {
        int index = this.authorProperties.indexOf(prop);
        if (index < 0) {
            throw new RuntimeException("Property " + prop + " is not found.");
        }
        authorResponses = new double[authorVocab.size()];
        for (int a = 0; a < authorResponses.length; a++) {
            Author author = authorTable.get(authorVocab.get(a));
            if(author.getProperty(prop) == null ||
                    author.getProperty(prop).isEmpty()) 
                throw new RuntimeException("Empty response: " + author.getId());
            authorResponses[a] = Double.parseDouble(author.getProperty(prop));
        }
    }

    @Override
    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);

        String[] rawTexts = textList.toArray(new String[textList.size()]);
        corpProc.setRawTexts(rawTexts);
        corpProc.process();

        outputWordVocab(outputFolder);
        outputTextData(outputFolder);

        formatAuthors(outputFolder);

        outputDocumentInfo(outputFolder);
        outputSentTextData(outputFolder);
    }

    private void formatAuthors(String outputFolder) throws Exception {
        if (verbose) {
            logln("--- Formatting authors. " + outputFolder);
        }
        // create author vocab
        this.authorVocab = new ArrayList<String>();
        for (int docIndex : this.processedDocIndices) {
            String author = authorList.get(docIndex);
            if (!this.authorVocab.contains(author)) {
                this.authorVocab.add(author);
            }
        }

        if (verbose) {
            logln("--- --- Author vocab size: " + this.authorVocab.size());
        }

        // output author vocab
        BufferedWriter writer = IOUtils.getBufferedWriter(new File(outputFolder,
                formatFilename + speakerVocabExt));

        // - headers
        writer.write("ID");
        for (String prop : authorProperties) {
            writer.write("\t" + prop);
        }
        writer.write("\n");

        // - authors
        for (int ii = 0; ii < this.authorVocab.size(); ii++) {
            String authorId = authorVocab.get(ii);
            Author author = authorTable.get(authorId);
            writer.write(authorId);
            for (String property : this.authorProperties) {
                writer.write("\t" + author.getProperty(property));
            }
            writer.append("\n");
        }
        writer.close();
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        // output document info
        File docInfoFile = new File(outputFolder, formatFilename + docInfoExt);
        if (verbose) {
            logln("--- Outputing document info ... " + docInfoFile);
        }
        BufferedWriter infoWriter = IOUtils.getBufferedWriter(docInfoFile);
        for (int docIndex : this.processedDocIndices) {
            String author = this.authorList.get(docIndex);
            int authorIdx = this.authorVocab.indexOf(author);
            if (authorIdx < 0) {
                throw new RuntimeException("Author " + author + " not found");
            }
            infoWriter.write(this.docIdList.get(docIndex)
                    + "\t" + authorIdx
                    + "\t" + this.responses[docIndex]
                    + "\n");
        }
        infoWriter.close();
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            inputAuthorVocab(new File(fFolder, formatFilename + speakerVocabExt));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void inputDocumentInfo(File filepath) throws Exception {
        if (verbose) {
            logln("--- Reading document info from " + filepath);
        }

        // load authors and responses from doc info file
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        ArrayList<String> dIdList = new ArrayList<String>();
        ArrayList<Integer> aList = new ArrayList<Integer>();
        ArrayList<Double> rList = new ArrayList<Double>();

        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            dIdList.add(sline[0]);
            aList.add(Integer.parseInt(sline[1]));
            rList.add(Double.parseDouble(sline[2]));
        }
        reader.close();

        this.docIds = dIdList.toArray(new String[dIdList.size()]);
        this.authors = new int[aList.size()];
        for (int i = 0; i < this.authors.length; i++) {
            this.authors[i] = aList.get(i);
        }
        this.responses = new double[rList.size()];
        for (int i = 0; i < this.responses.length; i++) {
            this.responses[i] = rList.get(i);
        }
    }

    protected void inputAuthorVocab(File authorVocFile) throws Exception {
        if (verbose) {
            logln("Loading authors from vocab file " + authorVocFile);
        }

        this.authorProperties = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(authorVocFile);
        String line = reader.readLine();
        String[] sline = line.split("\t");
        for (int ii = 1; ii < sline.length; ii++) {
            this.authorProperties.add(sline[ii]);
        }

        this.authorVocab = new ArrayList<String>();
        this.authorTable = new HashMap<String, Author>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            String id = sline[0];
            Author author = new Author(id);
            for (int ii = 1; ii < sline.length; ii++) {
                author.addProperty(authorProperties.get(ii - 1), sline[ii]);
            }
            this.authorVocab.add(id);
            this.authorTable.put(id, author);
        }
        reader.close();
    }

    class Author extends AbstractObject {

        Author(String id) {
            super(id);
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar:dist/lib/*' " + AuthorResponseTextDataset.class.getName() + " -help";
    }
}
