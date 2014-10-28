package core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import main.GlobalConstants;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public abstract class AbstractSampler implements Serializable {

    private static final long serialVersionUID = GlobalConstants.SerialVersionUID;
    public static final int MAX_NUM_PARALLEL_THREADS = 5;
    public static final String IterPredictionFolder = "iter-predictions";
    public static final String TopWordFile = AbstractExperiment.TopWordFile;
    public static final String TopicCoherenceFile = AbstractExperiment.TopicCoherenceFile;
//    public static final String IterPerplexityFolder = "iter-perplexities";
//    public static final String PerplexityFile = "perplexity.txt";
    public static final String IterPerplexityFolder = "iter-perps";
    public static final String PerplexityFile = "perp.txt";
    public static final String AveragingPerplexityFile = "avg-perplexity.txt";
    public static final String ModelFile = "model.zip";
    public static final String ReportFolder = "report/";
    public static final String AssignmentFileExt = ".assignment";
    public static final String ModelFileExt = ".model";
    public static final String LikelihoodFile = "likelihoods.txt";
    public static final String HyperparameterFile = "hyperparameters.txt";
    public static final int INIT = -1;
    public static final boolean REMOVE = true;
    public static final boolean ADD = true;
    public static final boolean OBSERVED = true;
    public static final boolean EXTEND = true;
    public static final int UNOBSERVED = AbstractExperiment.UNOBSERVED;
    public static final int PROPOSAL_INDEX = 0;
    public static final int ACTUAL_INDEX = 1;

    public static enum InitialState {

        RANDOM, SEEDED, FORWARD, PRESET, PRIOR
    }

    public static enum SamplingType {

        GIBBS, MH
    }
    protected static final long RAND_SEED = 1123581321;
    protected static final double MAX_LOG = Math.log(Double.MAX_VALUE);
    protected static final NumberFormat formatter = new DecimalFormat("###.###");
    protected static Random rand = new Random(RAND_SEED);
    protected static long startTime;
    // sampling configurations
    protected int BURN_IN = 5;          // burn-in
    protected int MAX_ITER = 100;       // maximum number of iterations
    protected int LAG = 1;              // for outputing log-likelihood
    protected int REP_INTERVAL = 10;    // report interval
    // test configuration
    protected int testBurnIn = 50;
    protected int testMaxIter = 100;
    protected int testSampleLag = 5;
    protected String folder;
    protected String name;
    protected String basename;
    protected ArrayList<Double> hyperparams; // should have used a HashMap instead of ArrayList
    protected boolean paramOptimized = false;
    protected String prefix = "";// to store description of predefined configurations (e.g., initialization)
    protected InitialState initState;
    protected double stepSize = 0.1;
    protected int numSliceSamples = 10;
    protected ArrayList<Double> logLikelihoods;
    protected ArrayList<ArrayList<Double>> sampledParams;
    protected ArrayList<String> wordVocab;
    protected int iter;
    protected boolean debug = false;
    protected boolean verbose = true;
    protected boolean log = true;
    protected boolean report = false;
    protected BufferedWriter logger;
    protected static CommandLineParser parser;
    protected static Options options;
    protected static CommandLine cmd;

    protected static void addOption(String optName, String optDesc) {
        options.addOption(OptionBuilder.withLongOpt(optName)
                .withDescription(optDesc)
                .hasArg()
                .withArgName(optName)
                .create());
    }

    public static void addSamplingOptions() {
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");
    }

    public static void addRunningOptions() {
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("help", false, "help");
        options.addOption("example", false, "example");
    }

    public String getBasename() {
        return this.basename;
    }

    public void setTestConfigurations(int tBurnIn, int tMaxIter, int tSampleLag) {
        if (tMaxIter < this.testMaxIter) {
            this.testBurnIn = tBurnIn;
            this.testMaxIter = tMaxIter;
            this.testSampleLag = tSampleLag;
        }
    }

    public void setSamplerConfiguration(int burn_in, int max_iter, int lag, int repInt) {
        BURN_IN = burn_in;
        MAX_ITER = max_iter;
        LAG = lag;
        REP_INTERVAL = repInt;
    }

    public boolean isReporting() {
        return verbose && iter % REP_INTERVAL == 0;
    }

    public int getBurnIn() {
        return this.BURN_IN;
    }

    public int getMaxIters() {
        return this.MAX_ITER;
    }

    public int getSampleLag() {
        return this.LAG;
    }

    public int getReportInterval() {
        return this.REP_INTERVAL;
    }

    public void setReportInterval(int repInt) {
        REP_INTERVAL = repInt;
    }

    protected String getIteratedStateFile() {
        return "iter-" + iter + ".zip";
    }

    protected String getIteratedTopicFile() {
        return "topwords-" + iter + ".txt";
    }

    public abstract void initialize();

    public abstract void iterate();

    public abstract double getLogLikelihood();

    public abstract double getLogLikelihood(ArrayList<Double> testHyperparameters);

    public abstract void updateHyperparameters(ArrayList<Double> newParams);

    public abstract void validate(String msg);

    public abstract void outputState(String filepath);

    public abstract void inputState(String filepath);

    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        return str.toString();
    }

    public File getFinalStateFile() {
        return new File(getReportFolderPath(), "iter-" + MAX_ITER + ".zip");
    }

    public void inputFinalState() {
        this.inputState(new File(getReportFolderPath(), "iter-" + MAX_ITER + ".zip"));
    }

    public void outputState(File file) {
        this.outputState(file.getAbsolutePath());
    }

    public void inputState(File file) {
        this.inputState(file.getAbsolutePath());
    }

    protected void outputZipFile(
            String filepath,
            String modelStr,
            String assignStr) throws Exception {
        String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
        ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

        if (modelStr != null) {
            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();
        }

        if (assignStr != null) {
            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            byte[] data = assignStr.getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();
        }

        writer.close();
    }

    public ArrayList<String> getWordVocab() {
        return this.wordVocab;
    }

    public void setWordVocab(ArrayList<String> vocab) {
        this.wordVocab = vocab;
    }

    public String[] getTopWords(double[] distribution, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("Word vocab empty");
        }
        ArrayList<RankingItem<String>> topicSortedVocab = IOUtils.getSortedVocab(distribution, this.wordVocab);
        String[] topWords = new String[numWords];
        for (int i = 0; i < numWords; i++) {
            topWords[i] = topicSortedVocab.get(i).getObject();
        }
        return topWords;
    }

    public String getSamplerName() {
        return this.name;
    }

    public String getSamplerFolder() {
        return this.getSamplerName() + "/";
    }

    public String getSamplerFolderPath() {
        return new File(folder, name).getAbsolutePath();
    }

    public String getReportFolderPath() {
        return new File(getSamplerFolderPath(), ReportFolder).getAbsolutePath();
    }

    public String getFormatNumberString(double value) {
        if (value > 0.001) {
            return formatter.format(value);
        } else {
            return Double.toString(value);
        }
    }

    public void sample() {
        if (log) {
            openLogger();
        }

        logln(getClass().toString() + "\t" + getSamplerName());
        startTime = System.currentTimeMillis();

        initialize();

        iterate();

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                IOUtils.createFolder(getSamplerFolderPath());
                this.outputSampledHyperparameters(
                        new File(getSamplerFolderPath(), HyperparameterFile));
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling");
        }
    }

    public void concatPrefix(String p) {
        if (this.prefix == null) {
            this.prefix = p;
        } else {
            this.prefix += "_" + p;
        }
    }

    public void setPrefix(String p) {
        this.prefix = p;
    }

    public String getPrefix() {
        if (this.prefix == null) {
            return "";
        } else {
            return this.prefix;
        }
    }

    public void openLogger() {
        try {
            this.logger = IOUtils.getBufferedWriter(new File(getSamplerFolderPath(), "log.txt"));
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void closeLogger() {
        try {
            this.logger.close();
            this.logger = null;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setReport(boolean report) {
        this.report = report;
    }

    public void setLog(boolean l) {
        this.log = l;
    }

    public void setDebug(boolean d) {
        this.debug = d;
    }

    public void setVerbose(boolean v) {
        this.verbose = v;
    }

    protected void logln(String msg) {
        System.out.println("[LOG] " + msg);
        try {
            if (logger != null) {
                this.logger.write(msg + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public boolean isLogging() {
        return this.logger != null;
    }

    public boolean areParamsOptimized() {
        return this.paramOptimized;
    }

    public void setParamsOptimized(boolean po) {
        this.paramOptimized = po;
    }

    public void outputLogLikelihoods(File file) throws Exception {
        IOUtils.outputLogLikelihoods(logLikelihoods, file.getAbsolutePath());
    }

    public void outputSampledHyperparameters(File file) throws Exception {
        this.outputSampledHyperparameters(file.getAbsolutePath());
    }

    public void outputSampledHyperparameters(String filepath) {
        System.out.println("Outputing sampled hyperparameters to file " + filepath);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            for (int i = 0; i < this.sampledParams.size(); i++) {
                writer.write(Integer.toString(i));
                for (double p : this.sampledParams.get(i)) {
                    writer.write("\t" + p);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing sampled hyperparameters");
        }
    }

    protected ArrayList<Double> cloneHyperparameters() {
        ArrayList<Double> newParams = new ArrayList<Double>();
        for (int i = 0; i < this.hyperparams.size(); i++) {
            newParams.add(this.hyperparams.get(i));
        }
        return newParams;
    }

    protected void updateHyperparameters() {
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

    /**
     * Slice sampling for hyper-parameter optimization.
     */
    protected void sliceSample() {
        int dim = hyperparams.size();
        double[] lefts = new double[dim];
        double[] rights = new double[dim];
        ArrayList<Double> tempParams = hyperparams;

        if (debug) {
            logln("ori params: " + MiscUtils.listToString(hyperparams));
        }

        for (int s = 0; s < numSliceSamples; s++) {
            if (debug) {
                logln("");
                logln("Slice sampling. Iter = " + s);
            }

            double cur_llh = getLogLikelihood(tempParams);
            double log_u_prime = Math.log(rand.nextDouble()) + cur_llh;
            for (int i = 0; i < dim; i++) {
                double r = rand.nextDouble();
                lefts[i] = tempParams.get(i) - r * stepSize;
                rights[i] = lefts[i] + stepSize;
                if (lefts[i] < 0) {
                    lefts[i] = 0;
                }
            }

            if (debug) {
                logln("cur_llh = " + cur_llh + ", log_u' = " + log_u_prime);
                logln("lefts: " + MiscUtils.arrayToString(lefts));
                logln("rights: " + MiscUtils.arrayToString(rights));
            }

            ArrayList<Double> newParams = null;
            while (true) {
                newParams = new ArrayList<Double>();
                for (int i = 0; i < dim; i++) {
                    newParams.add(rand.nextDouble() * (rights[i] - lefts[i]) + lefts[i]);
                }
                double new_llh = getLogLikelihood(newParams);

                if (debug) {
                    logln("new params: " + MiscUtils.listToString(newParams) + "; new llh = " + new_llh);
                }

                if (new_llh > log_u_prime) {
                    break;
                } else {
                    for (int i = 0; i < dim; i++) {
                        if (newParams.get(i) < tempParams.get(i)) {
                            lefts[i] = newParams.get(i);
                        } else {
                            rights[i] = newParams.get(i);
                        }
                    }
                }
            }

            tempParams = newParams;
        }

        updateHyperparameters(tempParams);

        if (debug) {
            logln("sampled params: " + MiscUtils.listToString(hyperparams)
                    + "; final llh = " + getLogLikelihood(hyperparams));
        }
    }

    /**
     * Run multiple threads in parallel.
     *
     * @param threads
     * @throws java.lang.Exception
     */
    public static void runThreads(ArrayList<Thread> threads) throws Exception {
        int c = 0;
        for (int ii = 0; ii < threads.size() / MAX_NUM_PARALLEL_THREADS; ii++) {
            for (int jj = 0; jj < MAX_NUM_PARALLEL_THREADS; jj++) {
                threads.get(ii * MAX_NUM_PARALLEL_THREADS + jj).start();
            }
            for (int jj = 0; jj < MAX_NUM_PARALLEL_THREADS; jj++) {
                threads.get(ii * MAX_NUM_PARALLEL_THREADS + jj).join();
                c++;
            }
        }
        for (int jj = c; jj < threads.size(); jj++) {
            threads.get(jj).start();
        }
        for (int jj = c; jj < threads.size(); jj++) {
            threads.get(jj).join();
        }
    }
}
