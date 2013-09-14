/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package core;

import java.io.BufferedWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Random;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public abstract class AbstractSampler {

    public static final String ReportFolder = "report/";
    public static final String AssignmentFileExt = ".assignment";
    public static final String ModelFileExt = ".model";
    public static final int INIT = -1;
    public static final boolean REMOVE = true;
    public static final boolean ADD = true;
    public static final boolean OBSERVED = true;
    public static final boolean EXTEND = true;
    public static final int UNOBSERVED = AbstractExperiment.UNOBSERVED;

    public static enum InitialState {

        RANDOM, SEEDED, FORWARD, PRESET, PRIOR
    }
    protected static final long RAND_SEED = 1123581321;
    protected static final double MAX_LOG = Math.log(Double.MAX_VALUE);
    protected static final NumberFormat formatter = new DecimalFormat("###.###");
    protected static Random rand = new Random(RAND_SEED);
    protected static long startTime;
    protected int BURN_IN = 5; // burn-in
    protected int MAX_ITER = 100; // maximum number of iterations
    protected int LAG = 1; // for outputing log-likelihood
    protected int REP_INTERVAL = 10; // report interval
    protected String folder;
    protected String name;
    protected ArrayList<Double> hyperparams;
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

    public void setSamplerConfiguration(int burn_in, int max_iter, int lag, int repInt) {
        BURN_IN = burn_in;
        MAX_ITER = max_iter;
        LAG = lag;
        REP_INTERVAL = repInt;
    }

    public void setReportInterval(int repInt) {
        REP_INTERVAL = repInt;
    }

    public abstract void initialize();

    public abstract void iterate();

    public abstract double getLogLikelihood();

    public abstract double getLogLikelihood(ArrayList<Double> testHyperparameters);

    public abstract void updateHyperparameters(ArrayList<Double> newParams);

    public abstract String getCurrentState();

    public abstract void validate(String msg);

    public abstract void outputState(String filepath);

    public abstract void inputState(String filepath);

    public void setWordVocab(ArrayList<String> vocab) {
        this.wordVocab = vocab;
    }

    public String[] getTopWords(double[] distribution, int numWords) {
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

        logln(getClass().toString());
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
                this.outputSampledHyperparameters(this.folder + this.getSamplerFolder() + "hyperparameters.txt");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
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
            this.logger = IOUtils.getBufferedWriter(this.folder
                    + this.getSamplerName() + "/" + "log.txt");
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void closeLogger() {
        try {
            this.logger.close();
            this.logger = null;
        } catch (Exception e) {
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
        } catch (Exception e) {
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

    public void outputSampledHyperparameters(String filepath) throws Exception {
        System.out.println("Outputing sampled hyperparameters to file " + filepath);

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < this.sampledParams.size(); i++) {
            writer.write(Integer.toString(i));
            for (double p : this.sampledParams.get(i)) {
                writer.write("\t" + p);
            }
            writer.write("\n");
        }
        writer.close();
    }

    protected ArrayList<Double> cloneHyperparameters() {
        ArrayList<Double> newParams = new ArrayList<Double>();
        for (int i = 0; i < this.hyperparams.size(); i++) {
            newParams.add(this.hyperparams.get(i));
        }
        return newParams;
    }

    /**
     * Slice sampling for hyper-parameter optimization
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
}