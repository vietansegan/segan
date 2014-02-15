package core;

/**
 *
 * @author vietan
 */
public abstract class AbstractExperiment<D extends AbstractDataset> extends AbstractRunner {

    public static final String SummaryFile = "summary.txt";
    public static final String SupervisedFolder = "supervised/";
    public static final String UnsupervisedFolder = "unsupervised/";
    public static final String SamplerFolder = "data/sampler/";
    
    public static final int UNOBSERVED = -1;

    public static enum RunType {

        SUPERFAST, FAST, FASTMOD, MODERATE, MODLONG, LONG, EXTENSIVE
    };
    public static int burn_in = 100;
    public static int max_iters = 1000;
    public static int sample_lag = 50;
    public static String experimentPath;
    protected D data;
    
    public abstract void preprocess() throws Exception;

    public abstract void setup() throws Exception;

    public abstract void run() throws Exception;

    public abstract void evaluate() throws Exception;

    protected void setRun(RunType type) {
        switch (type) {
            case SUPERFAST:
                burn_in = 1;
                max_iters = 2;
                sample_lag = 1;
                break;
            case FAST:
                burn_in = 2;
                max_iters = 10;
                sample_lag = 3;
                break;
            case FASTMOD:
                burn_in = 10;
                max_iters = 50;
                sample_lag = 10;
                break;
            case MODERATE:
                burn_in = 20;
                max_iters = 100;
                sample_lag = 20;
                break;
            case MODLONG:
                burn_in = 250;
                max_iters = 500;
                sample_lag = 25;
                break;
            case LONG:
                burn_in = 500;
                max_iters = 1000;
                sample_lag = 50;
                break;
            case EXTENSIVE:
                burn_in = 2000;
                max_iters = 5000;
                sample_lag = 50;
                break;
            default:
                burn_in = 200;
                max_iters = 1000;
                sample_lag = 20;
                break;
        }
    }

    public String getDatasetFolder() {
        return data.getFolder();
    }
    
    public static void addSamplingOptions() {
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");
        addOption("init", "Report interval");
    }
    
    public static void addCrossValidationOptions() {
        addOption("num-folds", "Number of folds. Default 5.");
        addOption("tr2dev-ratio", "Training-to-development ratio. Default 0.8.");
        addOption("cv-folder", "Folder to store cross validation folds");
        addOption("fold", "The cross-validation fold to run");
    }
    
    public static void addCorpusProcessorOptions() {
        addOption("u", "The minimum count of raw unigrams");
        addOption("b", "The minimum count of raw bigrams");
        addOption("bs", "The minimum score of bigrams");
        addOption("V", "Maximum vocab size");
        addOption("min-tf", "Term frequency minimum cutoff");
        addOption("max-tf", "Term frequency maximum cutoff");
        addOption("min-df", "Document frequency minimum cutoff");
        addOption("max-df", "Document frequency maximum cutoff");
        addOption("min-doc-length", "Document minimum length");
        options.addOption("s", false, "Whether stopwords are filtered");
        options.addOption("l", false, "Whether lemmatization is performed");
        options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
    }
    
    public static void addGreekParametersOptions() {
        addOption("alpha", "alpha");
        addOption("beta", "beta");
        addOption("gamma", "gamma");
        addOption("delta", "delta");
        addOption("lambda", "lambda");
        addOption("sigma", "sigma");
        addOption("rho", "rho");
    }
}
