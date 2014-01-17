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
}
