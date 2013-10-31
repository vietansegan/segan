package core;

/**
 *
 * @author vietan
 */
public abstract class AbstractModel {

    public static final String ResultFile = "results.txt";
    protected final String folder;
    protected String configuration;

    public AbstractModel(String folder, String config) {
        this.folder = folder;
        this.configuration = config;
    }

    public abstract String getName();

    public String getModelFolder() {
        return this.getName() + "/";
    }

    public String getModelPath() {
        return this.folder + this.getModelFolder();
    }
}
