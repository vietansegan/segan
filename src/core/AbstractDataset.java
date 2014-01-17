package core;

/**
 *
 * @author vietan
 */
public abstract class AbstractDataset extends AbstractRunner {

    protected final String name;

    public AbstractDataset(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public String getFolder() {
        return this.name + "/";
    }
}
