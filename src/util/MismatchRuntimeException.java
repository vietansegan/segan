package util;

/**
 *
 * @author vietan
 */
public class MismatchRuntimeException extends RuntimeException {

    public MismatchRuntimeException(int observed, int expected) {
        super("Mismatch. " + observed + " vs. " + expected);
    }
}
