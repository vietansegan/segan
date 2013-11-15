package svm;

import java.io.BufferedWriter;
import java.io.File;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class SVMUtils {

    public static void outputSVMLightFormat(
            File outputFile,
            double[][] features, double[] target) {
        if (features.length != target.length) {
            throw new RuntimeException("Number of instances mismatch. "
                    + features.length + " vs. " + target.length);
        }

        try {
            int N = features.length;
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int ii = 0; ii < N; ii++) {
                writer.write(Double.toString(target[ii]));
                for (int jj = 0; jj < features[ii].length; jj++) {
                    if (features[ii][jj] > 0) {
                        writer.write(" " + (jj+1) + ":" + features[ii][jj]);
                    }
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while writing to " + outputFile);
        }
    }
}
