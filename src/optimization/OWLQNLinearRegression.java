package optimization;

import core.AbstractLinearModel;
import edu.stanford.nlp.optimization.DiffFunction;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class OWLQNLinearRegression extends AbstractLinearModel {

    private final double l1;
    private final double l2;
    private int maxIters;

    public OWLQNLinearRegression(String basename) {
        this(basename, 1.0, 1.0);
    }

    public OWLQNLinearRegression(String basename, double l1, double l2) {
        this(basename, l1, l2, 1000);
    }

    public OWLQNLinearRegression(String basename, double l1, double l2,
            int maxIters) {
        super(basename);
        this.l1 = l1;
        this.l2 = l2;
        this.maxIters = maxIters;
    }

    @Override
    public String getName() {
        return this.name + "_l1-" + l1 + "_l2-" + l2 + "_m-" + maxIters;
    }

    public double getL1() {
        return this.l1;
    }

    public double getL2() {
        return this.l2;
    }

    public void train(SparseVector[] designMatrix, double[] responses, int K) {
        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        OWLQN minimizer = new OWLQN();
        OWLQN.biasParameters.add(0);
        minimizer.setQuiet(quiet);
        minimizer.setMaxIters(maxIters);
        DiffFunc diffFunction = new DiffFunc(designMatrix, responses, l2);
        double[] initParams = new double[K];
        this.weights = minimizer.minimize(diffFunction, initParams, l1);
    }

    public double[] test(SparseVector[] designMatrix) {
        if (verbose) {
            System.out.println("Testing ...");
            System.out.println("--- # instances: " + designMatrix.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        double[] predictions = new double[designMatrix.length];
        for (int d = 0; d < predictions.length; d++) {
            predictions[d] = designMatrix[d].dotProduct(weights);
        }
        return predictions;
    }

    class DiffFunc implements DiffFunction {

        // inputs
        private final double[] values; // [N]-dim vector
        private final SparseVector[] designMatrix; // [N]x[K] sparse matrix
        // derived
        private final int N;
        private final int K;
        private final double l2;

        public DiffFunc(SparseVector[] designMatrix, double[] values, double l2) {
            this.designMatrix = designMatrix;
            this.values = values;
            this.l2 = l2;
            // derived statistics
            this.N = this.designMatrix.length;
            this.K = this.designMatrix[0].getDimension();
            if (this.K <= 0) {
                throw new RuntimeException("Number of features = " + this.K);
            }
        }

        @Override
        public int domainDimension() {
            return K;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int n = 0; n < N; n++) {
                double dotprod = dotprod(designMatrix[n], w);
                for (int k : this.designMatrix[n].getIndices()) {
                    grads[k] -= 2 * (values[n] - dotprod) * designMatrix[n].get(k);
                }
            }
            if (l2 > 0) {
                for (int k = 0; k < w.length; k++) {
                    grads[k] += 2 * l2 * w[k];
                }
            }
            return grads;
        }

        @Override
        public double valueAt(double[] w) {
            double loss = 0.0;
            for (int n = 0; n < N; n++) {
                double dotprod = dotprod(designMatrix[n], w);
                double diff = values[n] - dotprod;
                loss += diff * diff;
            }
            double val = loss;
            if (l2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += l2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        private double dotprod(SparseVector designVec, double[] w) {
            double dotprod = 0.0;
            for (int k : designVec.getIndices()) {
                dotprod += w[k] * designVec.get(k);
            }
            return dotprod;
        }
    }
}
