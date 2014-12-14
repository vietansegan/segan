package optimization;

import core.AbstractLinearModel;
import edu.stanford.nlp.optimization.DiffFunction;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class OWLQNLogisticRegression extends AbstractLinearModel {

    private final double l1;
    private final double l2;
    private int maxIters;

    public OWLQNLogisticRegression(String basename) {
        this(basename, 1.0, 1.0);
    }

    public OWLQNLogisticRegression(String basename, double l1, double l2) {
        this(basename, l1, l2, 1000);
    }

    public OWLQNLogisticRegression(String basename, double l1, double l2,
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

    public void train(SparseVector[] designMatrix, int[] responses, int K) {
        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        OWLQN minimizer = new OWLQN();
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
            double expdotprod = Math.exp(designMatrix[d].dotProduct(weights));
            predictions[d] = expdotprod / (1.0 + expdotprod);
        }
        return predictions;
    }

    class DiffFunc implements DiffFunction {

        // inputs
        private final int[] values; // [N]-dim binary vector {0, 1}
        private final SparseVector[] designMatrix; // [N]x[K] sparse matrix
        // derived
        private final int N;
        private final int K;
        private final double l2;

        public DiffFunc(SparseVector[] designMatrix, int[] values, double l2) {
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
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int nn = 0; nn < N; nn++) {
                double dotProb = designMatrix[nn].dotProduct(w);
                llh -= values[nn] * dotProb - Math.log(Math.exp(dotProb) + 1);
            }

            double val = llh;
            if (l2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += l2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int nn = 0; nn < N; nn++) {
                double dotprod = designMatrix[nn].dotProduct(w);
                double expDotprod = Math.exp(dotprod);
                double pred = expDotprod / (expDotprod + 1);
                for (int kk = 0; kk < K; kk++) {
                    grads[kk] -= (values[nn] - pred) * designMatrix[nn].get(kk);
                }
            }
            if (l2 > 0) {
                for (int kk = 0; kk < w.length; kk++) {
                    grads[kk] += 2 * l2 * w[kk];
                }
            }
            return grads;
        }
    }
}
