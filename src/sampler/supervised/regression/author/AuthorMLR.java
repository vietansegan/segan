package sampler.supervised.regression.author;

import core.crossvalidation.Fold;
import data.AuthorResponseTextDataset;
import java.io.File;
import optimization.GurobiMLRL1Norm;
import optimization.GurobiMLRL2Norm;
import regression.MLR;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class AuthorMLR extends MLR<AuthorResponseTextDataset> {

    public static enum AggregateType {

        MICRO_AVG, MACRO_AVG, SUM, WEIGHTED_MICRO_AVG
    }
    protected AggregateType aggType;
    protected double[] docWeights;

    public AuthorMLR(String folder, Regularizer reg, double t, AggregateType aggType) {
        super(folder, reg, t);
        this.aggType = aggType;
    }

    public void clearDocumentWeights() {
        this.docWeights = null;
    }

    public void setDocumentWeights(double[] docWs) {
        this.docWeights = docWs;
    }

    public double[] getDocumentWeights() {
        return this.docWeights;
    }

    @Override
    public String getName() {
        if (name == null) {
            return "author-MLR-" + regularizer + "-" + param + "-" + aggType;
        }
        return name;
    }

    @Override
    public void train(AuthorResponseTextDataset trainData) {
        if (verbose) {
            System.out.println("Training ...");
        }
        int[][] trWords = trainData.getWords();
        int[] trAuthors = trainData.getAuthors();
        double[] trAuthorResponses = trainData.getAuthorResponses();
        int V = trainData.getWordVocab().size();
        train(trWords, trAuthors, trAuthorResponses, V);
    }

    public void train(int[][] trWords, int[] trAuthors, double[] trAuthorResponses, int V) {
        int D = trWords.length;
        int A = trAuthorResponses.length;

        double[][] designMatrix = new double[A][V];
        if (this.aggType == AggregateType.SUM) {
            for (int d = 0; d < D; d++) {
                int author = trAuthors[d];
                for (int n = 0; n < trWords[d].length; n++) {
                    designMatrix[author][trWords[d][n]]++;
                }
            }
        } else if (this.aggType == AggregateType.MACRO_AVG) {
            int[] authorTokenCounts = new int[A];
            for (int d = 0; d < D; d++) {
                int author = trAuthors[d];
                authorTokenCounts[author] += trWords[d].length;
                for (int n = 0; n < trWords[d].length; n++) {
                    designMatrix[author][trWords[d][n]]++;
                }
            }
            for (int a = 0; a < A; a++) {
                for (int v = 0; v < V; v++) {
                    designMatrix[a][v] /= authorTokenCounts[a];
                }
            }
        } else if (this.aggType == AggregateType.MICRO_AVG) {
            int[] authorDocCounts = new int[A];

            for (int d = 0; d < D; d++) {
                if (trWords[d].length == 0) {
                    continue;
                }
                int author = trAuthors[d];
                authorDocCounts[author]++;
                double[] docWordVec = new double[V];
                for (int n = 0; n < trWords[d].length; n++) {
                    docWordVec[trWords[d][n]]++;
                }

                for (int v = 0; v < V; v++) {
                    designMatrix[author][v] += docWordVec[v] / trWords[d].length;
                }
            }

            for (int a = 0; a < A; a++) {
                for (int v = 0; v < V; v++) {
                    designMatrix[a][v] /= authorDocCounts[a];
                }
            }
        } else if (this.aggType == AggregateType.WEIGHTED_MICRO_AVG) {
            if (this.docWeights == null) {
                int[] authorDocCounts = new int[A];
                for (int d = 0; d < D; d++) {
                    if (trWords[d].length > 0) {
                        authorDocCounts[trAuthors[d]]++;
                    }
                }

                this.docWeights = new double[D];
                for (int d = 0; d < D; d++) {
                    int author = trAuthors[d];
                    this.docWeights[d] = 1.0 / authorDocCounts[author];
                }
            }

            if (this.docWeights.length != D) {
                throw new RuntimeException("Lengths mismatch. " + this.docWeights.length + " vs. " + D);
            }

            for (int d = 0; d < D; d++) {
                if (trWords[d].length == 0) {
                    continue;
                }
                int author = trAuthors[d];
                double[] docWordVec = new double[V];
                for (int n = 0; n < trWords[d].length; n++) {
                    docWordVec[trWords[d][n]]++;
                }

                for (int v = 0; v < V; v++) {
                    designMatrix[author][v] += docWeights[d] * docWordVec[v] / trWords[d].length;
                }
            }

        } else {
            throw new RuntimeException("Average type " + this.aggType + " is not supported");
        }

        if (regularizer == Regularizer.L1) {
            GurobiMLRL1Norm mlr = new GurobiMLRL1Norm(designMatrix, trAuthorResponses, param);
            this.weights = mlr.solve();
        } else if (regularizer == Regularizer.L2) {
            GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, trAuthorResponses);
            mlr.setSigma(param);
            this.weights = mlr.solve();
        } else {
            throw new RuntimeException(regularizer + " regularization is not supported");
        }
        IOUtils.createFolder(getRegressorFolder());
        output(new File(getRegressorFolder(), MODEL_FILE));
    }

    @Override
    public void test(AuthorResponseTextDataset testData) {
        if (verbose) {
            System.out.println("Testing ...");
        }
        String[] teDocIds = testData.getDocIds();
        int[][] teWords = testData.getWords();
        int[] teAuthors = testData.getAuthors();
        double[] teAuthorResponses = testData.getAuthorResponses();
        int V = testData.getWordVocab().size();

        double[] predictions = test(teWords, teAuthors, teAuthorResponses, V);
        File predFile = new File(getRegressorFolder(), PREDICTION_FILE + Fold.TestExt);
        outputPredictions(predFile, teDocIds, teAuthorResponses, predictions);

        File regFile = new File(getRegressorFolder(), RESULT_FILE + Fold.TestExt);
        outputRegressionResults(regFile, teAuthorResponses, predictions);
    }

    public double[] test(int[][] teWords, int[] teAuthors, double[] teAuthorResponses, int V) {
        input(new File(getRegressorFolder(), MODEL_FILE));

        int D = teWords.length;
        int A = teAuthorResponses.length;
        double[][] designMatrix = new double[A][V];
        if (this.aggType == AggregateType.SUM) {
            for (int d = 0; d < D; d++) {
                int author = teAuthors[d];
                for (int n = 0; n < teWords[d].length; n++) {
                    designMatrix[author][teWords[d][n]]++;
                }
            }
        } else if (this.aggType == AggregateType.MACRO_AVG) {
            int[] authorTokenCounts = new int[A];
            for (int d = 0; d < D; d++) {
                int author = teAuthors[d];
                authorTokenCounts[author] += teWords[d].length;
                for (int n = 0; n < teWords[d].length; n++) {
                    designMatrix[author][teWords[d][n]]++;
                }
            }
            for (int a = 0; a < A; a++) {
                for (int v = 0; v < V; v++) {
                    designMatrix[a][v] /= authorTokenCounts[a];
                }
            }
        } else if (this.aggType == AggregateType.MICRO_AVG) {
            int[] authorDocCounts = new int[A];

            for (int d = 0; d < D; d++) {
                if (teWords[d].length == 0) {
                    continue;
                }
                int author = teAuthors[d];
                authorDocCounts[author]++;
                double[] docWordVec = new double[V];
                for (int n = 0; n < teWords[d].length; n++) {
                    docWordVec[teWords[d][n]]++;
                }

                for (int v = 0; v < V; v++) {
                    designMatrix[author][v] += docWordVec[v] / teWords[d].length;
                }
            }

            for (int a = 0; a < A; a++) {
                for (int v = 0; v < V; v++) {
                    designMatrix[a][v] /= authorDocCounts[a];
                }
            }
        } else if (this.aggType == AggregateType.WEIGHTED_MICRO_AVG) {
            if (this.docWeights == null) {
                int[] authorDocCounts = new int[A];
                for (int d = 0; d < D; d++) {
                    if (teWords[d].length > 0) {
                        authorDocCounts[teAuthors[d]]++;
                    }
                }

                this.docWeights = new double[D];
                for (int d = 0; d < D; d++) {
                    int author = teAuthors[d];
                    this.docWeights[d] = 1.0 / authorDocCounts[author];
                }
            }

            if (this.docWeights.length != D) {
                throw new RuntimeException("Lengths mismatch. " + this.docWeights.length + " vs. " + D);
            }

            for (int d = 0; d < D; d++) {
                if (teWords[d].length == 0) {
                    continue;
                }
                int author = teAuthors[d];
                double[] docWordVec = new double[V];
                for (int n = 0; n < teWords[d].length; n++) {
                    docWordVec[teWords[d][n]]++;
                }

                for (int v = 0; v < V; v++) {
                    designMatrix[author][v] += docWeights[d] * docWordVec[v] / teWords[d].length;
                }
            }
        } else {
            throw new RuntimeException("Average type " + this.aggType + " is not supported");
        }

        double[] predictions = new double[A];
        for (int a = 0; a < A; a++) {
            double predVal = 0.0;
            for (int v = 0; v < V; v++) {
                predVal += designMatrix[a][v] * this.weights[v];
            }

            predictions[a] = predVal;
        }
        return predictions;
    }
}
