package sampler.supervised.regression.author;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.crossvalidation.Fold;
import data.AuthorResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import regression.AbstractRegressor;
import regression.Regressor;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import util.IOUtils;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public class AuthorLBFGS extends AbstractRegressor implements Regressor<AuthorResponseTextDataset> {

    private double[] weights;
    private double rho;
    private double mean;
    private double sigma;

    public AuthorLBFGS(String folder, double rho, double mean, double sigma) {
        super(folder);
        this.rho = rho;
        this.mean = mean;
        this.sigma = sigma;
    }

    public AuthorLBFGS(String folder) {
        super(folder);
        this.rho = 1.0;
        this.mean = 0.0;
        this.sigma = 1.0;
    }

    @Override
    public String getName() {
        if (name == null) {
            return "L-BFGS-r" + MiscUtils.formatDouble(rho)
                    + "_m-" + MiscUtils.formatDouble(mean)
                    + "_s-" + MiscUtils.formatDouble(sigma);
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
        
        System.out.println("Running L-BFGS ...");
        System.out.println("# observations: " + D);
        System.out.println("# variables: " + V);

        this.weights = new double[V];
        double[][] designMatrix = new double[A][V];
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

        GaussianIndLinearRegObjective optimizable = new GaussianIndLinearRegObjective(
                weights, designMatrix, trAuthorResponses,
                rho, mean, sigma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
        }
        System.out.println("Converge? " + converged);

        // update regression parameters
        for (int i = 0; i < weights.length; i++) {
            weights[i] = optimizable.getParameter(i);
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
    
    @Override
    public void output(File file) {
        if (verbose) {
            System.out.println("Outputing model to " + file);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(weights.length + "\n");
            for (int ii = 0; ii < weights.length; ii++) {
                writer.write(weights[ii] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    @Override
    public void input(File file) {
        if (verbose) {
            System.out.println("Inputing model from " + file);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            int V = Integer.parseInt(reader.readLine());
            this.weights = new double[V];
            for (int ii = 0; ii < V; ii++) {
                this.weights[ii] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + file);
        }
    }

    public double[] getWeights() {
        return this.weights;
    }
}
