/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package optimization;

import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBQuadExpr;
import gurobi.GRBVar;
import java.util.Random;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public class GurobiMultipleLinearRegression {

    private double[][] designMatrix;
    private double[] responseVector;
    private double lambda;
    private double[] lambdas;

    public GurobiMultipleLinearRegression(double[][] X, double[] y, double lambda) {
        this.designMatrix = X;
        this.responseVector = y;
        this.lambda = lambda;
    }

    public GurobiMultipleLinearRegression(double[][] X, double[] y, double[] lambdas) {
        this.designMatrix = X;
        this.responseVector = y;
        this.lambdas = lambdas;
    }

    public int getNumObservations() {
        return designMatrix.length;
    }

    public int getNumVariables() {
        return designMatrix[0].length;
    }

    public double getLambda(int v) {
        if (this.lambdas == null) {
            return this.lambda;
        } else {
            return this.lambdas[v];
        }
    }

    public double[] solve() {
        double[] solution = new double[getNumVariables()];
        try {
            GRBEnv env = new GRBEnv("env.log");
            GRBModel model = new GRBModel(env);

            // add variables
            GRBVar[] regParams = new GRBVar[getNumVariables()];
            for (int v = 0; v < getNumVariables(); v++) {
                regParams[v] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "var-" + v);
            }

            GRBVar[] docAuxParams = new GRBVar[getNumObservations()];
            for (int d = 0; d < getNumObservations(); d++) {
                docAuxParams[d] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "dvar-" + d);
            }

            model.update();

            // objective function
            GRBQuadExpr obj = new GRBQuadExpr();
            for (int d = 0; d < docAuxParams.length; d++) {
                obj.addTerm(1.0, docAuxParams[d], docAuxParams[d]);
            }
            for (int v = 0; v < getNumVariables(); v++) {
                obj.addTerm(getLambda(v), regParams[v], regParams[v]);
            }
            model.setObjective(obj, GRB.MINIMIZE);

            // constraints
            for (int d = 0; d < getNumObservations(); d++) {
                GRBLinExpr expr = new GRBLinExpr();
                expr.addTerm(1.0, docAuxParams[d]);
                for (int v = 0; v < getNumVariables(); v++) {
                    expr.addTerm(designMatrix[d][v], regParams[v]);
                }
                model.addConstr(expr, GRB.EQUAL, responseVector[d], "c-" + d);
            }

            // optimize
            model.optimize();

            // get solution
            for (int v = 0; v < getNumVariables(); v++) {
                solution[v] = regParams[v].get(GRB.DoubleAttr.X);
            }

            // dispose of model and environment
            model.dispose();
            env.dispose();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return solution;
    }
    
    public static void main(String[] args){
        test();
    }
    
    private static void test(){
        Random rand = new Random(1);
        
        int D = 10;
        int V = 10;
        double[][] designMatrix = new double[D][V];
        for(int d=0; d<D; d++){
            for(int v=0; v<V; v++)
                designMatrix[d][v] = rand.nextFloat();
        }
        
        double[] trueParams = new double[V];
        for(int i=0; i<3; i++){
            trueParams[i] = i+1;
            trueParams[V-1-i] = -i-1;
        }
        System.out.println("true params: " + MiscUtils.arrayToString(trueParams));
        
        // generate response
        double[] responseVector = new double[D];
        for(int d=0; d<D; d++){
            for(int v=0; v<V; v++){
                responseVector[d] += designMatrix[d][v] * trueParams[v];
            }
        }
        
        GurobiMultipleLinearRegression lasso = new GurobiMultipleLinearRegression(designMatrix, responseVector, 6.0);
        lasso.solve();
    }
}
