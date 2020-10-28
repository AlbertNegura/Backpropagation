package NeuralNetwork;

import AlbertUtils.Matrix;

/**
 * This is Layer class for Neural Network layers, which are essentially a combination of a weight Matrix and a bias Matrix.
 * The Matrix used is AlbertUtils.Matrix implementation.
 * @author Albert Negura (i6145864)
 * @date 28/10/2020
 * @version 1.0
 */
public class Layer {
    Matrix weights, bias;

    public Layer(){
        this.weights = new Matrix();
        this.bias = new Matrix();
    }

    public Layer(int rows, int cols){
        this.weights = new Matrix(rows, cols);
        this.bias = new Matrix();
    }

    public Layer(int weightRows, int weightCols, int biasRows, int biasCols){
        this.weights = new Matrix(weightRows,weightCols);
        this.bias = new Matrix(biasRows, biasCols);
    }


    public Layer(Matrix bias){
        this.bias=bias;
        this.weights = new Matrix(bias.rows, bias.cols);
    }

    public Layer(Matrix weights, Matrix bias){
        this.weights = weights;
        this.bias = bias;
    }
}
