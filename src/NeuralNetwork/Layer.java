package NeuralNetwork;

import AlbertUtils.Matrix;

import java.util.HashMap;

/**
 * This is Layer class for Neural Network layers, which are essentially a combination of a weight Matrix and a bias Matrix.
 * The Matrix used is AlbertUtils.Matrix implementation.
 * @see AlbertUtils.Matrix
 * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
 * @date 28/10/2020
 * @version 1.1
 */
public class Layer {
    Matrix weights, bias;

    public Matrix inputs;
    public Matrix outputs;
    public Integer thisActivation;
    public HashMap<String,Integer> activationFunction = new HashMap<>(){
        {
            put("sigmoid", 0);
            put("relu", 1);
            put("sinusoid", 2);
            put("softmax", 2);
        }
    };

    public Layer(){
        /**
         * Creates a new single node layer.
         */
        this.weights = new Matrix();
        this.bias = new Matrix();
    }

    public Layer(int rows, int cols){
        /**
         * Creates a new layer with constant bias and a weights Matrix given by the specified values.
         * @param rows the number of rows of the weights Matrix.
         * @param cols the number of columns of the weights Matrix.
         */
        this.weights = new Matrix(rows, cols);
        this.bias = new Matrix();
    }

    public Layer(int weightRows, int weightCols, int biasRows, int biasCols){
        /**
         * Creates a new layer with the given weight Matrix and bias Matrix dimensions.
         * @param weightRows the number of rows of the weights Matrix.
         * @param weightCols the number of columns of the weights Matrix.
         * @param biasRows the number of rows of the bias Matrix.
         * @param biasCols the number of columns of the bias Matrix.
         */
        this.weights = new Matrix(weightRows,weightCols);
        this.bias = new Matrix(biasRows, biasCols);
    }


    public Layer(Matrix bias){
        /**
         * Creates a new Layer with the given bias Matrix and a weights matrix with a size corresponding to the bias Matrix.
         * @param bias the Matrix object for the bias.
         * @see AlbertUtils.Matrix
         */
        this.bias=bias;
        this.weights = new Matrix(bias.rows, bias.cols);
    }

    public Layer(Matrix weights, Matrix bias){
        /**
         * Creates a new layer with the given weights and bias Matrix objects.
         * @param weights the weights Matrix
         * @param bias the bias Matrix
         * @see AlbertUtils.Matrix
         */
        this.weights = weights;
        this.bias = bias;
    }

    public Layer(Matrix weights, Matrix bias, String activation){
        /**
         * Creates a new layer with the given weights and bias Matrix objects.
         * @param weights the weights Matrix
         * @param bias the bias Matrix
         * @see AlbertUtils.Matrix
         */
        this.weights = weights;
        this.bias = bias;
        this.thisActivation = activationFunction.get(activation);
    }

    public Matrix output(Matrix input) throws Exception {
        this.inputs = input;
        inputs.multiply(weights);
        inputs.add(bias);
        switch (thisActivation){
            case 0:
                inputs.sigmoid();
                break;
            case 1:
                inputs.relu(0d);
                break;
            case 2:
                inputs.sinusoid();
                break;
            case 3:
                inputs.softmax();
                break;
        }
        this.outputs = new Matrix(this.inputs);
        return this.outputs;
    }
}
