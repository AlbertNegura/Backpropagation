package NeuralNetwork;

import AlbertUtils.Matrix;

import java.util.*;
import com.github.sh0nk.matplotlib4j.*;

/**
 * This is an initial implementation for a simple Neural Network class.
 * The Matrix used is AlbertUtils.Matrix implementation.
 * The various layers are represented by Layer objects.
 * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
 * @date 28/10/2020
 * @version 1.0
 */
public class NeuralNetwork {
    Layer input, output, hidden;
    List<Layer> hidden_layers;
    public Matrix weights_ih , weights_ho , bias_h , bias_o;
    final boolean DEBUG = false;
    public Double l_rate=0.9;
    public Double decay = 0.0002;

    public NeuralNetwork(int i, int o) {
        /**
         * Constructor with a simple input and output layers and no hidden layers.
         * @param i input layer size
         * @param o output layer size
         */
        input = new Layer(o, i, o, 1);
        output = new Layer(o, o, o, 1);
    }

    public NeuralNetwork(int i,int h,int o) {
        /**
         * Constructor with an implied hidden layer between input and output layers.
         * @param i input layer size
         * @param h hidden layer size
         * @param o output layer size
         */
        input = new Layer(h, i, h, 1);
        output = new Layer(o, h, o, 1);
        //hidden_layers = new ArrayList<Layer>();
        //hidden_layers.add(new Layer(o,h,h,1));

        weights_ho = new Matrix(o,h);
        weights_ih = new Matrix(h,i);

        bias_h = new Matrix(h, 1);
        bias_o = new Matrix(o, 1);
    }

    public NeuralNetwork(int i,int o, int[][] hiddenLayers) {
        /**
         * Constructor with an implied hidden layer between input and output layers.
         * @param i input layer size
         * @param o output layer size
         * @param hiddenLayers hidden layer dimensions as a 2d array, where the first value hiddenLayer[x][0] represents the weight Matrix dimension, and the second value hiddenLayer[x][1] is the bias Matrix dimension FOR THE NEXT LAYER.
         */
        input = new Layer(hiddenLayers[0][0], i, hiddenLayers[0][0], 1);
        output = new Layer(o, hiddenLayers[hiddenLayers.length-1][1], o, 1);
        hidden_layers = new ArrayList<Layer>();

        for(int j = 1; j < hiddenLayers.length; j++){
            hidden_layers.add(new Layer(hiddenLayers[j-1][0], hiddenLayers[j][0], hiddenLayers[j][1],1));
        }
    }

    public List<Double> predict(Double[] X) throws Exception {
        /**
         * Returns a list of Double objects corresponding to the prediction given a particular input.
         * @param X A Double 1d array corresponding to the input to be predicted.
         * @return A List of Double objects corresponding to the output of the neural network.
         * @throws Exception in cases where the matrix dimensions don't match.
         */
        System.out.println("Weights first layer: " + weights_ih.toArray());
        System.out.println("Weights second layer: " + weights_ho.toArray());

        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(bias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho,hidden);
        output.add(bias_o);
        output.sigmoid();

        return output.toArray();
    }

    public void train(Double[] X, Double[] Y) throws Exception {
        if(DEBUG) {
            System.out.println("--------------\n");
            System.out.println("Weights first layer: " + weights_ih.toArray());
            System.out.println("Weights second layer: " + weights_ho.toArray());
        }
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(bias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho,hidden);
        output.add(bias_o);
        output.sigmoid();

        backpropagation(input, hidden, output, Y);

    }

    public ArrayList<Double> error_list = new ArrayList<Double>();
    private void backpropagation(Matrix input, Matrix hidden, Matrix output, Double[] Y) throws Exception {
        if(DEBUG) {
            System.out.println("--------------\n");
            System.out.println("Weights first layer: " + weights_ih.toArray());
            System.out.println("Weights second layer: " + weights_ho.toArray());
        }
        Matrix target = Matrix.fromArray(Y);

        Matrix error = Matrix.subtract(target, output);
        error.elementMultiply(error);
        error.multiply(0.5d);
        Matrix gradient = output.dsigmoid();
        gradient.elementMultiply(error);
        gradient.multiply(l_rate);

        error_list.add(error.sum());

        if(DEBUG)
            System.out.println("Gradient: " + Arrays.deepToString(gradient.data));

        Matrix weights_ho_delta =  Matrix.multiply(gradient, Matrix.transpose(hidden));
        if(DEBUG)
            System.out.println("Weights hidden/output delta: " + Arrays.deepToString(weights_ho_delta.data));

        Double weights_sum = Arrays.stream(weights_ho.data).mapToDouble(arr -> arr[0]).sum() + Arrays.stream(weights_ih.data).mapToDouble(arr -> arr[0]).sum();;
        if(DEBUG)
            System.out.println("Weights sum: " + weights_sum);

        weights_ho.add(weights_ho_delta);
        weights_ho.subtract((weights_sum*decay));
        bias_o.add(gradient);

        Matrix hidden_errors = Matrix.multiply(Matrix.transpose(weights_ho), error);
        if(DEBUG)
            System.out.println("Hidden layer errors: " + Arrays.deepToString(hidden_errors.data));

        Matrix hidden_gradient = hidden.dsigmoid();
        hidden_gradient.elementMultiply(hidden_errors);
        hidden_gradient.multiply(l_rate);
        if(DEBUG)
            System.out.println("Hidden layer gradient: " + Arrays.deepToString(hidden_gradient.data));

        Matrix weights_ih_delta = Matrix.multiply(hidden_gradient, Matrix.transpose(input));
        if(DEBUG)
            System.out.println("Weights hidden/output delta: " + Arrays.deepToString(weights_ih_delta.data));

        weights_ih.add(weights_ih_delta);
        weights_ih.subtract((weights_sum*decay));
        bias_h.add(hidden_gradient);

        if(DEBUG) {
            System.out.println("New weights first layer: " + weights_ih.toArray());
            System.out.println("New weights second layer: " + weights_ho.toArray());
            System.out.println("------------\n");
        }

    }

    public void fit(Double[][]X, Double[][]Y, int epochs) throws Exception {
        for(int i=0;i<epochs;i++)
        {
            int sampleN =  (int)(Math.random() * X.length);
            this.train(X[sampleN], Y[sampleN]);
        }
    }
}
