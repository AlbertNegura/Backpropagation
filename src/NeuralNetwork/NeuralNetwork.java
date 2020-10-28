package NeuralNetwork;

import AlbertUtils.Matrix;

import java.util.*;

public class NeuralNetwork {
    Layer input, output;
    List<Layer> hidden_layers;
    Matrix weights_ih , weights_ho , bias_h , bias_o;
    Double l_rate=0.01;

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
        //input = new Layer(h, i, h, 1);
        //output = new Layer(o, h, o, 1);

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
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(bias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho,hidden);
        output.add(bias_o);
        output.sigmoid();

        Matrix target = Matrix.fromArray(Y);

        Matrix error = Matrix.subtract(target, output);
        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(l_rate);

        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta =  Matrix.multiply(gradient, hidden_T);

        weights_ho.add(who_delta);
        bias_o.add(gradient);

        Matrix who_T = Matrix.transpose(weights_ho);
        Matrix hidden_errors = Matrix.multiply(who_T, error);

        Matrix h_gradient = hidden.dsigmoid();
        h_gradient.multiply(hidden_errors);
        h_gradient.multiply(l_rate);

        Matrix i_T = Matrix.transpose(input);
        Matrix wih_delta = Matrix.multiply(h_gradient, i_T);

        weights_ih.add(wih_delta);
        bias_h.add(h_gradient);

    }

    public void fit(Double[][]X, Double[][]Y, int epochs) throws Exception {
        for(int i=0;i<epochs;i++)
        {
            int sampleN =  (int)(Math.random() * X.length );
            this.train(X[sampleN], Y[sampleN]);
        }
    }
}
