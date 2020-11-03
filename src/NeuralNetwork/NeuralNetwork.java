package NeuralNetwork;

import AlbertUtils.Matrix;

import java.util.*;

/**
 * This is an initial implementation for a simple Neural Network class.
 * The Matrix used is AlbertUtils.Matrix implementation.
 * The various layers are represented by Layer objects.
 * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
 * @date 28/10/2020
 * @version 2.0
 */
public class NeuralNetwork {
    Layer input, output, hidden;
    List<Layer> hidden_layers;
    public Matrix weights_ih , weights_ho , bias_h , bias_o;
    public boolean DEBUG = false;
    public Double l_rate=5d;
    public Double decay = 0.00002;

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
        if(DEBUG) {
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

        return output.toArray();
    }

    public void train(Double[] X, Double[] Y) throws Exception {
        /**
         * Train the neural network given a particular input / output.
         * @param X The input of the current iteration.
         * @param Y The matching output.
         */
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

        backpropagation(input, hidden, output, Y, true);

    }

    Matrix delta_weights_ho, delta_weights_ih;
    Matrix delta_bias_o, delta_bias_h;
    public void train(ArrayList<Double[]> X, ArrayList<Double[]> Y) throws Exception {
        /**
         * Train the neural network given a particular batch of input / output combinations.
         * @param X The ordered list of inputs of the current iteration.
         * @param Y The matching ordered list of outputs.
         */
        if(DEBUG) {
            System.out.println("--------------\n");
            System.out.println("Weights first layer: " + weights_ih.toArray());
            System.out.println("Weights second layer: " + weights_ho.toArray());
        }
        delta_weights_ho = new Matrix(weights_ho.rows,weights_ho.cols);
        delta_weights_ih = new Matrix(weights_ih.rows,weights_ih.cols);
        Matrix average_error = new Matrix(Matrix.fromArray(Y.get(0)).rows, Matrix.fromArray(Y.get(0)).cols);
        for(int i = 0; i < X.size(); i++) {
            Double[] x = X.get(i);
            Matrix input = Matrix.fromArray(x);
            Matrix hidden = Matrix.multiply(weights_ih, input);
            hidden.add(bias_h);
            hidden.sigmoid();

            Matrix output = Matrix.multiply(weights_ho, hidden);
            output.add(bias_o);
            output.sigmoid();
            Matrix error = Matrix.subtract(Matrix.fromArray(Y.get(i)), output);
            average_error.add(error);
            backpropagation(input, hidden, output, Y.get(i));
        }

        Double value = average_error.sum();
        value /= Y.size();
        value *= value;
        if(error_list.size() < 100000)  //more causes plotting issues
            error_list.add(value);

        delta_weights_ho.multiply(1d/(double)X.size());
        delta_weights_ih.multiply(1d/(double)X.size());
        Matrix temp_ho = new Matrix(weights_ho);
        Matrix temp_ih = new Matrix(weights_ih);
        temp_ho.multiply(decay);
        temp_ih.multiply(decay);
        temp_ho.add(delta_weights_ho);
        temp_ih.add(delta_weights_ih);
        temp_ho.multiply(-l_rate);
        temp_ih.multiply(-l_rate);
        weights_ho.add(temp_ho);
        weights_ih.add(temp_ih);

        delta_bias_o = new Matrix(bias_o);
        delta_bias_h = new Matrix(bias_h);
        delta_bias_o.multiply(1d/(double)X.size());
        delta_bias_h.multiply(1d/(double)X.size());
        delta_bias_h.multiply(-l_rate);
        delta_bias_o.multiply(-l_rate);
        bias_h.add(delta_bias_h);
        bias_o.add(delta_bias_o);
    }

    public ArrayList<Double> error_list = new ArrayList<>(); // use with matplotlib4j to generate error plot
    public ArrayList<Double> temp_error_list = new ArrayList<>(); // use with matplotlib4j to generate error plot
    private void backpropagation(Matrix input, Matrix hidden, Matrix output, Double[] Y) throws Exception {
        /**
         * Batch-based backpropagation.
         * @param input The input.
         * @param hidden The activated output to the hidden layer.
         * @param output The activated output of the hidden layer to the output layer.
         * @param Y The target output for the given input.
         */
        if(DEBUG) {
            System.out.println("--------------\n");
            System.out.println("Weights first layer: " + weights_ih.toArray());
            System.out.println("Weights second layer: " + weights_ho.toArray());
        }
        Matrix target = Matrix.fromArray(Y);
        Matrix error = Matrix.subtract(target,output);
        Matrix gradient = output.dsigmoid();
        gradient.elementMultiply(error);
        gradient.multiply(-1d);

        Matrix hidden_errors = Matrix.multiply(Matrix.transpose(weights_ho), gradient);
        Matrix hidden_gradient = hidden.dsigmoid();
        hidden_gradient.elementMultiply(hidden_errors);


        Matrix weights_ho_delta =  Matrix.multiply(gradient, Matrix.transpose(hidden));
        bias_o.add(gradient);

        Matrix weights_ih_delta = Matrix.multiply(hidden_gradient, Matrix.transpose(input));
        bias_h.add(hidden_gradient);


        delta_weights_ho.add(weights_ho_delta);
        delta_weights_ih.add(weights_ih_delta);

        if(DEBUG){
            System.out.println("Gradient: " + Arrays.deepToString(gradient.data));
            System.out.println("Weights hidden/output delta: " + Arrays.deepToString(weights_ho_delta.data));
            //System.out.println("Weights sum: " + weights_sum);
            System.out.println("Hidden layer errors: " + Arrays.deepToString(hidden_errors.data));
            System.out.println("Hidden layer gradient: " + Arrays.deepToString(hidden_gradient.data));
            System.out.println("Weights hidden/output delta: " + Arrays.deepToString(weights_ih_delta.data));
            System.out.println("New weights first layer: " + weights_ih.toArray());
            System.out.println("New weights second layer: " + weights_ho.toArray());
            System.out.println("------------\n");
        }

    }

    public int j=0;
    public boolean draw = false;
    public boolean flag = false;
    private void backpropagation(Matrix input, Matrix hidden, Matrix output, Double[] Y, boolean old) throws Exception {
        /**
         * Simple backpropagation.
         * @param input The input.
         * @param hidden The activated output to the hidden layer.
         * @param output The activated output of the hidden layer to the output layer.
         * @param Y The target output for the given input.
         */
        if (DEBUG) {
            System.out.println("--------------\n");
            System.out.println("Weights first layer: " + weights_ih.toArray());
            System.out.println("Weights second layer: " + weights_ho.toArray());
        }
        Matrix target = Matrix.fromArray(Y);

        Matrix error = Matrix.subtract(target, output);
        Matrix gradient = output.dsigmoid();
        gradient.elementMultiply(error);

        Matrix outer_error = Matrix.elementMultiply(error, error);
        if(outer_error.sum() < 0.001) {
            prev_epoch = outer_error.sum();
            flag = true;
        }

        if (draw && j % 100 == 0 && error_list.size() < 100000){ //this is by far the slowest part of the code - execution time increased 30th fold after adding this
            temp_error_list.add(outer_error.sum());
            Double sum = Arrays.stream(temp_error_list.toArray()).mapToDouble(arr -> (double) arr).sum();
            error_list.add(sum/temp_error_list.size());
        }
        else if(draw) {
            j++;
            temp_error_list.add(outer_error.sum());
        }

        Matrix weights_ho_delta =  Matrix.multiply(gradient, Matrix.transpose(hidden));
        weights_ho_delta.multiply(l_rate);

        Matrix hidden_errors = Matrix.multiply(Matrix.transpose(weights_ho), gradient);

        Matrix hidden_gradient = hidden.dsigmoid();
        hidden_gradient.elementMultiply(hidden_errors);

        Matrix weights_ih_delta = Matrix.multiply(hidden_gradient, Matrix.transpose(input));
        weights_ih_delta.multiply(l_rate);

        Matrix temp = new Matrix(weights_ho);
        weights_ho.add(weights_ho_delta);
        temp.multiply(decay);
        weights_ho.subtract(temp);
        bias_o.add(gradient);

        temp = new Matrix(weights_ih);
        weights_ih.add(weights_ih_delta);
        temp.multiply(decay);
        weights_ih.subtract(temp);
        bias_h.add(hidden_gradient);


        if(DEBUG){
            System.out.println("Gradient: " + Arrays.deepToString(gradient.data));
            System.out.println("Weights hidden/output delta: " + Arrays.deepToString(weights_ho_delta.data));
            System.out.println("Hidden layer errors: " + Arrays.deepToString(hidden_errors.data));
            System.out.println("Hidden layer gradient: " + Arrays.deepToString(hidden_gradient.data));
            System.out.println("Weights hidden/output delta: " + Arrays.deepToString(weights_ih_delta.data));
            System.out.println("New weights first layer: " + weights_ih.toArray());
            System.out.println("New weights second layer: " + weights_ho.toArray());
            System.out.println("------------\n");
        }

    }

    public boolean shuffle = false;
    public Double prev_epoch = 0d;
    public void fit(Double[][]X, Double[][]Y, int epochs) throws Exception {
        /**
         * Gradient Descent with backpropagation.
         * @param X Inputs as a 2d Double object array.
         * @param Y Corresponding outputs as a 2d Double object array
         * @param epochs The number of epochs to train the neural network.
         */
        for(int i=0;i<epochs;i++) {

            if (shuffle) {
                //use a random input/output combination as training data.
                int sampleN = (int) (Math.random() * X.length);
                this.train(X[sampleN], Y[sampleN]);
            } else{
                this.train(X[i%X[0].length], Y[i%Y[0].length]);
                shuffle = !shuffle;
            }

            if(flag) {
                System.out.println("Converged at " + i + " epochs");
                flag = false;
                return;
            }
        }
    }

    public void fit(Double[][]X, Double[][]Y, int epochs, int batch_size) throws Exception {
        /**
         * Batch gradient descent, currently not working properly.
         * @param X Inputs as a 2d Double object array.
         * @param Y Corresponding outputs as a 2d Double object array
         * @param epochs The number of epochs to train the neural network.
         * @param batch_size The desired batch size for the neural network inputs.
         */
        for(int i=0;i<epochs;i++)
        {
            ArrayList<Double[]> batch_xs = new ArrayList<>(), batch_ys = new ArrayList<>();
            for(int j = 0; j < batch_size; j++) {
                for(int k = 0; k < X.length; k++) {
                    batch_xs.add(X[k]);
                    batch_ys.add(Y[k]);
                }
            }
            train(batch_xs, batch_ys);
        }
    }
}
