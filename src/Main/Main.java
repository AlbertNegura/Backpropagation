package Main;
import NeuralNetwork.*;
// Note that the following import allows the use of matplotlib syntax from python within java.
// It is not necessary and is merely a helper library.
// There are full instructions on how to add it to the project using gradle or maven on their github: https://github.com/sh0nk/matplotlib4j
import com.github.sh0nk.matplotlib4j.Plot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This is the main class for the Neural Network package, created specifically for Assignment 1 for the Advanced Concepts in Machine Learning course (2020-2021).
 * This main class features the necessary helper methods and classes in order to fully simulate a working neural network with backpropagation.
 * The remaining classes in this package were designed to be reusable in the future.
 * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
 * @date 26/10/2020
 * @version 1.0
 */
public class Main {
    static Double[][] X= {
            {1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d},
            {0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d},
            {0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d},
            {0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d},
            {0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d},
            {0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d},
            {0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d},
            {0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d}
    };
    static Double [][] Y= {
            {1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d},
            {0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d},
            {0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d},
            {0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d},
            {0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d},
            {0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d},
            {0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d},
            {0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d}
    };

    private static boolean test = false;

    public static void main(String[] args) throws Exception {
        NeuralNetwork nn = new NeuralNetwork(8,3,8);

        //for drawing purposes
        nn.draw = true;
        nn.j = 0;

        List<Double> initial_h = nn.weights_ih.toArray();
        initial_h.addAll(nn.bias_h.toArray());
        List<Double> initial_o = nn.weights_ho.toArray();
        initial_o.addAll(nn.bias_o.toArray());

        //plot initial weights
        Plot plt = Plot.create();
        plt.title("Initial weights and final weights"); //note that matplotlib4j doesn't allow scatter plots
        plt.plot().add(initial_h).label("Initial input -> hidden weights").linestyle("--");
        plt.plot().add(initial_o).label("Initial hidden -> output weights").linestyle("-");

        nn.fit(X, Y, 50000);

        List<Double> final_h = nn.weights_ih.toArray();
        final_h.addAll(nn.bias_h.toArray());
        List<Double> final_o = nn.weights_ho.toArray();
        final_o.addAll(nn.bias_o.toArray());

        plt.plot().add(final_h).label("Final input -> hidden weights").linestyle("--");
        plt.plot().add(final_o).label("Final hidden -> output weights").linestyle("-");
        plt.legend();
        plt.xlabel("Weight number");
        plt.ylabel("Weight value");
        plt.show();


        plt = Plot.create();
        plt.plot().add(nn.error_list).label("").linestyle("-");
        plt.title("Example convergence for a learning rate of " + nn.l_rate + " and decay " + nn.decay);
        plt.xlabel("Number of epochs");
        plt.ylabel("Error between output and prediction");
        plt.show();


        List<Double> output;

        Double [][] input ={{0d,0d,0d,0d,1d,0d,0d,0d}};
        //for(Double d[]:input)
        //{
        output = nn.predict(input[0]);
        System.out.println(output.toString());
        //}
        if(test)
            testSuite(50000);
    }

    public static void testSuite(int epochs) throws Exception {
        /**
         * Plot the effects of different learning rates and weight decays to the convergence.
         */
        Double[] l_rates = {0.01, 0.05, 0.1, 0.25, 0.5, 1d, 2d, 5d, 10d, 100d};
        Double[] decays = {0d, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1};

        ArrayList<ArrayList<Double>> errors = new ArrayList<>();


        for (Double l_rate : l_rates) {
            NeuralNetwork nn = new NeuralNetwork(8, 3, 8);
            nn.l_rate = l_rate;
            nn.decay = 0.001;
            nn.draw = true;
            nn.j = 0;
            nn.fit(X, Y, epochs);
            errors.add(nn.error_list);
        }

        Plot plt = Plot.create();
        int i = 0;
        for(ArrayList<Double> error: errors)
            plt.plot().add(error).label(""+l_rates[i++]).linestyle("-");
        plt.xlabel("Number of epochs");
        plt.ylabel("Error between output and prediction");
        plt.title("Learning rate convergence for different learning rates with shuffled data");
        plt.legend();
        plt.show();

        for (Double l_rate : l_rates) {
            NeuralNetwork nn = new NeuralNetwork(8, 3, 8);
            nn.l_rate = l_rate;
            nn.decay = 0.001;
            nn.draw = true;
            nn.shuffle = true;
            nn.j = 0;
            nn.fit(X, Y, epochs);
            errors.add(nn.error_list);
        }

        plt = Plot.create();
        i = 0;
        for(ArrayList<Double> error: errors)
            plt.plot().add(error).label(""+l_rates[i++]).linestyle("-");
        plt.xlabel("Number of epochs");
        plt.ylabel("Error between output and prediction");
        plt.title("Learning rate convergence for different learning rates");
        plt.legend();
        plt.show();

        i = 0;
        errors = new ArrayList<>();
        for (Double decay : decays) {
            NeuralNetwork nn = new NeuralNetwork(8, 3, 8);
            nn.l_rate = 0.9;
            nn.decay = decay;
            nn.draw = true;
            nn.shuffle = false;
            nn.j = 0;
            nn.fit(X, Y, epochs);

            errors.add(nn.error_list);
        }
        plt = Plot.create();
        for(ArrayList<Double> error: errors)
            plt.plot().add(error).label(""+decays[i++]).linestyle("-");
        plt.xlabel("Epochs");
        plt.ylabel("Error");
        plt.title("Decay rate convergence for different decay rates");
        plt.legend();
        plt.show();


        i = 0;
        errors = new ArrayList<>();
        for (int temp = 0; temp < 2; temp++) {
            NeuralNetwork nn = new NeuralNetwork(8, 3, 8);
            nn.l_rate = 0.9;
            nn.decay = 0.001;
            nn.draw = true;
            nn.j = 0;
            nn.fit(X, Y, epochs);

            errors.add(nn.error_list);
        }
        plt = Plot.create();
        int count = 0;
        for(ArrayList<Double> error: errors) {
            plt.plot().add(error).label(count==0 ? "unshuffled":"shuffled").linestyle("-");
            count++;
        }
        plt.xlabel("Number of epochs");
        plt.ylabel("Error between output and prediction");
        plt.title("Learning rate convergence for different learning rates");
        plt.legend();
        plt.show();

    }
}
