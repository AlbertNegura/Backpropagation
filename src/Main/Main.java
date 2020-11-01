package Main;
import NeuralNetwork.*;
import com.github.sh0nk.matplotlib4j.Plot;

import java.util.List;

/**
 * This is the main class for the Neural Network package, created specifically for Assignment 1 for the Advanced Concepts in Machine Learning course (2020-2021).
 * This main class features the necessary helper methods and classes in order to fully simulate a working neural network with backpropagation.
 * The remaining classes in this package were designed to be reusable in the future.
 * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
 * @date 26/10/2020
 * @version 0.1
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

    public static void main(String[] args) throws Exception {
        NeuralNetwork nn = new NeuralNetwork(8,3,8);

        Plot plt = Plot.create();
        plt.plot().add(nn.weights_ih.toArray()).label("weights_ho").linestyle("--");
        plt.plot().add(nn.weights_ho.toArray()).label("weights_ho").linestyle("-");

        nn.fit(X, Y, 50000);

        plt.plot().add(nn.weights_ih.toArray()).label("weights_ho").linestyle("--");
        plt.plot().add(nn.weights_ho.toArray()).label("weights_ho").linestyle("-");
        plt.show();


        plt = Plot.create();
        plt.plot().add(nn.error_list).label("error").linestyle("-");
        plt.show();


        List<Double> output;

        Double [][] input ={{0d,0d,0d,0d,1d,0d,0d,0d}};
        //for(Double d[]:input)
        //{
        output = nn.predict(input[0]);
        System.out.println(output.toString());
        //}
    }
}
