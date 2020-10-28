package Ass1;
import AlbertUtils.*;
import NeuralNetwork.*;

import java.util.List;

/**
 * This is the main class for the Neural Network package, created specifically for Assignment 1 for the Advanced Concepts in Machine Learning course (2020-2021).
 * This main class features the necessary helper methods and classes in order to fully simulate a working neural network with backpropagation.
 * The remaining classes in this package were designed to be reusable in the future.
 * @author Albert Negura (i6145864)
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
        for(Double[] row : X)
            nn.fit(X, X, 50000);

        List<Double> output;

        Double [][] input ={{0d,0d,0d,0d,1d,0d,0d,0d}};
        //for(Double d[]:input)
        //{
        output = nn.predict(input[0]);
        System.out.println(output.toString());
        //}
    }
}
