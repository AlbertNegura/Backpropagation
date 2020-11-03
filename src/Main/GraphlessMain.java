package Main;
import NeuralNetwork.*;

import java.util.*;

/**
 * This is the main class for the Neural Network package, created specifically for Assignment 1 for the Advanced Concepts in Machine Learning course (2020-2021).
 * This main class features the necessary helper methods and classes in order to fully simulate a working neural network with backpropagation.
 * The remaining classes in this package were designed to be reusable in the future.
 * Note that unlike the Main class in the same package, this class utilizes command line arguments AND does not include an external graphing package.
 * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
 * @date 26/10/2020
 * @version 1.0
 */
public class GraphlessMain {
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
        // note that the following arg parsing solution was taken from
        // https://stackoverflow.com/questions/7341683/parsing-arguments-to-a-java-command-line-program
        // last accessed on 3/11/2020

        final Map<String, List<String>> params = new HashMap<>();

        List<String> options = null;
        for(int i = 0; i < args.length; i++){
            final String a = args[i];
            if(a.charAt(0) == '-'){
                if(a.length() < 2){
                    System.err.println("Error at argument + " + a);
                }

                options = new ArrayList<>();
                params.put(a.substring(1), options);
            } else if(options != null)
                options.add(a);
            else{
                System.err.println("Illegal parameter usage");
                return;
            }
        }

        int epochs = Integer.parseInt(params.get("epoch").get(0));

        // remaining code is self-implemented

        NeuralNetwork nn = new NeuralNetwork(8,3,8);

        //for drawing purposes
        nn.draw = false;

        List<Double> initial_h = nn.weights_ih.toArray();
        initial_h.addAll(nn.bias_h.toArray());
        List<Double> initial_o = nn.weights_ho.toArray();
        initial_o.addAll(nn.bias_o.toArray());

        nn.l_rate = Double.parseDouble(params.get("learn").get(0));
        nn.decay = Double.parseDouble(params.get("decay").get(0));
        nn.DEBUG = Boolean.parseBoolean(params.get("DEBUG").get(0));

        nn.fit(X, Y, epochs);

        List<Double> final_h = nn.weights_ih.toArray();
        final_h.addAll(nn.bias_h.toArray());
        List<Double> final_o = nn.weights_ho.toArray();
        final_o.addAll(nn.bias_o.toArray());


        List<Double> output;
        Double[][] input = new Double[params.get("input").size()][8];
        for(int i = 0; i < params.get("input").size(); i++){
            String inputString = params.get("input").get(i);
            for(int j = 0; j < 8; j++){
                input[i][j] = Double.parseDouble(Character.toString(inputString.charAt(j)));
            }
        }
        //for(Double d[]:input)
        //{
        for(Double[] in: input) {
            output = nn.predict(in);
            System.out.println(output.toString());
        }
        //}
        if(test)
            testSuite(epochs);
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

    }
}
