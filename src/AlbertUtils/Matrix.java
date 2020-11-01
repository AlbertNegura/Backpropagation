package AlbertUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
        * This is a custom Matrix class with self-implemented functions, designed for my own general use.
        * @author Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
        * @date 26/10/2020
        * @version 1.5
        */
public class Matrix {
    public Double[][] data;
    public int rows, cols;

    public Matrix(int size){
        /**
         * Initializes a square matrix object of Double of size size x size, with each element having a value of 0.
         * @param size size of the new square array
         */
        data = new Double[size][size];
        this.rows = size;
        this.cols = size;
        for(int i = 0; i < size; i++)
            for(int j = 0; j < size; j++)
                data[i][j] = Math.random()/100;
    }

    public Matrix(){
        /**
         * Initialize a 1x1 Matrix object.
         */
        this.rows = 1;
        this.cols = 1;
        this.data = new Double[][]{{Math.random()/100}};
    }

    public Matrix(int rows, int cols){
        /**
         * Initializes a matrix object of Double of size rows x cols, with each element having a value of 0.
         * @param rows number of rows of the new array
         * @param cols number of columns of the new array
         */
        data = new Double[rows][cols];
        this.rows = rows;
        this.cols = cols;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                data[i][j] = Math.random()/100;
    }


    public Matrix(Double[][] m){
        /**
         * Initializes a matrix object that is exactly the same as the given Double[][] array.
         * @param m the 2d Double array to instantiate a matrix object out of.
         */
        this.rows = m.length;
        this.cols = m[0].length;
        this.data = new Double[rows][cols];
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                data[i][j] = m[i][j];
    }


    public void add(int scalar){
        /**
         * Basic addition function for adding a scalar quantity to every value in the Matrix.
         * @param scalar scalar value which to add to every value in the Matrix.
         */
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++)
                this.data[i][j]+=scalar;
        }
    }

    public void add(Double scalar){
        /**
         * Basic addition function for adding a scalar quantity to every value in the Matrix.
         * @param scalar scalar value which to add to every value in the Matrix.
         */
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++)
                this.data[i][j]+=scalar;
        }
    }

    public Matrix add(Matrix mat, boolean returnNewMatrix) throws Exception{
        /**
         * Basic addition function that returns the resulting Matrix object.
         * @param mat the Matrix object to be added to this object.
         * @param returnNewMatrix whether to return a new instance of a Matrix object or this Matrix object
         */
        if (this.cols!= mat.cols || this.rows != mat.rows){
            throw new Exception("Shape mismatch exception");
        }
        Double[][] newMatrix = new Double[this.rows][this.cols];
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++){
                if(returnNewMatrix)
                    newMatrix[i][j] = this.data[i][j] + mat.data[i][j];
                else
                    this.data[i][j] += mat.data[i][j];

            }
        }
        return returnNewMatrix? new Matrix(newMatrix) : this;
    }

    public void add(Matrix mat) throws Exception{
        /**
         * Basic addition function that adds all the values of the given Matrix to this Matrix.
         * @param mat the Matrix object whose values are to be added to this Matrix object.
         */
        if (this.cols!= mat.cols || this.rows != mat.rows){
            throw new Exception("Shape mismatch exception");
        }
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++)
                this.data[i][j] += mat.data[i][j];
        }
    }

    public void subtract(int scalar){
        /**
         * Inverse addition operation.
         * @see Matrix.add(int scalar)
         * @param scalar scalar value to be subtracted from every element of this Matrix object.
         */
        add(-1*scalar);
    }

    public void subtract(Double scalar){
        /**
         * Inverse addition operation.
         * @see Matrix.add(Double scalar)
         * @param scalar scalar value to be subtracted from every element of this Matrix object.
         */
        add(-1*scalar);
    }

    public Matrix subtract(Matrix mat, boolean returnNewMatrix) throws Exception{
        /**
         * Inverse addition operation.
         * @see Matrix.add(Matrix mat, boolean returnNewMatrix)
         * @param Matrix Matrix object to be subtracted from this Matrix object.
         * @param returnNewMatrix whether to return a new instance of a Matrix object or this Matrix object
         */
        if (this.cols!= mat.cols || this.rows != mat.rows){
            throw new Exception("Shape mismatch exception");
        }
        Double[][] newMatrix = new Double[this.rows][this.cols];
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++){
                if(returnNewMatrix)
                    newMatrix[i][j] = this.data[i][j] - mat.data[i][j];
                else
                    this.data[i][j] -= mat.data[i][j];

            }
        }
        return returnNewMatrix? new Matrix(newMatrix) : this;
    }

    public static Matrix subtract(Matrix a, Matrix b) throws Exception{
        /**
         * Inverse addition operation.
         * @see Matrix.add(Matrix mat, boolean returnNewMatrix)
         * @param Matrix Matrix object to be subtracted from this Matrix object.
         * @param returnNewMatrix whether to return a new instance of a Matrix object or this Matrix object
         */
        if (a.cols!= b.cols || a.rows != b.rows){
            throw new Exception("Shape mismatch exception");
        }
        Double[][] newMatrix = new Double[a.rows][b.cols];
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < b.cols; j++){
                newMatrix[i][j] = a.data[i][j] - b.data[i][j];

            }
        }
        return new Matrix(newMatrix);
    }

    public void subtract(Matrix mat) throws Exception{
        /**
         * Inverse addition operation.
         * @see Matrix.add(Matrix mat, boolean returnNewMatrix)
         * @param Matrix Matrix object to be subtracted from this Matrix object.
         */
        if (this.cols!= mat.cols || this.rows != mat.rows){
            throw new Exception("Shape mismatch exception");
        }
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++)
                this.data[i][j] -= mat.data[i][j];
        }
    }

    public static Matrix multiply(Matrix a, Matrix b) throws Exception{
        /**
         * Returns a Matrix object that is a multiplication between a and b.
         * @param a the first Matrix object.
         * @param b the second Matrix objects.
         * @return the resulting Matrix
         */
        if (a.cols != b.rows){
            throw new Exception("Shape mismatch exception");
        }
        Matrix temp = new Matrix(a.rows, b.cols);
        for(int i = 0; i < temp.rows; i++){
            for(int j = 0; j < temp.cols; j++){
                Double sum = 0d;
                for(int k = 0; k < a.cols; k++)
                    sum+= a.data[i][k]*b.data[k][j];
                temp.data[i][j] = sum;
            }
        }
        return temp;
    }

    public void multiply(Matrix a){
        /**
         * Multiplies the 2d Double array values of this Matrix to those of the given Matrix, and sets the current Matrix data values to the new values.
         * @param a the Matrix object to multiply by.
         */
        Double[][] temp = new Double[this.rows][a.cols];

        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < a.cols; j++){
                Double sum = 0d;
                for(int k = 0; k < this.cols; k++)
                    sum+= this.data[i][k]*a.data[k][j];
                temp[i][j] = sum;
            }
        }

        this.data = temp;
    }

    public void multiply(Double a){
        /**
         * Multiplies every value in this Matrix object by the given value.
         * @param a the value to multiply by.
         */
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++)
                this.data[i][j] *= a;
        }
    }

    public Matrix product(Matrix a, Matrix b) throws Exception {
        /**
         * Element-wise multiplication between the given matrix and this matrix.
         * @param a the given Matrix object.
         */
        if (a.cols != b.cols && a.rows != b.rows){
            throw new Exception("Shape mismatch exception");
        }
        Matrix temp = new Matrix(a.rows, b.cols);
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < b.cols; j++)
                temp.data[i][j] = a.data[i][j] * b.data[i][j];
        }
        return temp;
    }

    public void product(Matrix a) throws Exception {
        /**
         * Element-wise multiplication between the given matrix and this matrix.
         * @param a the given Matrix object.
         */
        if (this.cols != a.rows){
            throw new Exception("Shape mismatch exception");
        }
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++)
                this.data[i][j] *= a.data[i][j];
        }
    }

    public static Matrix transpose(Matrix a){
        /**
         * Return the transpose of the given Matrix object.
         * @param a the given Matrix object.
         * @return a new Matrix object that is the tranpose of the given Matrix object.
         */

        Matrix temp = new Matrix(a.cols, a.rows);
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.cols; j++)
                temp.data[j][i] = a.data[i][j];
        }
        return temp;
    }

    public void sigmoid() {
        /**
         * Apply the signmoid function to the elements of the Matrix.
         */
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++)
                this.data[i][j] = 1/(1+Math.exp(-this.data[i][j]));
        }

    }

    public Matrix dsigmoid() {
        /**
         * Apply the derivative signmoid function to the elements of the Matrix.
         * @return a new Matrix object after applying the derivative sigmoid function.
         */
        Matrix temp=new Matrix(rows,cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0;j < cols; j++)
                temp.data[i][j] = this.data[i][j] * (1-this.data[i][j]);
        }
        return temp;
    }

    public void sinusoid(){
        /**
         * Apply the sine function to the elements of the Matrix.
         */
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++)
                this.data[i][j] = Math.sin(-this.data[i][j]);
        }
    }

    public void softmax(){
        /**
         * Apply the softmax function to the elements of the Matrix.
         */
        Double sum = 0d;

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++)
                sum+= Math.exp(data[i][j]);
        }

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++)
                this.data[i][j] = Math.exp(this.data[i][j]) / sum;
        }
    }


    public void relu(Double summedInput){
        /**
         * Apply the Rectified Linear Unit (ReLU) activation function to this Matrix.
         * @param summedInput the threshold for the ReLU activation function.
         */
        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++)
                this.data[i][j] = Math.max(0, summedInput);
        }
    }

    public Matrix drelu(double summedInput){
        /**
         * Apply the derivative ReLU activation function.
         * @param summedInput the threshold for the ReLU activation function.
         * @return a new Matrix object after applying the derivative ReLU activation function.
         */
        Matrix temp = new Matrix(this.rows, this.cols);

        for(int i = 0; i < this.rows; i++){
            for(int j = 0; j < this.cols; j++)
                temp.data[i][j] = summedInput > 0 ? 1 : summedInput != 0 ? 0 : 0.5;
        }

        return temp;
    }

    public static Matrix fromArray(Double[] x){
        /**
         * Constructs a x.length x 1 Matrix from the given 1d Double array.
         * @param x 1d Double array to construct a Matrix out of.
         */
        Matrix temp = new Matrix(x.length, 1);
        for(int i = 0; i < x.length; i++){
            temp.data[i][0] = x[i];
        }

        return temp;
    }

    public List<Double> toArray(){
        /**
         * Transcribes the values of the matrix into a List of Double objects
         * @return the resulting List
         */
        List<Double> temp = new ArrayList<Double>();
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++)
                temp.add(data[i][j]);
        }

        return temp;
    }

}
