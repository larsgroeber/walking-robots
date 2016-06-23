#include <math.h>
//#include <random>
#include <stdlib.h>
//#include <cstdlib>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo> // libary for linear algebra


using namespace arma;


/**
 * Class to create and use a single layered neural network
 *
 * @author: Lars Gröber
 */
class Neural_Network {

protected:
    int inputSize;
    int outputSize;
    int numberOfNeurons;

    double randomWeightsRange = 2;
    double chanceOfZero = 0.2;

    double fitness;

    double chanceOfMutate = 0.01;
    double maxMutate = 2;

    mat outputVec;
    mat inputVec;

   
    /**
     * Activation function
     */
    void sigmoid(double &input) {
        input = 1 / (1+exp(-input));
    }

public:

    mat inputWeights;   // weights on synapses going to neurons
    mat outputWeights;  // weights on synapses going to output


    /**
     * Actually use network to get output from input vector
     *
     * @param: the input matrix - dim (inputRows, inputSize)
     * @returns: outputVec - Matrix of size (inputRows, outputSize)
     */
    mat forward(mat inputVec) {
        //inputVec = normalise(inputVec, 2, 1);                     
        this->inputVec = inputVec;
        mat nInput;
        nInput = inputVec * this->inputWeights;                     // adding all inputs for every neuron together - nInput has dim(input.rows, numberOfNeurons)

        for (int i = 0; i < this->inputVec.n_rows; ++i) {           // applying sigmoid function to every neuron
            for (int j = 0; j < this->numberOfNeurons; ++j) {
                this->sigmoid(nInput(i,j));
            }
        }

        this->outputVec = nInput * this->outputWeights;             // adding all inputs for every output together - outputVec has dim(inputRows, outputSize)

        for (int i = 0; i < this->inputVec.n_rows; ++i) {           // applying sigmoid function to every output
            for (int j = 0; j < this->outputSize; ++j) {
                this->sigmoid(this->outputVec(i,j));
            }
        }

        //outputVec = normalise(outputVec, 2, 1);
        outputVec.print();    
        return outputVec;
    }

    /**
     * Initialize the neural network
     *
     * @param: information needed for the size of the neural network
     */
    void initNetwork(int inputSize, int outputSize, int numberOfNeurons) {
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        this->numberOfNeurons = numberOfNeurons;

        this->inputWeights.set_size(inputSize, numberOfNeurons);
        this->outputWeights.set_size(numberOfNeurons, outputSize);
    }

    /**
     * Initialize Weights-Matrices with random values between -randomWeightsRange and randomWeightsRange
     * or with 0 (chanceOfZero) and effectively not connecting neurons
     */
    void initWeightsRandom() {
        std::srand(rand());        

        for (int i = 0; i < this->inputWeights.n_rows; ++i) 
            for (int j = 0; j < this->inputWeights.n_cols; ++j) 
                this->inputWeights(i,j) = ((double)rand() / INT_MAX) < 1 - this->chanceOfZero ? (double)rand() / (INT_MAX / 2*this->randomWeightsRange) -this->randomWeightsRange : 0;    

        for (int i = 0; i < this->outputWeights.n_rows; ++i) 
            for (int j = 0; j < this->outputWeights.n_cols; ++j) 
                this->outputWeights(i,j) = ((double)rand() / INT_MAX) < 1 - this->chanceOfZero ? (double)rand() / (INT_MAX / 2*this->randomWeightsRange) -this->randomWeightsRange : 0;
    }

    /**
     * Initialize Weights-Matrices with given values
     */
    void setWeights(mat inputWeights, mat outputWeights) {
        this->inputWeights = inputWeights;
        this->outputWeights = outputWeights;
    }   

};

/**
 * This class contains all custom functions for training and breeding the neural network
 *
 * @author: Lars Gröber
 */
class Neural_Custom: public Neural_Network {

protected:
    

public:
    /**
     * Set and retrive the fitness-value 
     */
    void setFitness(double fitness) {
        this->fitness = fitness;
    }

    double getFitness() {
        return this->fitness;
    }

    /**
     * Breed a new network with another network and initialize it (has the same topology)
     *
     * @param: the other network to breed with
     * @return: the newly created network
     */
    Neural_Custom* breed(Neural_Custom* other_Neural) {

        mat theirInputW = other_Neural->inputWeights;
        mat theirOutputW = other_Neural->outputWeights;
        
        mat newInputW = theirInputW;
        mat newOutputW = theirOutputW;

        for (int i = 0; i < this->inputWeights.n_rows; ++i) 
            for (int j = 0; j < this->inputWeights.n_cols; ++j) 
                newInputW(i,j) = rand() < INT_MAX / 2 ? theirInputW(i,j) : this->inputWeights(i,j);     // with a chance of 1/2 use the own weight for the new network 
        

        for (int i = 0; i < this->outputWeights.n_rows; ++i) 
            for (int j = 0; j < this->outputWeights.n_cols; ++j)
                newOutputW(i,j) = rand() < INT_MAX / 2 ? theirOutputW(i,j) : this->outputWeights(i,j);

        Neural_Custom* newNetwork = new Neural_Custom;
        newNetwork->initNetwork(this->inputSize, this->outputSize, this->numberOfNeurons);  
        newNetwork->setWeights(newInputW, newOutputW);
        newNetwork->mutate();   // important part: mutate the weights

        return newNetwork;
    }

    /**
     * Function to mutate the weights randomly
     */
    void mutate() {

        // with a chance of chanceOfMutate change one weight by something between 1-maxMutate and 1+maxMutate
        for (int i = 0; i < this->inputWeights.n_rows; ++i) 
            for (int j = 0; j < this->inputWeights.n_cols; ++j)
                this->inputWeights(i,j) *= (double)rand()/INT_MAX < this->chanceOfMutate ? 1+fmod((double)rand()/(INT_MAX/2)-1, this->maxMutate) : 1;   
                
        for (int i = 0; i < this->outputWeights.n_rows; ++i) 
            for (int j = 0; j < this->outputWeights.n_cols; ++j)
                this->outputWeights(i,j) *= (double)rand()/INT_MAX < this->chanceOfMutate ? 1+fmod((double)rand()/(INT_MAX/2)-1, this->maxMutate) : 1;
    }

};

///// end of class /////