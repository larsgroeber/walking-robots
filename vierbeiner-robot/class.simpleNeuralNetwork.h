#include <math.h>
//#include <random>
#include <stdlib.h>
#include <assert.h>
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

    double randomWeightsRange = 4;
    double chanceOfZero = 0.1;    

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
        assert(inputVec.n_cols == inputSize);

        this->inputVec = inputVec;
        mat nInput;
        assert(inputWeights.n_rows == inputVec.n_cols);
        nInput = inputVec * inputWeights;                     // adding all inputs for every neuron together - nInput has dim(input.rows, numberOfNeurons)

        for (int i = 0; i < inputVec.n_rows; ++i) {           // applying sigmoid function to every neuron
            for (int j = 0; j < numberOfNeurons; ++j) {
                sigmoid(nInput(i,j));
            }
        }

        outputVec = nInput * outputWeights;             // adding all inputs for every output together - outputVec has dim(inputRows, outputSize)

        for (int i = 0; i < inputVec.n_rows; ++i) {           // applying sigmoid function to every output
            for (int j = 0; j < outputSize; ++j) {
                sigmoid(outputVec(i,j));
            }
        }
        
        //outputVec = normalise(outputVec, 2, 1);    
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

        inputWeights.set_size(inputSize, numberOfNeurons);
        outputWeights.set_size(numberOfNeurons, outputSize);
    }

    /**
     * Initialize Weights-Matrices with random values between -randomWeightsRange and randomWeightsRange
     * or with 0 (chanceOfZero) and effectively not connecting neurons
     */
    void initWeightsRandom() {
        std::srand(rand());    

        for (int i = 0; i < inputWeights.n_rows; ++i) 
            for (int j = 0; j < inputWeights.n_cols; ++j) 
                inputWeights(i,j) = ((double)rand() / INT_MAX) < 1 - chanceOfZero ? 2*(double)rand() / INT_MAX * randomWeightsRange -randomWeightsRange : 0;    

        for (int i = 0; i < outputWeights.n_rows; ++i) 
            for (int j = 0; j < outputWeights.n_cols; ++j) 
                outputWeights(i,j) = ((double)rand() / INT_MAX) < 1 - chanceOfZero ? 2*(double)rand() / INT_MAX * randomWeightsRange -randomWeightsRange : 0;
        inputWeights.print();
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
    double fitness = 0;

    double chanceOfMutate = 0.05;
    double maxMutate = 2;

public:
    /**
     * Set and retrive the fitness-value 
     */
    void setFitness(double fitness) {
        this->fitness = fitness;
    }

    double getFitness() {
        return fitness;
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

        for (int i = 0; i < inputWeights.n_rows; ++i) 
            for (int j = 0; j < inputWeights.n_cols; ++j) 
                newInputW(i,j) = rand() < INT_MAX / 2 ? theirInputW(i,j) : inputWeights(i,j);     // with a chance of 1/2 use the own weight for the new network 
        

        for (int i = 0; i < outputWeights.n_rows; ++i) 
            for (int j = 0; j < outputWeights.n_cols; ++j)
                newOutputW(i,j) = rand() < INT_MAX / 2 ? theirOutputW(i,j) : outputWeights(i,j);

        Neural_Custom* newNetwork = new Neural_Custom;
        newNetwork->initNetwork(inputSize, outputSize, numberOfNeurons);  
        newNetwork->setWeights(newInputW, newOutputW);
        newNetwork->mutate();   // important part: mutate the weights

        return newNetwork;
    }

    /**
     * Function to mutate the weights randomly
     */
    void mutate() {

        // with a chance of chanceOfMutate change one weight by something between 1-maxMutate and 1+maxMutate
        for (int i = 0; i < inputWeights.n_rows; ++i) 
            for (int j = 0; j < inputWeights.n_cols; ++j)
                inputWeights(i,j) *= (double)rand()/INT_MAX < chanceOfMutate ? 1+((double)rand()/(INT_MAX/2)-1) * maxMutate : 1;   
                
        for (int i = 0; i < outputWeights.n_rows; ++i) 
            for (int j = 0; j < outputWeights.n_cols; ++j)
                outputWeights(i,j) *= (double)rand()/INT_MAX < chanceOfMutate ? 1+ ((double)rand()/(INT_MAX/2)-1) * maxMutate : 1;
    }

};

///// end of class /////