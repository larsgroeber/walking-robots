#include <math.h>
#include <stdlib.h>
#include <assert.h>

// libary for linear algebra
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>


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

    double randomWeightsRange = 4;    // weights are initialized within [-randomWeightsRange, randomWeightsRange]
    double chanceOfZero = 0.1;        // chance that weight might be zero (effectively not connecting nodes)

    mat outputVec;
    mat inputVec;


    /**
     * Activation function
     */
    void sigmoid(double &input) {
        input = 1 / (1+exp(-input));
    }

public:

    mat inputWeights;   // matrix of weights on synapses going to neurons
    mat outputWeights;  // matrix of weights on synapses going to output


    /**
     * Actually use network to get output from input vector
     *
     * @param: the input matrix - dim (inputRows, inputSize)
     * @return: outputVec - Matrix of dim(inputRows, outputSize)
     */
    mat forward(mat inputVec) {
        assert(inputVec.n_cols == inputSize);
        assert(inputWeights.n_rows == inputVec.n_cols);

        this->inputVec = inputVec;
        mat nInput;   // matrix that holds the input to the neurons

        nInput = inputVec * inputWeights;   // adding all inputs for every neuron together - nInput has dim(input.rows, numberOfNeurons)

        // applying sigmoid function to every neuron
        for (int i = 0; i < inputVec.n_rows; ++i)
            for (int j = 0; j < numberOfNeurons; ++j)
                sigmoid(nInput(i,j));

        outputVec = nInput * outputWeights;   // adding all inputs for every output together - outputVec has dim(inputRows, outputSize)

        // applying sigmoid function to every output
        for (int i = 0; i < inputVec.n_rows; ++i)
            for (int j = 0; j < outputSize; ++j)
                sigmoid(outputVec(i,j));

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
     * Initialize Weights-Matrices with random values in [-randomWeightsRange, randomWeightsRange]
     * or with 0 (chanceOfZero) and effectively not connecting neurons
     */
    void initWeightsRandom() {
        std::srand(rand());

        for (int i = 0; i < inputWeights.n_rows; ++i)
            for (int j = 0; j < inputWeights.n_cols; ++j)
                inputWeights(i,j) = ((double)rand() / RAND_MAX) < 1 - chanceOfZero ?
                2*(double)rand() / RAND_MAX * randomWeightsRange - randomWeightsRange : 0;

        for (int i = 0; i < outputWeights.n_rows; ++i)
            for (int j = 0; j < outputWeights.n_cols; ++j)
                outputWeights(i,j) = ((double)rand() / RAND_MAX) < 1 - chanceOfZero ?
                2*(double)rand() / RAND_MAX * randomWeightsRange - randomWeightsRange : 0;
    }

    /**
     * Initialize weights-watrices with given values
     *
     * @param: the desired matrizes
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

    double chanceOfMutate = 0.05;   // chance that one weight gets mutated
    double maxMutate = 2;           // weight gets multiplied with a value in [1-maxMutate, 1+maxMutate]

public:

    /**
     * Sets and retrives the fitness-value
     */
    void setFitness(double fitness) {
        this->fitness = fitness;
    }

    double getFitness() {
        return fitness;
    }

    /*bool operator<(Neural_Custom const & a, Neural_Custom const & b)
    {
        return a.getFitness() < b.getFitness();
    }*/

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

        // with a chance of 1/2 use the own weight for the new inputWeights
        for (int i = 0; i < inputWeights.n_rows; ++i)
            for (int j = 0; j < inputWeights.n_cols; ++j)
                newInputW(i,j) = rand() < INT_MAX / 2 ? theirInputW(i,j) : inputWeights(i,j);

        // with a chance of 1/2 use the own weight for the new outputWeights
        for (int i = 0; i < outputWeights.n_rows; ++i)
            for (int j = 0; j < outputWeights.n_cols; ++j)
                newOutputW(i,j) = rand() < INT_MAX / 2 ? theirOutputW(i,j) : outputWeights(i,j);

        Neural_Custom* newNetwork = new Neural_Custom;
        // initialize new network
        newNetwork->initNetwork(inputSize, outputSize, numberOfNeurons);
        newNetwork->setWeights(newInputW, newOutputW);
        // important part: mutate the weights
        newNetwork->mutate();

        return newNetwork;
    }

    /**
     * Function to mutate own weights randomly
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
