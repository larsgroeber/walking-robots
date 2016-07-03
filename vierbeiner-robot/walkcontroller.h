#ifndef __WALKCONTROLLER_H
#define __WALKCONTROLLER_H
#include "class.simpleNeuralNetwork.h"



#include <stdio.h>
#include <selforg/abstractcontroller.h>
#include <fstream>

/**
 * robot controller for vierbeiner walk (hard coded)
 */
class WalkController : public AbstractController {
public:

  WalkController();

  virtual void init(int sensornumber, int motornumber, RandGen* randGen = 0);
  virtual int getSensorNumber() const {return number_sensors;}
  virtual int getMotorNumber() const {return number_motors;}
  virtual void step(const sensor* sensors, int sensornumber,
                    motor* motors, int motornumber);
  virtual void stepNoLearning(const sensor* , int number_sensors,
                              motor* , int number_motors);

  //// Custom ////
 
  void forwardSensor(const sensor* sensors, int sensornumber,
                          motor* motors, int motornumber, Neural_Custom* neural);
  double calFitness(double posNow[3]);
  void startNextGen();
  void forwardSensorCustom(const sensor* sensors, int sensornumber,
                          motor* motors, int motornumber);

  //// End Custom ////

  virtual std::list<iparamkey> getInternalParamNames()const  { return std::list<iparamkey>(); }

  virtual std::list<iparamval> getInternalParams() const { return std::list<iparamval>(); }

  /********* STORABLE INTERFACE ******/
  /// @see Storable
  virtual bool store(FILE* f) const {
    Configurable::print(f,"");
    return true;
  }

  /// @see Storable
  virtual bool restore(FILE* f) {
    Configurable::parse(f);
    return true;
  }


protected:

  int t;
  std::string name;
  int number_sensors;
  int number_motors;

  bool startOfSim;
  bool endOfSim;
  bool useCustom;

  //// Neural Network ////
  int inputSize = 0 + 2;          // number of input nodes
  int outputSize = 10;            // number of output nodes
  int numberOfNeurons = 2;      
  int maxTime = 500;              // max time each network has (in sim-steps)
  
  int numberOfNetworks = 50;      // number of networks to be used per generation
  int generation = 1;             // current generation
  int curNetID = 0;               // current network ID
  double penalty;
  

  std::vector<Neural_Custom*> networkList;
  std::vector<Neural_Custom*> nextNetworkList;

  std::vector< std::vector<Neural_Custom*> > generationList;

  // to keep weights of the network with the highest fitness
  Neural_Custom* bestNetwork;
  double highestFitness;

  //// End Network ////



  mat input = mat(1,inputSize);
  mat output = mat(1,outputSize);

  double startPos[3];   // array holding the starting position of the robot
  double posArray[3];   // array holding the current position of the robot

  double lastPos[3];

  double averageSpeed;
  double totalSpeed;
  double distanceThen;

  paramval speed;
  paramval sinMod;
  paramval kneeamplitude;
  paramval hipamplitude;

  paramval resetRobot;            // pseudo parameter to cummunicate with simulation, see main.cpp -> addCallBack
  paramval numberOfGenerations;   // number of generations to run through

  std::ofstream ofFile;
  std::string fitnessFile;
  std::ofstream motorFile;
};

#endif
