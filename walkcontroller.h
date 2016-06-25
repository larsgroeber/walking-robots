#ifndef __WALKCONTROLLER_H
#define __WALKCONTROLLER_H
#include "class.simpleNeuralNetwork.h"



#include <stdio.h>
#include <selforg/abstractcontroller.h>

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
  void setRobot();
  void forwardSensor(const sensor* sensors, int sensornumber,
                          motor* motors, int motornumber, Neural_Custom* neural);
  double calFitness(double posNow[3]);
  void startNextGen();


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


  //// Neural Network ////
  int inputSize = 0 + 1;
  int outputSize = 10;
  int numberOfNeurons = 5;
  int maxTime = 500;
  
  int numberOfNetworks = 3;     // number of networks to be used per generation
  int numberOfGenerations = 5;  // number of generations to run through
  int generation = 1;
  int curNetID = 0;


  std::vector<Neural_Custom*> networkList;
  std::vector<Neural_Custom*> nextNetworkList;

  //// End Network ////



  mat input = mat(1,inputSize);
  mat output = mat(1,outputSize);

  double startPos[3];   // array holding the starting position of the robot
  double posArray[3];   // array holding the current position of the robot

  paramval speed;
  paramval sinMod;
  paramval kneeamplitude;
  paramval hipamplitude;
};

#endif
