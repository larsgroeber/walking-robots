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
  Neural_Custom neural;
  mat input = mat(1,12);
  mat output = mat(1,12);

  double startPos[3];   // array holding the starting position of the robot
  double posArray[3];   // array holding the current position of the robot

  paramval speed;
  paramval phase;
  paramval kneeamplitude;
  paramval hipamplitude;
};

#endif
