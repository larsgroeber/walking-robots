#include <stdio.h>
#include <math.h>
#include "walkcontroller.h"


using namespace std;
using namespace arma;

WalkController::WalkController()
  : AbstractController("walkcontroller", "$Id$"){
  t=0;
  addParameterDef("speed",&speed, 12.0);
  addParameterDef("phase",&phase, 0.0);
  addParameterDef("hipampl",&hipamplitude, 0.8);
  addParameterDef("kneeampl",&kneeamplitude, 0.8);

  number_sensors=0;
  number_motors=0;

  

  neural.initNetwork(12,12,5);
  neural.initWeightsRandom();

  
};

void WalkController::init(int sensornumber, int motornumber, RandGen* randGen){
  number_sensors=sensornumber;
  number_motors=motornumber;
  if(motornumber < 12) {
    cerr << "Walkcontroller needs 12 motors!" << endl;
    exit(1);
  }
};

void WalkController::step(const sensor* sensors, int sensornumber,
                          motor* motors, int motornumber) {
  /** sensors/motors: 0: neck, 1: tail
                     2,3,4,5 : hip:  rh, lh, rf, lf
                     6,7,8,9 : knee: rh, lh, rf, lf
                     10,11   : ankle rh, lh
   */

  for (int i = 0; i < 12; ++i)
  {
    input(0,i) = motors[i];
  }

  output = neural.forward(input);

  for (int i = 0; i < 12; ++i)
  {
    motors[i] = output(0,i);
  }

};

void WalkController::stepNoLearning(const sensor* sensors, int number_sensors,
                                    motor* motors, int number_motors) {  
  step(sensors, number_sensors, motors, number_motors);
  //double w = t/speed;
  // Horse Walk from wikipedia
  /* The walk is a four-beat gait that averages about 4 mph.
     When walking, a horse's legs follow this sequence:
     left hind leg, left front leg, right hind leg, right front leg,
     in a regular 1-2-3-4 beat. .... */
  /*
  double phases[4]= { w + 2*(M_PI/2),
                      w + 0*(M_PI/2),
                      w + 3*(M_PI/2),
                      w + 1*(M_PI/2) };

  motors[0] = sin(phases[0]+2)*0;
  motors[1] = 0;

  // hips
  for(int i=0; i<4; i++)
    motors[i+2]=sin(phases[i])*hipamplitude;

  // knees
  for(int i=0; i<2; i++){
    motors[i+6]=sin(phases[i]+1.05)*kneeamplitude;
  }
  for(int i=2; i<4; i++){
    motors[i+6]=sin(phases[i]+1.8)*kneeamplitude;
  }
  // ankles
  for(int i=0; i<2; i++){
    motors[i+10]=sin(phases[i]+1.05)*0.8;
  }

  //rest sine wave
  for(int i=12; i<number_motors; i++){
    motors[i]=sin(phases[i%4])*0.77;
  }
  */

  

};