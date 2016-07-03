#include <stdio.h>
#include <math.h>
#include <fstream>
#include <stdlib.h>

#include "walkcontroller.h"


using namespace std;
using namespace arma;


WalkController::WalkController()
  : AbstractController("walkcontroller", "$Id$"){
  t=0;
  addParameterDef("speed",&speed, 30.0);
  addParameterDef("sinMod",&sinMod, 10);
  addParameterDef("hipampl",&hipamplitude, 0.8);
  addParameterDef("kneeampl",&kneeamplitude, 0.8);
  addParameterDef("resetRobot",&resetRobot, 0);
  addParameterDef("numberOfGenerations",&numberOfGenerations, 100);

  number_sensors=0;
  number_motors=0;

  startOfSim = true;
  endOfSim = false;
  highestFitness = 0;
  penalty = 0;

  fitnessFile = "fitness";
  ofFile.open(fitnessFile);
  
  motorFile.open("motorOutput");

  srand(time(NULL));

  //// Network ////
  for (int i = 0; i < numberOfNetworks; ++i)
  {
    networkList.push_back(new Neural_Custom);
    networkList[i]->initNetwork(inputSize,outputSize,numberOfNeurons);
    networkList[i]->initWeightsRandom();
  }  
  generationList.push_back(networkList);
  
  useCustom = false;
  if (useCustom) {  // use higher powerfactor
    t = maxTime;
    generation = numberOfGenerations;
    
    mat inputW = { { -3.1385,21.8814},
                   { 0,  -0.3185} };
                   
    mat outputW = { {8.9953,  -65.6696,   55.4174,   -0.7012,   -2.8553,   -2.9402,    1.2224,  0,   -1.5797,    2.2100},
                    {-0.6922,    5.5808,    0.0003,   15.7358,    1.1298,   -4.7488,    1.9581, 4.0304,    4.4995,    0.0092}};
    
    bestNetwork = new Neural_Custom;
    bestNetwork->initNetwork(inputSize,outputSize,numberOfNeurons);    
    bestNetwork->setWeights(inputW,outputW);
    startOfSim = false;
    endOfSim = true;
  }
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

  // get starting position
  assert(sensornumber == 12+3);



  if (startOfSim) {
    cout << "Starting simulation." << endl;
    cout << "Are now using network " << curNetID << " from generation " << generation << endl;
    startOfSim = false;
  }


  if (t==2) // wait two steps to get a good value
    for (int i = 0; i < 3; ++i) {
      startPos[i] = sensors[12+i];
    }
  else // get current position
    for (int i = 0; i < 3; ++i) {
      posArray[i] = sensors[12+i];
    }

  // current network in use
  Neural_Custom* curNet = !useCustom ? networkList[curNetID] : new Neural_Custom;

  if (t < maxTime && !endOfSim) {    
    // let simulation run
    forwardSensor(sensors, sensornumber, motors, motornumber, curNet);
    
    // calculate current fitness of the network
    curNet->setFitness(t > 2 ? max(curNet->getFitness(), calFitness(posArray)) : 0);

    resetRobot = 0;

    // step time forward
    t++;  
  }

  // at end of evaluation time
  else if (!endOfSim) { 
    cout << "Network " << curNetID << " got a fitness of " << curNet->getFitness() << endl;
    if (curNetID < numberOfNetworks - 1) {  

      curNetID++;       // move to next network        
    }

    //at end of generation
    else if (generation < numberOfGenerations){
      cout << "Generation " << generation << " completed." << endl;

      curNetID = 0; 
      startNextGen();   // breed new generation of networks
    }

    // at end of run
    else { 
      startNextGen();     
      cout << "Finished last generation!" << endl;
      cout << "Using now best network with fitness " << highestFitness << endl;      
      endOfSim = true;

      cout << "Best inputWeights:" << endl;
      bestNetwork->inputWeights.print();  
      cout << endl; 
      cout << "Best outputWeights:" << endl;   
      bestNetwork->outputWeights.print();
      cout << endl;
    }

    if (!endOfSim)
      cout << "Are now using network " << curNetID << " from generation " << generation << endl;

    resetRobot = 1;   // reset robot to starting position
    t = 0;
  }

  // if endOfSim == true
  else {
    forwardSensor(sensors, sensornumber, motors, motornumber, bestNetwork);
    resetRobot = 0;
    t++;  
  }  
};


double WalkController::calFitness(double posNow[3]) {
  double distanceNow = 0.;
  double currentSpeed = 0.;
  
  int penStepSize = 5;

  // calculate current distance
  for (int i = 0; i < 1; ++i)
  {
    distanceNow += -pow(startPos[i] - posNow[i],1);
  }

  // calculate penalty
  if (t == 3) {   
    penalty = 0; 
    averageSpeed = 0;
    totalSpeed = 0;
    distanceThen = distanceNow;
  } 
  else if (t % penStepSize == 0) {
    currentSpeed = (distanceNow - distanceThen) / penStepSize;
    totalSpeed += currentSpeed;
    averageSpeed = totalSpeed / ((double)(t/penStepSize));
    //penalty += abs(averageSpeed - currentSpeed);
    distanceThen = distanceNow;
    //cout << averageSpeed << endl;
  }

  //return exp(pow(averageSpeed,2)) - 1;
  return exp((double)distanceNow) - 1;
}

void WalkController::startNextGen() {
  nextNetworkList.erase(nextNetworkList.begin(), nextNetworkList.end());

  // first calculate sum of all fitnesses and get highest Fitness
  double totalFitness = 0;
  double thisHighestFitness = 0;
  for (unsigned int i = 0; i < networkList.size(); ++i) {
    double thisFitness = networkList[i]->getFitness();
    
    if (highestFitness < thisFitness) {
      highestFitness = thisFitness;
      bestNetwork = networkList[i];
    }
    
    thisHighestFitness = max(thisFitness, thisHighestFitness);
    totalFitness += networkList[i]->getFitness();
  }

  ofFile << generation << "  " << thisHighestFitness << "  " << totalFitness << endl;

  cout << "Generation " << generation << " hat an average fitness of " << totalFitness/numberOfNetworks << endl;
  cout << "And a highest fitness of " << thisHighestFitness << endl << endl;

  // Breed two networks to new one using fitness as probability
  for (int i = 0; i < numberOfNetworks; ++i) {

      double randn1 = (double)rand() / INT_MAX;
      double randn2 = (double)rand() / INT_MAX;

      double a = 0;
      int first = 0;      // first network to breed with
      int second = 0;     // second network to breed with

      for (int j = 0; j < numberOfNetworks; j++)
      {
          a += networkList[j]->getFitness() / totalFitness;

          //std::cout << a << std::endl;

          first = (a > randn1 && first == 0) ? j : first;
          second = (a > randn2 && second == 0) ? j : second;

          // make sure not to use the same network to breed with
          if (first == second && first != 0 && second != 0)
              continue;
      }
      
      //std::cout << first << ", " << second << std::endl;

      nextNetworkList.push_back(networkList[first]->breed(networkList[second]));
  }

  assert(nextNetworkList.size() == numberOfNetworks);

  networkList = nextNetworkList;

  generationList.push_back(nextNetworkList);

  generation++;
}

void WalkController::forwardSensor(const sensor* sensors, int sensornumber,
                          motor* motors, int motornumber, Neural_Custom* neural) {
  motors[0] = 0;
  motors[1] = 0;

  //stepNoLearning(sensors, sensornumber, motors, motornumber);
  

  for (int i = 0; i < inputSize; ++i)
  {
    input(0,i) = sensors[i+2];
  }
  input(0,0) = sin(t/speed) * sinMod;
  input(0,1) = sin(t/speed + (M_PI/2)) * sinMod;
  // input(0,2) = sin(t/speed + (M_PI)) * sinMod;     // zwei Inputs erscheinen besser
  // input(0,3) = sin(t/speed + (3*M_PI/2)) * sinMod;

  output = neural->forward(input);
  //output.print();

  motorFile << t << "\t";
  for (int i = 0; i < outputSize; ++i)
  {
    motors[i+2] = 2 * output(0,i) - 1;
    motorFile << motors[i+2] << "\t";
  }
  //output.print();
  motorFile << endl;
};

void WalkController::stepNoLearning(const sensor* sensors, int number_sensors,
                                    motor* motors, int number_motors) {  
  
  /*double w = t/speed;
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

  motors[0] = sin(phases[0]+2)*0.1;
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
  }*/
 

  

};