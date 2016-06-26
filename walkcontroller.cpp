#include <stdio.h>
#include <math.h>
#include <fstream>

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

  number_sensors=0;
  number_motors=0;

  startOfSim = true;
  endOfSim = false;
  highestFitness = 0;

  fitnessFile = "fitness";
  ofFile.open(fitnessFile);

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
  if (useCustom) {  // does not work yet
    t = maxTime;
    generation = numberOfGenerations + 1;
    
    mat inputW = { { 1.81, -0.4901},
                   { -3.9227, 1.3830} };
                   
    mat outputW = { {-4.3729, 0.0423,-0.3863, 7.9904,0.3091,-5.4452,0,-5.2018,0,0.0144},
                    {6.7878,-1.8403,4.2642,0.1563,-36.3675,0.0514,-3.5695,4.4355,9.0594,1.8276}};

    Neural_Custom* neural;
    neural = networkList[0];
    cout << "test" << endl;
    neural->setWeights(inputW,outputW);
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
  Neural_Custom* curNet = generationList[generation-1][curNetID];


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
  double result = 0.;
  for (int i = 0; i < 2; ++i)
  {
    result += pow(startPos[i] - posNow[i], 2);
  }
  return exp((double)sqrt(result))-1;
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

  cout << "Generation " << generation << " hat a total fitness of " << totalFitness << endl;
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
  input(0,1) = sin(t/speed) * sinMod;
  input(0,0) = sin(t/speed + (M_PI/2)) * sinMod;

  output = neural->forward(input);
  //output.print();

  for (int i = 0; i < outputSize; ++i)
  {
    motors[i+2] = 2 * output(0,i) - 1;
  }
  //output.print();
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