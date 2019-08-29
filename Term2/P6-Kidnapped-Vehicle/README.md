# CarND Particle Filter Project
This project use 2-d particle filter to implement localization for a self-driving car.

Data includes a map of landmark location, a (noisy) GPS estimate of vehicle's initial location, and lots of (noisy) sensor and control data.

## Running the Code
This project involves the Term 2 Simulator. After getting uWebSocketIO, one can run this program by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh

## Accuracy
The particle filter completes execution within the time of 100 seconds. And the output of simulator shows a success.
<img src="https://github.com/jane212/CarND-Kidnapped-Vehicle/blob/master/output.png" width="500">

## Algorithm
This program is built on the starter code provided by Udacity CarND team.

Some key modifications are listed below:

* Initial number of particles is set to 20
* Use sensor range to narrow down the predictions
* Do data association within update weight step to avoid extra loop
* A hard-coded initial minimum distance of 10000 is used considering the sensor range is 50