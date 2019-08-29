# CarND Particle Filter Project
This project use 2-d particle filter to implement localization for a self-driving car.

Data includes a map of landmark location, a (noisy) GPS estimate of vehicle's initial location, and lots of (noisy) sensor and control data.

## Running the Code
This project involves the Term 2 Simulator. After getting uWebSocketIO, one can run this program by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh

## Files Demonstration

* [CMakeLists.txt](CMakeLists.txt) is the cmake file.
* [src](src) folder contains the source code.
* [clean.sh](clean.sh) cleans the project.
* [build.sh](build.sh) builds the project.
* [run.sh](run.sh) runs the project.
* [install-mac](install-mac.sh) install uWebSockets in Mac.
* [install-ubuntu](install-ubuntu.sh) install uWebSockets in Ubuntu.

## Accuracy
The particle filter completes execution within the time of 100 seconds. And the output of simulator shows a success.
<img src="https://github.com/jane212/CarND-Kidnapped-Vehicle/blob/master/output.png" width="500">

Here is the result video.
[Video](result.mp4)

## Algorithm
This program is built on the starter code provided by Udacity CarND team.

Some key modifications are listed below:

* Initial number of particles is set to 20
* Use sensor range to narrow down the predictions
* Do data association within update weight step to avoid extra loop
* A hard-coded initial minimum distance of 10000 is used considering the sensor range is 50
