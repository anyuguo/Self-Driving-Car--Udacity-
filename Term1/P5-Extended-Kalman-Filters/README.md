# Extended Kalman Filters Project

In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases).

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.


**INPUT**: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


**OUTPUT**: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x

["estimate_y"] <= kalman filter estimated position y

["rmse_x"]

["rmse_y"]

["rmse_vx"]

["rmse_vy"]

---

### Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

### Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `

### Files in the Github src Folder
The files you need to work with are in the `src` folder of the github repository.

1. `main.cpp` - communicates with the Term 2 Simulator receiving data measurements, calls a function to run the Kalman filter, calls a function to calculate RMSE
2. `FusionEKF.cpp` - initializes the filter, calls the predict function, calls the update function
3. `kalman_filter.cpp` - defines the predict function, the update function for lidar, and the update function for radar
4. `tools.cpp`- function to calculate RMSE and the Jacobian matrix

The only files you need to modify are `FusionEKF.cpp`, `kalman_filter.cpp`, and `tools.cpp`.

### How the Files Relate to Each Other

1. `Main.cpp` reads in the data and sends a sensor measurement to `FusionEKF.cpp`
2. `FusionEKF.cpp` takes the sensor data and initializes variables and updates variables. The Kalman filter equations are not in this file. `FusionEKF.cpp` has a variable called `ekf_`, which is an instance of a `KalmanFilter` class. The `ekf_` will hold the matrix and vector values. You will also use the `ekf_` instance to call the predict and update equations. 
3. The `KalmanFilter` class is defined in `kalman_filter.cpp` and `kalman_filter.h`. You will only need to modify 'kalman_filter.cpp', which contains functions for the prediction and update steps.

### Accuracy

#### The px, py, vx, vy output coordinates have an RMSE <= [.11, .11, 0.52, 0.52] when using the file: "obj_pose-laser-radar-synthetic-input.txt". 
My code gave me this RMSE measurement for file `obj_pose-laser-radar-synthetic-input.txt`:
```
Accuracy - RMSE:
 0.0973
 0.0855
 0.4513
 0.4399
```
The accuracy of the program is within the specified parameters.


### Follows the Correct Algorithm

#### Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.
I took the template code offered by Udacity and filled in the `TODO` entries with the expected implementation. Basically, the implementation follows this structure:
  * `kalman_filter.cpp`: implements the predict and update steps, with an EKF implementation for the update step as well.
  * `FusionEKF.cpp`: consumes the flow of measurements and feeds the kalman filter accordingly.
  * `tools.cpp`: provides utilitarian functions to calculate a Jacobian matrix, transform measurements from cartesian to polar system, and calculate the RMSE of the program.
  * `main.cpp`: loads the data, feeds the FusionEKF with the measurements, and keeps track of the ground truth to provide an RMSE measurement at the end of the execution.

#### Your Kalman Filter algorithm handles the first measurements appropriately.
When the fusion EKF is not initialized, the first measurement is used to initialize the state of `x`. If the measurement is radar data, we convert it from the polar to the cartesian coordiante system before feeding it to the kalman filter init funcion. Regardless of the measurement type, we assume zero velocity on both axes.

Also, during this call to the kalman filter's init function we provide initialization values for `P`, `F`, `H`, `R` and `Q`. `P` is initialized with very high confidence of the position values but very low confidence of the velocity values. Both `F` and `Q` are initialized to default values but these are then immediately updated by subsequent measurement intake calls.

#### Your Kalman Filter algorithm first predicts then updates.
After the fusion EKF has been initialized we first calculate the `F`, `transposed F`, and `Q` matrices before executing the predict step. After the prediction has been made, we take the new measurement and update the kalman filter with it.

#### Your Kalman Filter can handle radar and lidar measurements.
The call to the update step of the kalman filter bifurcates depending on what type of measurement we're taking. Given that our prediction model is based off of a linear (cartesian) model, the lidar measurements use the normal kalman filter code path for the update step. On the other hand the radar measurements use an extended kalman filter update step, which calculates the Jacobian matrix for the current state of the filter at every update step. Also, given that the measurement covariance matrices (`R`) differ between lidar and radar measurements, it's set to the appropriate value before every call to update.


