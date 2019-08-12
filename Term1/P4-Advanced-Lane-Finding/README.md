# Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Project Instructions
The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Navigating this directory
* Project pipelines are in `p4-advanced-lane-lines.ipynb`.
* Helper functions are in `helperfunctions.py`.
* The images for camera calibration are stored in the folder called `camera_cal`.  
* The images in `test_images` are for testing your pipeline on single frames.


## Project Outline:
The code for each step is in the correspondingly named section of `P4-Advanced-Lane-Lines.ipynb`.

##  Camera Calibration

### 1. Computing the camera matrix and distortion coefficients
This was done in Step 1 of the ipynb.
* Read in calibration images.
* Generate object points (points I want to map the chessboard corners to in the undistorted image).
* Find the image points (chessboard corners) using `cv2.findChessboardCorners`.
* Calibrate the camera and obtain distortion coefficients using `cv2.calibrateCamera`.

#### Example of a distortion corrected calibration image.
![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/camera%20calibration.png)


## Image Pipeline

### 2. Apply distortion correction to each image
* Apply `cv2.undistort` with the camera matrix and distortion coefficients obtained in Step 1. 

#### Example of a distortion-corrected image
![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/2.png)

### 3. Perspective transform
* Show region of interest.

* Transform the image from the car camera's perspective to a birds-eye-view perspective.
* Hard-code the source and destination polygon coordinates and obtain the matrix `M` that maps them onto each other using `cv2.getPerspective`.
* Warp the image to the new birds-eye-view perspective using `cv2.warpPerspective` and the perspective transform matrix `M` we just obtained.

#### Example of a transformed image

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/3.png)


### 4. Create a thresholded binary image

* Threshold x gradient (for grayscaled image)
* Threshold colour channel (S channel)
* Combine the two binary thresholds to generate a binary image.
* The parameters (e.g. thresholds) were determined via trial and error (see Discussion). 
    * Improvement: determine the parameters in a more rigorous way.
* Warp the image to the new birds-eye-view perspective using `cv2.warpPerspective` and the perspective transform matrix `M` we just obtained.

#### Example of a thresholded binary image
![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/5.png)




### 5. Identify lane-line pixels and fit their positions with a polynomial

#### Identify lane line pixels
* Divide the image into `n` horizontal strips (steps) of equal height.
* For each step, take a count of all the pixels at each x-value within the step window using a histogram generated from `np.sum`.
* Find the peaks in the left and right halves (one half for each lane line) histogram.
* Get (add to our collection of lane line pixels) the pixels in that horizontal strip that have x coordinates close to the two peak x coordinates.

#### Fit positions of lane-line pixels with a polynomial
* Fit a second order polynomial to each lane line using `np.polyfit`.

#### Example plot
Histogram:

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/6.png)


Apply sliding window method to find lane line pixels:

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/7.png)


Lane line pixels and lines highlighted:

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/8.png)


Lane area highlighted:

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/9.png)


### 6. Plot result back down onto tho road such that the lane area is identified clearly.
* Warp lane lines back onto original image (car camera's perspective) using `cv2.warpPerspective`.
* Combine lane lines with original image (version corrected for distortion) using `cv2.add`.

#### Result: Lane lines combined with original image:

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/10.png)

### 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/11.png)

## 8. Lane Finding Pipeline

* Define a lane class in order to check lane finding result for every image generated from project video.

#### Example plot
Screenshot:

![](https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P4-Advanced-Lane-Finding/readme_images/12.jpg)

[Video output](./project_output.mp4)


