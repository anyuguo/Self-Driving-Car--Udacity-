## Traffic Sign Recognition Program
In this project, deep neural networks are used to classify traffic signs. The model is trained so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the training, the model program is tested on new images of traffic signs found on the web. 

The working code in Python along with partial outputs is presented in [Jupyter Notebook file](Traffic_Sign_Classifier.ipynb) or equivalent [HTML document](Traffic_Sign_Classifier.html).

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

### Data Set Summary & Exploration
To calculate summary statistics of the traffic signs data set the [Numpy](http://www.numpy.org/) library is used:

The size of training set is 34799
The size of the validation set is 4410
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of classes in the data set is 43

Here is an exploratory visualization of the train data set. </br></br>
<img src="https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P2-Traffic-Sign-Classifier/readme_sources/sample.png"></br></br>
This bar chart is showing how many images there are in the traninig set for each class. </br></br>
<img src="https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P2-Traffic-Sign-Classifier/readme_sources/hist.png"></br></br>


### Data preprocessing
I decided to convert the images to grayscale because it makes 3 times less the data which strongly influences on the training time.I also normalized the image data to be between -1 and 1. It prevents from the numerical unstabilities which can ocuur when the data resides far away from zero.


### Design and Test a Model Architecture
My final model architecture is the LeNet convolutional neural network which proved to work on many similar problems, especially for image classification. The model consists of two convolution layers and three fully connected layers. Average pooling was used. 

The model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x16  				|
| Fully connected		| inputs 400, outputs 120        									|
| RELU					|		
| Fully connected		| inputs 120, outputs 84        									|
| RELU					|		
| Fully connected		| inputs 84, outputs 43        									|
| Softmax				|        									|

To train the model, I used the Adam Optimizer. The batch size was 128 and number of epochs equaled 50. Initial learning rate was set on 0.001. L2 loss function was added to the main loss for the regularization with the weight 0.0001.

My final model results are:

- Validation set accuracy of <b>94.5 %</b>
- Test set accuracy of <b>93.1 %</b>


### Test a Model on New Images

Here are seveb German traffic signs that I found on the web, and I make them resized as 32 * 32:

<img src="https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P2-Traffic-Sign-Classifier/readme_sources/test.png"></br></br>

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      		| Slippery road  									| 
| Roundabout mandatory     			|No entry										|
| Stop					| Stop											|
| Children crossing	      		| Children crossing			 				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection     							|
| Road work			| Road work    							|
| Speed limit (60km/h)			| Speed limit (60km/h)    							|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 85.7%.


Here is the specific predition result of my model. </br></br>
<img src="https://github.com/anyuguo/Self-Driving-Car--Udacity-/blob/master/Term1/P2-Traffic-Sign-Classifier/readme_sources/result.png"></br></br>

From the image above, We can find that my model isn't accurate at all facing with Roundabout mandatory. The reason might be that Roundabout mandatory samples are less than 250. This reason can also explain why my model isn't 100% sure about Slippery road since there are less than 500 samples.
