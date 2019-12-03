﻿﻿﻿﻿﻿# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./onbewerkt.JPG "Original image"
[image3]: ./crop.JPG "Cropped image"
[image2]: ./flip.JPG "Horizontally flipped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The Nvidia model is used as the network architecture. My model consists of a convolution neural network with 3x3 filter and 5x5 filter sizes and depths between 24 and 64. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (line 8). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data using augmentation techniques to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I chose to use the sample data given by Udacity. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidia architecture. The main reason I wanted to try that architecture is because of the fact that it was designed for self driving cars. I decided to keep the architecture the same. The Nvidia paper expects input sizes of (60,266,3), our training images have a different size so I changed the input size to (70,160,3).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting I also added flipped images so I had more data. The data also has an imbalance; there is way more data for left turns than for right turns. Adding horizontally flipped images will compensate for this.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

`    def model(loss='mse', optimizer='adam'):
    model = Sequential()
    #Nvidia model
    #Normalization and change of the input shape:
    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
    #Add 5 Convolutional Layers
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    #Flatten Layers
    model.add(Flatten())
    #3 fully connected layers, 1 output layer
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)

    return model 



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Driving in the center of the lane was pretty difficult. I concluded that my own driving data wouldn't be good enough to use in the project, so I decided to use the already given sample data instead.

For my augmented data set I used the left, right, center and horizontally flipped center images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I tried to find the ideal number of epochs by trial and error. I started with 50 epochs, but that was taking me too long to process. So after that I chose to try it with just 2 epochs and that turned out to be sufficient. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Original image:

![alt_text][image1]

Horizontally flipped image:

![alt_text][image2]

Cropped unflipped image:

![alt_text][image3]




