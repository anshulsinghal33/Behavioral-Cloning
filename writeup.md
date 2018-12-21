
# **Behavioral Cloning** 
**Using Deep Learning via Keras to Clone Driving Behavior**

---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: writeup_images/nvidia_model.png
[image2]: writeup_images/sample_camera_imgs_vis.png
[image3]: writeup_images/model_plot.png

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results
* `project_video.mp4` showing the recorded video output of the autonomous driving using `model.h5`

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

`python drive.py model.h5`


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The obvious start for me was to first follow the methods and simple networks shown in the lessons and try what works and what doesn't. That really helped gain a very important insight into the importance of data collection and augmentation and its impact on improving the driving behaviour of the car even for the simple networks trained during lessons. I then started experimenting with different model architectures suggested in the lessons and elsewhere but limited my efforts when I didn't see near good results. Few of them were able to make the car head in the right direction but again failed at sharp turns or at the bridge. Later I decided to try the nVidia Autonomous Car Group model, and the car somewhat drove the complete first track after just 5 training epochs. The diagram below is a depiction of the NVIDIA model architecture.

#### NVIDIA's Autonomous Car's CNN architecture _( Source: [Paper](https://arxiv.org/pdf/1604.07316v1.pdf) )_
![alt text][image1]


---
#### The Final Architecture I implemented is slightly modified. 
##### The summary of the model using the command `model.summary()` is given below:
```
                        _________________________________________________________________
                        Layer (type)                 Output Shape              Param #   
                        =================================================================
                        lambda_1 (Lambda)            (None, 160, 320, 3)       0         
                        _________________________________________________________________
                        cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
                        _________________________________________________________________
                        conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
                        _________________________________________________________________
                        conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
                        _________________________________________________________________
                        conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
                        _________________________________________________________________
                        conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
                        _________________________________________________________________
                        conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
                        _________________________________________________________________
                        flatten_1 (Flatten)          (None, 8448)              0         
                        _________________________________________________________________
                        dense_1 (Dense)              (None, 100)               844900    
                        _________________________________________________________________
                        dense_2 (Dense)              (None, 50)                5050      
                        _________________________________________________________________
                        dense_3 (Dense)              (None, 10)                510       
                        _________________________________________________________________
                        dense_4 (Dense)              (None, 1)                 11        
                        =================================================================
                        Total params: 981,819
                        Trainable params: 981,819
                        Non-trainable params: 0
                        _________________________________________________________________

```

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 100 _(model.py lines 205-217)_ 

The model includes `RELU layers` to introduce nonlinearity after each `convolutional layer` (code line 208-212), and the data is normalized in the model using a `Keras lambda layer` (code line 206). The images were also cropped to remove the sky and the hood of the car that was distracting the model with information not very useful in making steering decisions. This was also incorporated in the model using the `Keras Cropping2D` (code line 207).

#### 2. Attempts to reduce overfitting in the model

I decided not to finally include the regularization techniques like Dropout or Max pooling as it made the models performance worse when I tried it on my dataset. Instead, I decided to keep the training epochs low. In addition the model was trained and validated on different shuffled data sets (Code line 182) to ensure that the model was not overfitting. I split my sample data into training and validation data. Using 80% as training and 20% as validation (code line 229) . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually _( model.py line 240 )_.
```python
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For example, when I saw the car wasn't able to turn in the sharp cornerns, recorded more data around the particular corners in both the directions and further augmented them by flipping them (Code Line 196). Also, upon visualization of the data using a histogram of steering angle over approx. 30 bins, it became very obvious that there was around 4 times data for near straight driving conditions in comparison to the turns and approach to turns or recovery from sides to center. This was probably hindering the model to generalize the driving conditions and was fitting more towards straight driving conditions or near 0 steering angles. Removing several repeated copies of frames from the same track of straight roads helped in balancing the distribution and further improved the driving behaviour. 

This although didn't make the car drive very well but I could see the effect of data collection on the network architecture. This important learning step was possible only on simple architectures and small data samples so that the time spent on training overall epochs would be extremely small and I could spend more time collecting important and well distributed data which would serve as a great dataset for more complex architectures to be explored.

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car in the center of the lane smoothly.

My first step was to start with the most basic neural network possible and then went about adding new layers to it and gauging it performance. I thought this approach might be appropriate because because it would give me an insight into how each layer is effecting the performance of the network and also side by side understand how data augmentation was affecting the performance of these simple networks.

* First layer I added was a normalization layer to mean center the training data.
* Then my next observation was that the upper half of the frame was mostly trees,mountains and sky which harm the networks performance more than improving it so I added a cropping layer to crop it off.
* In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.
* Then I expiremented with adding more convolution layer and fully connected layers. I observed slight increase in performance but the car steered offcourse majority of the times.
* Then I decided to try the NVIDIA Autonomous Car Group model, and the car drove the complete first track after just 5 training epochs.
* Then I tried adding dropout layer to avoid overfitting but it made the models performance worse when I tried it with my dataset.

So to combat the overfitting, I decided not to modify the model rather I reduced the training epochs.

To improve the performance of the model I also used few preprocessing and data augmentation techniques which are discussed in the earlier sections.

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 205-217) consisted of a convolution neural network with the following layers and layer sizes ...

![alt text][image3]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recove from the left side and right side. Then I repeated this process on track two in order to get more data points. Here are a few examples of the driving:

![alt text][image2]


To augment the data sat, I also flipped images and angles thinking that this would help generalize the data aswell as provides more data.
After the collection process, I had 12486 number of data points. I then preprocessed this data by converting all images to RGB using cv2.cvtColor.
I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Conclusion

I could easily spend hours upon hours tuning the data and model to perform optimally on both tracks, but to manage my time effectively I chose to conclude my efforts as soon as the model performed satisfactorily on both tracks. I fully plan to revisit this project when time permits. I would also like to modify the architecture to reduce the number of parameters as it would make it a lot easier for the model to train aswell as it would be able to predict in near real time for which I plan to experiment with the comma.ai model in the future when time permits.

---
### Simulation

#### 1. Is the car able to navigate correctly on test data?
	
No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).


```python

```
