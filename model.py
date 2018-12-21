#Behavioral Cloning
#Using Deep Learning via Keras to Clone Driving Behavior

#Importing Required Libraries
import zipfile
from helpers import*
import pickle
import csv
import cv2
import os
from os.path import exists
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.utils import plot_model

%load_ext autoreload
%autoreload 2


#Defining Import Folder Paths
#Folder path for storing captured dataset from simulator
data_fp ='Data/' #fp = folder path
zip_dfp = data_fp+'Zip_Data/' #dfp = data fp
unzip_dfp = data_fp+'Unzip_Data/' #dfp = data fp
processed_dfp = data_fp+'Processed_Data/' #dfp = data fp


#Importing Dataset
#Unzipping uploaded Dataset
while True:
    decision = input("Would you like to Unzip a dataset? (Y/N): ")
    
    if decision in ['Y','y']:
        while True:
            try:
                data_set_no = int(input("Enter the dataset number for unzipping: "))
                
                if data_set_no <=0:
                    print("ERROR --- Dataset number should greater than `0`. Please try again.")  
                
                elif not exists(zip_dfp+'Data_'+str(data_set_no)+'.zip'):
                    print("ERROR --- `Data_"+str(data_set_no)+".zip` File Not Found.")
                
                else:
                    print("\nUnzipping Dataset #"+str(data_set_no)+" ...")
                    data = zip_dfp+'Data_'+str(data_set_no)+'.zip'
                    zip_ref = zipfile.ZipFile(data, 'r')
                    zip_ref.extractall(unzip_dfp)
                    zip_ref.close()
                    print("Unzipped Dataset #"+str(data_set_no)+" can be found at: "+unzip_dfp+"Data_"+str(data_set_no))
                    break
            
            except ValueError:
                print("ERROR --- Dataset number should be an integer and only greater than `0`. Please try again.")           
        break
        
    elif decision in ['N','n']:
        print("No data was uzipped.")
        break
        
    else:
        print("ERROR --- Invalid choice. Please try again.")


# Helper Functions
def pickle_load(path):
    if os.path.isfile(path):
        print("Loading Files...")
        try:
            with open(path, mode='rb') as f:
                data = pickle.load(f)
        finally:
            f.close()
            print("Files Successfully Loaded from: ",path,"\n")
        return [data['center'],data['left'],data['right'],data['steering']]
    else:
        print("File Not Found at path: ",path,"\n")
        return None

    
def pickle_dump(path,data): 
    if os.path.isfile(path):
        return print("File already exists here: ", path)
    else:
        print("Writing Data to file...")
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        finally:
            f.close()
            print("Generated file can be found here : ",path,"\n")

            
# Function to parse through csv file and get all the records
def csv_to_pickle(folder_path):
    
    csv_path = folder_path+'/driving_log.csv'
    img_path = folder_path+'/IMG/'
    
    samples = []
    
    #Loading all Records from CSV File
    if os.path.isfile(csv_path):
        print("Loading Data from CSV File: ",csv_path)
        try:
            with open(csv_path) as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None) #Skip the Header
                for line in reader:
                    samples.append(line)
        finally:
            csvfile.close()
            print("Data from file successfully loaded.")
    else:
        print("File Not Found at path: ",csv_path,"\n")
    
    #Splitting and Storing all the records in the required format in a dictonary
    data = {'center':[], 'left':[], 'right':[], 'steering':[]}
    for line in samples:
        data['center'].append(img_path+line[0].split('/')[-1])
        data['left'].append(img_path+line[1].split('/')[-1])
        data['right'].append(img_path+line[2].split('/')[-1])
        data['steering'].append(line[3].split('/')[-1])
    
    pickle_dump(folder_path+"/driving_log.p",data)

    
#Loading Samples from Data_1 CSV
read_data_set_no = '1'
csv_fp = unzip_dfp+'Data_'+read_data_set_no
csv_to_pickle(csv_fp)


#Loading Samples from Data_1 pickle
data = pickle_load('Data/Unzip_Data/Data_1/driving_log.p')


#Sample Data Visualization
indices = [0,100,275]
imgs = []
titles = []
for index in indices:
    imgs.extend([mpimg.imread(data[1][index]), mpimg.imread(data[0][index]), mpimg.imread(data[2][index])])
    titles.extend(['Left Camera Image\nSteering Angle: '+data[3][index],
          'Center Camera Image\nSteering Angle: '+data[3][index],
          'Right Camera Image\nSteering Angle: '+data[3][index]])
    
plot_images(imgs,(3,3),(20,12),titles,save_plot=1,filepath='writeup_images/sample_camera_imgs_vis')


# This function consolidates data into to arrays of image paths and its corrosponding measurements
def zip_data(center_imgs,left_imgs,right_imgs,steer_angles):
    img_paths = []
    measurements = []
    total_measure = len(steer_angles)
    
    img_paths.extend(center_imgs) #Adds all center images to the list
    #Adds all steerings measurements corresponding to center images to the list
    measurements.extend(steer_angles) 
    
    img_paths.extend(left_imgs) #Adds all left images to the list
    #Adds all corrected steerings measurements corresponding to left images to the list
    measurements.extend([float(steer_angles[i]) + 0.2 for i in range(total_measure)])
    
    img_paths.extend(right_imgs)#Adds all right images to the list
    #Adds all corrected steerings measurements corresponding to right images to the list
    measurements.extend([float(steer_angles[i]) - 0.2 for i in range(total_measure)])
    
    return list(zip(img_paths,measurements))


# Generator for batch training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples) # Shuffles data to help feed model with varied data to help generalize.
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for img_paths, measurement in batch_samples:
                originalImage = cv2.imread(img_paths)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(float(measurement))
                
                # Flipping Images to augment data to further help generalize the model.
                # Balanced right and left turning image data.
                images.append(cv2.flip(image, 1))
                angles.append(float(measurement) * -1.0) #Inverting the steering angle

            yield np.array(images) , np.array(angles)
            
            
#Model Architecture
#NVIDIA's Autonomous Car's CNN architecture
def train_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


#Model Training
model_fp = 'Models/'
print("Initiating Model Training\n")

#Preparing the sample data
samples = zip_data(data[0], data[1], data[2], data[3])

# splitting the data into test set (80%) and validation set (20%)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generating Data to feed the model in batches.
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Making Model
model = train_model()
model.summary() #To Obtain a Summary of the Model

# Training model using Adam optimizer
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
history=model.fit_generator(train_generator,samples_per_epoch=len(train_samples),nb_epoch=5,validation_data=validation_generator, nb_val_samples=len(validation_samples), verbose = 1)

#SAVING FILES ...
print("\nSaving files in the folder: ",model_fp,"\n")

#Saving Model Plot ...
model_plot_fp = model_fp+'model_plot.png'
plot_model(model,show_shapes=True,to_file=model_plot_fp)
print('Model Plot saved at: ',model_plot_fp)

Saving Model History ...
history_fp = model_fp+'history.pckl'
f = open(history_fp, 'wb')
pickle.dump(history.history, f)
f.close()
print('Model History saved at: ',history_fp)

#Saving Model ...
model_path = model_fp+'model.h5'
model.save(model_path)
print("Model Saved at: ",model_path,'\n')

print('ALL FILES HAVE BEEN SUCCESSFULLY SAVED.')