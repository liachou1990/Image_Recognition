import pandas as pd
import numpy as np

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

from __future__ import print_function
import keras
from keras import models
from keras import layers
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model



#--------------------------------------Read the training data---------------------------------------------

#read images
written_train = np.load("written_train.npy")
written_train = written_train/255

#read speech
spoken_train_pre = np.load("spoken_train.npy")

#create timesteps for the speech data
#identify the maximum lenght
counter_list_train = []

for line in spoken_train_pre:
    counter = 0
    for element in line:
        counter += 1      
    counter_list_train.append(counter)
    

threshold = 70
N = 13

spoken_train = np.zeros((spoken_train_pre.size,threshold,N))

for line in range(spoken_train_pre.size):
    for i in range(min(len(spoken_train[0]), len(spoken_train_pre[line]))):
        spoken_train[line,i] += spoken_train_pre[line][i]
        


#read match
labels = np.load("match_train.npy")

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)


#--------------------------------------split the data into train and validation set-------------------------

text_train = written_train[:36000,:]
text_validation = written_train[36000:,:]

speech_train = spoken_train[:36000,:]
speech_validation = spoken_train[36000:,:]

match_train = labels[:36000]
match_validation = labels[36000:]


#-----------reshape the image train data to fed the first Convolutional 2D layer

text_train = np.reshape(text_train,(text_train.shape[0], 28, 28,1))
text_validation = np.reshape(text_validation,(text_validation.shape[0], 28, 28,1))




#-------------------------------------------------read the test data-----------------------------------------------------

#read images
written_test = np.load("written_test.npy")
written_test = written_test/255




#read speech
spoken_test_pre = np.load("spoken_test.npy")


spoken_test = np.zeros((spoken_test_pre.size,threshold,N))

for line in range(spoken_test_pre.size):
    for i in range(min(len(spoken_test[0]), len(spoken_test_pre[line]))):
        spoken_test[line,i] += spoken_test_pre[line][i]
        




#-----------reshape the image test data to fed the first Convolutional 2D layer
written_test = np.reshape(written_test,(written_test.shape[0], 28, 28,1))




#----------------------------------------------hybrid CNN model with 2 inputs--------------------------------------

timesteps = 70
filters = 10

#----------------------image input (1st CNN model)

img_input = layers.Input(shape=(28, 28, 1,), name='img_input') 

img_conv_1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(img_input)
batch_norm = layers.BatchNormalization()(img_conv_1)
img_pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm)


img_conv_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm) #padding same
img_pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(img_conv_2)

img_drop_1 = layers.Dropout(0.4)(img_pool_2)
img_flat_1 = layers.Flatten()(img_drop_1)

img_dense_1 = layers.Dense(64, activation='relu')(img_flat_1)
img_dense_2 = layers.Dense(10, activation='relu')(img_dense_1)



#----------------------speech input (2nd CNN model)

speech_input = layers.Input(shape=(timesteps, 13,), name='spoken_input')

speech_conv_1 = layers.Conv1D(filters,(3),padding='valid',activation='relu', strides=1)(speech_input)
speech_drop_1 = layers.Dropout(0.4)(speech_conv_1)
speech_pool_1 = layers.GlobalMaxPooling1D()(speech_drop_1)

speech_dense_1 = layers.Dense(64, activation='relu')(speech_pool_1)
speech_dense_2 = layers.Dense(10, activation='relu')(speech_dense_1)

#----------------------concatenate outputs and create new input
#----------------------for the final fully-connected layers

merge_1 = layers.concatenate([img_dense_2, speech_dense_2])
full_dense = layers.Dense(16,activation='relu')(merge_1)
full_dense_2 = layers.Dense(8, activation='relu')(full_dense)

output = layers.Dense(1, activation='sigmoid')(full_dense_2)


#-----------------------define the model and specify inputs
model = models.Model(inputs=[img_input, speech_input], outputs=output)

#-----------------------compile the model
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])




#----------------------------------------Build the Data Generator------------------------------------------------------------

class DatGenerator(object):  
    import numpy
    def __init__(self, img_file, spoken_file, match_file, batch_size):
        self.img_file = img_file
        self.spoken_file = spoken_file
        self.match_file = match_file
        self.batch_size = batch_size
        
    def generate(self):
        import numpy
        while True:

            for cbatch in range(0, self.img_file.shape[0], self.batch_size):
                data_batch = {'img_input': self.img_file[cbatch:(cbatch+self.batch_size),:,:,:],
                                   'spoken_input': self.spoken_file[cbatch:(cbatch+self.batch_size),:,:]}

                
                yield data_batch, self.match_file[cbatch:(cbatch+self.batch_size)]
              
            
            
#------------------------------------------------------------------------------------------------------------------------




#----------------------fit the data to the data generator
train_generator = DatGenerator(text_train,speech_train,match_train,128)
val_generator = DatGenerator(text_validation,speech_validation,match_validation,128)




#---------------------balance the data
#---------------------define class weights to pass them as a parameter
#---------------------when training the model
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(match_train),
                                                 match_train)





#define the early stopping parameter
es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

#create checkpoints when training the model
filepath = "weights_best_dev_v2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")




#------------------------------------------------train the model----------------------------------------

model.fit_generator(train_generator.generate(),
                           validation_data=val_generator.generate(), 
                           steps_per_epoch=len(text_train) // 128,
                           validation_steps=len(text_validation) // 128, epochs=80,
                          class_weight=class_weights, callbacks=[es,checkpoint])


#------------------------------------------------------------------------------------------------------

#----save the model
model.save("best_model.h5")

#----load the model
model = load_model("best_model.h5")

#----load the model's best weights using the checkpoint
model.load_weights("weights_best_dev_v2.hdf5")




#-------------------------predict the class of the test set (unseen data)
y_pred = model.predict([written_test,spoken_test])



#transform the output of the sigmoid classification to boolean
class_one = y_pred > 0.5

#-----generate the solution file
np.save("result.npy",class_one)

