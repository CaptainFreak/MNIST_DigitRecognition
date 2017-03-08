# linear algebra
import numpy as np 
# data processing
import pandas as pd 
# data visualization
import matplotlib.pyplot as plt

#CNN
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split

#loading data
train = pd.read_csv("train.csv")
test_images = (pd.read_csv("test.csv").values).astype('float32')	
train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

#reshaping for cnn
train_images = train_images.reshape(train_images.shape[0],  28, 28)
train_images = train_images.reshape((42000, 28 * 28))

#normalization
train_images = train_images / 255
test_images = test_images / 255

from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
 
# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

#network 
model = Sequential()
model.add(Dense(64, activation='relu',input_dim=(28 * 28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='softmax'))

#training
model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy',
 metrics=['accuracy'])

history=model.fit(train_images, train_labels, 
            nb_epoch=15, batch_size=64)
predictions = model.predict_classes(test_images, verbose=0)

#saving predictions
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("predictions.csv", index=False, header=True)
