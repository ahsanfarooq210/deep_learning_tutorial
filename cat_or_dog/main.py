from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# initializing tyhe cnn
classifier = Sequential()
#step 1 convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
#adding a second convolution layer
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=1,activation="sigmoid"))

classifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)







