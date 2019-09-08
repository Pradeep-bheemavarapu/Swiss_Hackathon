# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'softmax'))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('animals/training_set',target_size = (64, 64))
test_set = test_datagen.flow_from_directory('animals/testing_set',target_size = (64, 64))
print(training_set)
classifier.fit_generator(training_set,steps_per_epoch = 25,epochs = 5,validation_data = test_set,validation_steps = 2000)

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('animals/validation_set/OIP-4rPAIFx0zpiMrovgDrmqLgHaF6.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = list(classifier.predict(test_image))
print('Probabilities for each class : ',result[0])
x = list(result[0])
print('Classes are : ',list(training_set.class_indices.keys())[:])
print(list(training_set.class_indices.keys())[x.index(max(x))])

test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory('animals/testing_set', 
                                                         target_size=(64, 64), 
                                                         batch_size=32, 
                                                         shuffle = False)
                                                         
predictions = classifier.predict(test_data_generator)

import numpy

predicted_classes = numpy.argmax(predictions, axis=1)

print(predicted_classes)

true_classes_data = test_data_generator.classes
class_labels_data = list(test_data_generator.classes_indices.keys())

print(true_classes_data)
print(class_labels_data)

from sklearn import metrics

confusion_matrix_results = metrics.classification_report(true_classes_data, predicted_classes,target_names = class_labels_data)

print(confusion_matrix_results)
