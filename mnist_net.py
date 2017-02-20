from mnist import MNIST

import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation

mndata = MNIST('/home/sparsh/Documents/Dataset/mnist')

# Load training and testing Dataset
train_images , train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Create the network using keras Sequential()
model = Sequential([
    Dense(output_dim = 32,input_dim = 784),
    Activation('sigmoid'),
    Dense(output_dim=10),
    Activation('softmax')
]);

model.compile(optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

np_train_images = np.array(train_images);
np_train_labels = keras.utils.np_utils.to_categorical(np.array(train_labels),nb_classes = None);

model.fit(np_train_images , np_train_labels , nb_epoch = 20 , batch_size=50)

# Evaluate the model
np_test_images = np.array(test_images);
np_test_labels = keras.utils.np_utils.to_categorical(np.array(test_labels),nb_classes = None);
scores = model.evaluate(np_test_images , nptest_labels);

# Print the score
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
