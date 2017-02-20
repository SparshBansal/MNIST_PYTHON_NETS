import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load the data and define the number of batches
mnist = input_data.read_data_sets("./MNIST_DATASET" , one_hot = True)
num_batches = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float',[None, 10])

# Initialize Weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# Initialize Biase
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# Defining a convolution operation
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1] , padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# Inialize weight variables for layer 1
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# Initialize weight variables for fully_connected layer
w_fc1 = weight_variable([14*14*32 , 100])
b_fc1 = bias_variable([100])

# Initialize weights and biases for final output layer
w_fc2 = weight_variable([100,10])
b_fc2 = bias_variable([10])

x_image = tf.reshape(x,[-1,28,28,1])

def neural_network_model(data):

    # Convolve and max_pool
    h_conv1 = conv2d(x_image,w_conv1)+b_conv1
    h_pool1 = max_pool_2x2(h_conv1)

    h_flat_from_pool = tf.reshape(h_pool1,[-1,14*14*32])
    h_fc1 = tf.sigmoid(tf.add(tf.matmul(h_flat_from_pool,w_fc1) , b_fc1))


    y_conv = tf.add(tf.matmul(h_fc1,w_fc2) , b_fc2)
    return y_conv


def train(x):

    # get the output from the net
    y_conv = neural_network_model(x)

    # Now define the loss function and the optimizer and train
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 30
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/num_batches)):
                epoch_x,epoch_y = mnist.train.next_batch(num_batches)
                _,c = sess.run([optimizer,cost] , feed_dict = {x : epoch_x , y : epoch_y})

            correct = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            train_accuracy = accuracy.eval(feed_dict={x:epoch_x, y: epoch_y})
            print ("Epoch ",epoch, " completed of " , hm_epochs, " loss : ",train_accuracy)


        print "Accuracy : {0}".format(accuracy.eval({x : mnist.test.images , y : mnist.test.labels}))

train(x)
