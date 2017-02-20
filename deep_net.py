import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_DATASET" , one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float',[None, 10])

def neural_network_model(data):

    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                       'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                       'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                       'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                       'biases' : tf.Variable(tf.random_normal([n_classes]))}

    l1_z = tf.add(tf.matmul(data,hidden_1_layer['weights']) , hidden_1_layer['biases'])
    l1_o = tf.nn.relu(l1_z)

    l2_z = tf.add(tf.matmul(l1_o , hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2_o = tf.nn.relu(l2_z)

    l3_z = tf.add(tf.matmul(l2_o , hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3_o = tf.nn.relu(l3_z)

    output = tf.add(tf.matmul(l3_o , output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 30
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost] , feed_dict = {x : epoch_x , y : epoch_y})
                epoch_loss += c

            print ("Epoch ",epoch, " completed of " , hm_epochs, " loss : ",epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct , 'float'))

        print "Accuracy : {0}".format(accuracy.eval({x : mnist.test.images , y : mnist.test.labels}))

train_neural_network(x)