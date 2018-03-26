# This code helps to classify images in CIFAR-10 dataset using multi layer convolutional neural network 
# Derived inspiration from the Code: https://github.com/TapanBhavsar/image-classification-CIFAR-10-using-tensorflow
# The updated code is used as part of a homewrok assignment 2.
# Course : CIS 700 Advances in Deep Learning
import sys
import os
import numpy as np
import math
import cv2
import matplotlib as mp
import matplotlib.pyplot as plt
import argparse
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow as tf

#n_hidden_1 = 1024 # Features in layer 1
#n_hidden_2 = 600 # Features in layer 2
#n_input = 3072 # CIFAR-10 data input (img shape: 32*32)
n_classes = 10 # CIFAR-10 total classes 
# classes of images in CIFAR-10 dataset
class_name = ["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# image class prediction function
def classify_name(predicts):
    max = predicts[0,0]
    temp = 0
    for i in range(len(predicts[0])):
        #highest probable class is chosen
        if predicts[0,i]>max:
                max = predicts[0,i]
                temp = i
    # print the class name
    print(class_name[temp])

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    # shuffle is used in train the data
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def build_model(x, w, b, task):
    #first convolutional layer
    conv1 = tf.nn.conv2d(x,w['w1'],strides = [1,1,1,1], padding = 'SAME')
    convone = conv1
    conv1 = tf.nn.bias_add(conv1,b['b1'])
    conv1 = tf.nn.relu(conv1)

    #first pooling layer
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#Second convolutional layer
    conv2 = tf.nn.conv2d(pool1,w['w2'],strides = [1,1,1,1], padding = 'SAME')
    conv2 = tf.nn.bias_add(conv2,b['b2'])
    conv2 = tf.nn.relu(conv2)

    #second pooling layer
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#Third Convolutional layer
    conv3 = tf.nn.conv2d(pool2,w['w3'],strides = [1,1,1,1], padding = 'SAME')
    conv3 = tf.nn.bias_add(conv3,b['b3'])
    conv3 = tf.nn.relu(conv3)

    #third pooling layer
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    shape = pool3.get_shape().as_list()
    dense = tf.reshape(pool3,[-1,shape[1]*shape[2]*shape[3]])
    dense1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense,w['w4']),b['b4']))
    
    # used for training the CNN model
    out = tf.nn.bias_add(tf.matmul(dense1,w['w5']),b['b5'])
    
    # used after training the CNN
    softmax = tf.nn.softmax(out)
    
    return out, softmax, convone

# main function where network train and predict the output on random image
def main_function(num_epochs=500):
    # initialize input data shape and datatype for data and labels
    x = tf.placeholder(tf.float32,[None,32,32,3])
    y = tf.placeholder(tf.int32,[None,10])
    
    # initialize weights for every different layers also first layer shows the filter size are 5*5 and the total filters out of layer one are 32
    weights = {
		#the weights are stored in a tensor of the form [filter_height, filter_width, in_channels, out_channels]
        'w1': tf.Variable(tf.random_normal([5,5,3,32],stddev = 0.1)),
        'w2': tf.Variable(tf.random_normal([5,5,32,60],stddev = 0.1)),
        'w3': tf.Variable(tf.random_normal([4,4,60,30],stddev = 0.1)),
        'w4': tf.Variable(tf.random_normal([4*4*30,30],stddev = 0.1)),
        'w5': tf.Variable(tf.random_normal([30,10],stddev = 0.1))
    }
    
    # initialize biases for every different layers
    biases = {
        'b1': tf.Variable(tf.random_normal([32],stddev = 0.1)),
        'b2': tf.Variable(tf.random_normal([60],stddev = 0.1)),
        'b3': tf.Variable(tf.random_normal([30],stddev = 0.1)),
        'b4': tf.Variable(tf.random_normal([30],stddev = 0.1)),
        'b5': tf.Variable(tf.random_normal([10],stddev = 0.1))
    }
    
    # call model
    predict,out_predict, convone = build_model(x,weights,biases,sys.argv[1])
    out_predict = tf.nn.softmax(predict)
    # error is propogated
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = y))
    optm = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(error) # adam optimizer is used for optimization
    corr = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
    # initialize saver for saving weight and bias values
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    if not os.path.exists("model/model.ckpt") and sys.argv[1]=="train":
        # initialize tensorflow session
        sess = tf.Session()
        sess.run(init)
        # load dataset using keras and dividing the dataset into train and test
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3).astype('float32')
        X_test = X_test.reshape(-1, 32, 32, 3).astype('float32')
        X_train = X_train / 255
        X_test = X_test / 255
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)
        # training will start
        bsize=100 #batch size used
        print("Loop\t Train Loss\t Train Acc % \t Test Loss \t \tTest Acc %\n")
        for epoch in range(0,num_epochs,50):
            train_err = 0
            train_acc = 0
            train_batches = 0
            # devide data into mini batch
            for batch in iterate_minibatches(X_train, y_train, bsize, shuffle=True):
                inputs, targets = batch
                # this is update weights
                sess.run([optm],feed_dict = {x: inputs,y: targets})
                # cost function
                err,acc= sess.run([error,accuracy],feed_dict = {x: inputs,y: targets})
                train_err += err
                train_acc += acc
                train_batches += 1
                
            test_err = 0
            test_acc = 0
            test_batches = 0
            # divide validation data into mini batch without shuffle
            for batch in iterate_minibatches(X_test, y_test, bsize, shuffle=False):
                inputs, targets = batch
                sess.run([optm],feed_dict = {x: inputs,y: targets}) # this is update weights
                err, acc = sess.run([error,accuracy],feed_dict = {x: inputs,y: targets}) # cost function
                test_err += err
                test_acc += acc
                test_batches += 1
            # Current epoch with total number of epochs, training and testing loss with training and testing accuracy
            print("{}/{}\t\t{:.3f}\t\t{:.2f}\t\t{:.3f}\t\t{:.2f}\n".format(epoch, num_epochs,train_err / (train_batches), train_acc/train_batches * 100,test_err / (test_batches),(test_acc / test_batches * 100)))
            #epcoh=epoch+50
            
        # save weights values in ckpt file in given folder path
            save_path = saver.save(sess,"model/model.ckpt")

    #below portion of the code uses already trained model to test/predict
    elif (sys.argv[1]=="test"):
        sess = tf.Session()
        sess.run(init)
        #restore weights value for this neural network
        saver.restore(sess,"model/model.ckpt")
        # test the trained model using a random image (recommended to use 32x32 image
        img_s = sys.argv[2]#contains file name of the image used for testing
        img = cv2.imread(img_s)# read the image using opencv
        new_img = cv2.resize(img,dsize = (32,32),interpolation = cv2.INTER_CUBIC)
        new_img = np.asarray(new_img, dtype='float32') / 255
        img_ = new_img.reshape((-1, 32, 32, 3))
        # output prediction for above image it gives 10 numeric numbers with it's class probability
        prediction = sess.run(out_predict,feed_dict={x: img_})
        # print predicted sclass
        classify_name(prediction)
        # session to save the filters after first convolutional layer 
        units = sess.run(convone, feed_dict={x: img_})
        filters = units.shape[3]
        plt.figure(1, figsize=(32,32))
        n_columns = 6
        n_rows = math.ceil(filters/n_columns)+1
        plt.subplots_adjust(wspace=0)
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.axis('off')
            plt.subplots_adjust(wspace=0,hspace=0.001)
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.subplots_adjust(wspace=0.00001,hspace=0.001)
        plt.savefig('CONV_rslt.png')
        sess.close()
    else:
        print("Enter correct arguments")
# main function call
if __name__ == '__main__':
    main_function()