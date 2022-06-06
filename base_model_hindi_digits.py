# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 09:16:36 2021

@author: rohin
"""

from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
#from keras.datasets import mnist
#from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
#from cpar.digit import load_data
import gzip
import os
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm


BASE_URL = "https://cpar.s3.amazonaws.com/"


def download_from_s3(file_name):
   
  download_url = BASE_URL + file_name
  save_path = os.path.join('data', file_name)
  urlretrieve(download_url, save_path)

def load_data(path=None):
    """Loads the cpar-char dataset.
    # Returns
        Tuple of Numpy arrays: `(trainX, trainY), (testX, testY)`.
    """
    files = ['digit_train-labels-idx1-ubyte.gz', 'digit_train-images-idx3-ubyte.gz', 
             'digit_test-labels-idx1-ubyte.gz', 'digit_test-images-idx3-ubyte.gz']
  
    paths = []
    if path is None:
      if os.path.isdir('data') is not True:
        os.mkdir('data')
      for fname in tqdm(files):
          if os.path.exists(os.path.join('data', fname)) is False:
            download_from_s3(fname)
          paths.append(fname)

    with gzip.open(os.path.join('data', paths[0]), 'rb') as lbpath:
        trainY = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join('data', paths[1]), 'rb') as imgpath:
        trainX = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(trainY), 28, 28)

    with gzip.open(os.path.join('data', paths[2]), 'rb') as lbpath:
        testY = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join('data', paths[3]), 'rb') as imgpath:
        testX = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(testY), 28, 28)

    return (trainX, trainY), (testX, testY)

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_data()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()