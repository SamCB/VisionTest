# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
import h5py

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.regularizers import l2
from keras.regularizers import l1l2
from keras.preprocessing import image as kIm
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import log_loss
from scipy.ndimage import imread
from scipy import misc
from scipy import stats
from skimage import transform as tf

# Whether data should be loaded from the cache.
useCache = 1
# Colour type: 1 - grey, 3 - BGR?
colourTypeGlobal = 3
# Whether to read test data and output predictions.
prepSubmission = False
# Dimensions of the image input to the conv net.
imgRows, imgCols = 16, 16
# Batch size to use in training.
batchSize = 512
# The number of epoch to run training over.
numEpoch = 40
# Whether to subtract the mean, rather than divide by 255.
subMean = False
# The number of segments to load each train (memory management).
segmentsToLoad = 101
# The random seed to be used in generating the cross validation sets.
randomState = 51
# The location at which cache data is to be stored.
cache = 'cache'
# Any extra unique identifier for cache entries.
cacheAppend = ''
# Path to the folder containing the training images.
trainPath = os.path.join('NaoNet Train Data')
# Path to the folder containing the testing images.
testPath = os.path.join('NaoNet Train Data')
# The number of classes available.
numClasses = 5
# Whether to convert images to yuv.
yuv = False

if yuv:
    cacheAppend += 'YUV'

def getIm(path, imgRows, imgCols, colourType=1):
    """
    Reads a single image, cropping and resizing as required. If the aspect ratio
    is too far off the image will not be read and None is returned.
    
    Parameters
    ----------
    path: string
        The location from which the image should be read.
    imgRows: int
        The number of rows that the output image should have.
    imgCols: int
        The number of columsn that the output image should have.
    colourType: int
        The colour type to use. 1 is greyscale, 3 is BGR?
        
    Returns
    -------
    numpy.array
        The image read converted to the dimensions specified. If the aspect 
        ratio is too far off None is returned instead.
    """

    # Load as grayscale
    if colourType == 1:
        img = cv2.imread(path, 0)
    elif colourType == 3:
        img = cv2.imread(path)
        
    # Resize.
    
    # First check aspect ratio is reasonably close.
    imShape = np.shape(img)
    if float(imShape[1])/float(imShape[0]) > 3.0 or \
                                      float(imShape[0])/float(imShape[1]) > 3.0:
        return None
    
    # If the aspect ratio is acceptable, crop. 

    # Determine which axis to crop on.
    #if imShape[1] < imShape[0]:
    #    cut = (imShape[0]-imShape[1])/2
    #    img = img[cut:((cut+1)*-1),:]
    #if imShape[0] < imShape[1]:
    #    cut = (imShape[1]-imShape[0])/2
    #    img = img[:,cut:((cut+1)*-1)]

    # And stretch the rest of the way.
    img = cv2.resize(img, (imgCols, imgRows))
    
    return img

def loadTrain(croppedFolder, imgRows, imgCols, colourType=1):
    """
    Loads a set of training data from cropped images, along with folder labels.
    
    Parameters
    ----------
    croppedFolder: string
        Path to the folder containing the labeled folders containing the cropped
        images.
    imgRows: int
        The number of rows that each image should be reshaped to have.
    imgCols: int
        The number of columns that each image should be reshaped to have.
    colourType: int
        The colour type to use. 1 is grey, 3 is BGR?
        
    Returns
    -------
    trainData: List
        A list of numpy.array representing all the images read.
    trainLabels: List
        A list of image labels, such that each image has one corresponding label
        in this list. Labels are integers corresponding to the order the class
        folders are found in croppedFolder.
    """
    
    # The images and labels read.
    trainData = []
    trainLabels = []

    # Read the images.
    print('Read train images')
    
    # Go through all folders, grabbing images.
    path = os.path.join(croppedFolder, '*')
    folders = glob.glob(path)
    for folder in range(len(folders)):
        
        # Grab all images from the folder.
        fPath = os.path.join(folders[folder], '*')
        imagePaths = glob.glob(fPath)
        for file in imagePaths:
            img = getIm(file, imgRows, imgCols, colourType)
            if img is not None:
                trainData.append(img)
                trainLabels.append(folder)
    
    # Return the collected images and labels.
    return trainData, trainLabels
    
def cacheData(data, path):
    """
    Caches the read data in a pickle file, so that it can be restored more 
    quickly for the next run.
    
    Parameters
    ----------
    data: Any
        The data to cache.
    path: string
        The location at which the data should be cached.
    """

    # If the relevant directory exists, save the data.
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
        
    # Otherwise don't save.
    else:
        print('Directory doesnt exist')

def restoreData(path):
    """
    Loads data previously stored with cacheData.
    
    Parameters
    ----------
    path: string
        The location of the data to load.
        
    Returns
    -------
    Any
        The data read, or None if the directory does not exist.
    """

    # If the data exists load and return it.
    data = None
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def saveModel(model, path):
    """
    Saves the struture of a neural network model to a JSON file and the weights
    to a h5 file. 
    
    Parameters
    ----------
    model: Model
        The Keras neural network model to be saved.
    path: str
        The path to the location at which the files are to be saved. Note that 
        the end of the path should give the name of the network but not the file
        extensions, which are added automatically.
    """
    
    json_string = model.to_json()
    open(path + 'Structure.json', 'w').write(json_string)
    model.save_weights(os.path.join(path + 'Weights.h5'), overwrite=True)

def readModel(path):
    """
    Loads the model found at path.
    
    Parameters
    ----------
    path: str
        The path to the model to be loaded. Note that the end of the path should
        give the name of the network but not the file extensions, which are
        added automatically.
        
    Returns
    -------
    Model
        A Keras neural network with the structure and weights of the one found
        at path.
    """
    
    model = model_from_json(open(os.path.join(path + 'Structure.json')).read())
    model.load_weights(os.path.join(path + 'Weights.h5'))
    return model

def splitValidationSet(train, target, testPortion):
    """
    Splits off a validation set from the data given.
    
    Parameters
    ----------
    train: List
        The training data.
    target: List
        The labels corresponding to the training data.
    testPortion:
        The proportion of the data to split off.
    """

    # Use sklearn to create the validation split.
    trainData, validationData, trainLabels, validationLabels = \
                        train_test_split(train, target, test_size=testPortion, 
                                                       random_state=randomState)
    return trainData, validationData, trainLabels, validationLabels

def create_submission(predictions, testID, info):
    """
    Not yet implemented.
    """
    return None

def readAndNormalizeLabelledData(folderPath, imgRows, imgCols, colourType=1):
    """
    Reads and normalises the trainind data found in folderPath.
    
    Parameters
    ----------
    folderPath: string
        The folder containing the training images. Each class should be in its
        own labelled folder within this folder.
    imgRows: int
        The number of rows that each image should be reshaped to have.
    imgCols: int
        The number of columns that each image should be reshaped to have.
    colourType: int
        The colour type to use. 1 is grey, 3 is BGR?
        
    Returns
    -------
    trainData: List
        A numpy.array of numpy.array representing all the images read. Note that
        it is reshaped to have dimensions in the order Image, Channel, Row, 
        Column.
    trainLabels: List
        A list of image labels, such that each image has one corresponding label
        in this list. Labels are integers corresponding to the order the class
        folders are found in croppedFolder.
    """    
    
    # Create the path to the cache.
    cachePath = os.path.join(cache, 'trainR' + str(imgRows) + 'C' + 
                              str(imgCols) + 'T' + str(colourType) + 
                                                           cacheAppend + '.dat')
    
    # If there is no cache load the images fresh and create a cache.
    if not os.path.isfile(cachePath) or useCache == 0:
        trainData, trainLabels = loadTrain(folderPath, imgRows, imgCols, 
                                                                     colourType)
        if yuv:
            for entry in range(len(trainData)):
                trainData[entry] = bgr2yuv422(trainData[entry])
        cacheData((trainData, trainLabels), cachePath)
    
    # If the cache exists, load it.
    else:
        print('Restore train from cache!')
        (trainData, trainLabels) = restoreData(cachePath)

    # Prepare images.
    trainData = np.array(trainData, dtype=np.float)
    trainLabels = np.array(trainLabels, dtype=np.uint8)
    if colourType == 1:
        trainData = trainData.reshape(trainData.shape[0], 1, imgRows, imgCols)
    else:
        trainData = trainData.transpose((0, 3, 1, 2))
    trainLabels = np_utils.to_categorical(trainLabels, numClasses)
    
    # Preparation.
    if subMean:
        trainData -= np.mean(trainData, axis = 0)
    else:
        trainData /= 255.0
    
    # Display the shape of the data read.
    print('Train shape:', trainData.shape)
    print(trainData.shape[0], 'train samples')
    
    # Return the resulting data and labels.
    return trainData, trainLabels

def bgr2yuv422(im):
    """
    Converts from bgr to yuv422. Not clear which conversion is the correct one
    to use, so one was grabbed from the Nao camera documentation. The camera
    documentation does not seem to provide the exact kind of YUV422 used 
    natively.
    
    Nao camera docs: http://www.onsemi.com/pub_link/Collateral/MT9M114-D.PDF
    Conversion: Y'Cb'Cr' Using sRGB Formulas
    
    Parameters
    ----------
    im: np.array
        The image to be converted. Should be in uint8 BGR format.
    
    Returns
    -------
    np.array
        The converted image. Note that as a YUV422 image it is now half the
        original width and has 4 channels. Each channel is a uint8.
    """

    # Avoid overflows in average calculations.
    im = im.astype(np.uint16)

    # Initialise the new image.
    imShape = im.shape
    converted = np.zeros((imShape[0], int(imShape[1]/2), 4))
    
    # Perform the conversion calculations.
    converted[:,:,0] = (0.2126*im[:,0:imShape[1]:2,2] + 
                        0.7152*im[:,0:imShape[1]:2,1] + 
                           0.0722*im[:,0:imShape[1]:2,0]) * (219.0/256.0) + 16.0
    converted[:,:,2] = (0.2126*im[:,1:imShape[1]:2,2] + 
                        0.7152*im[:,1:imShape[1]:2,1] + 
                           0.0722*im[:,1:imShape[1]:2,0]) * (219.0/256.0) + 16.0
    #print((((converted[:,:,0] + converted[:,:,2])/2.0)))
    converted[:,:,1] = (((im[:,0:imShape[1]:2,0]+im[:,1:imShape[1]:2,0])/2.0) -\
                        ((converted[:,:,0] + converted[:,:,2])/2.0)) * 0.5389 *\
                                                             (224.0/256.0) + 128
    converted[:,:,3] = (((im[:,0:imShape[1]:2,2]+im[:,1:imShape[1]:2,2])/2.0) -\
                        ((converted[:,:,0] + converted[:,:,2])/2.0)) * 0.635 *\
                                                             (224.0/256.0) + 128
    #print(converted.astype(np.uint8))
    # Return the converted image.
    return converted.astype(np.uint8)
    
def dictToList(d):
    """
    Creates a list from the given dictionary.
    
    Parameters
    ----------
    d: Dictionary
        The dictionary to convert.
    
    Returns
    -------
    List
        The dictionary items as a list with entries of the form [key, value].
    """
    
    newList = [value for key, value in d.items()]
    return newList


def mergeSeveralFoldsMean(data):
    """
    Creates an ensemble using the mean results of several CV folds.
    
    Parameters
    ----------
    data: List
        The data from which the mean is to be found.
        
    Returns
    -------
    List
        The predictions made by the ensemble for each image.
    """

    return np.mean(data, 0).tolist()


def mergeSeveralFoldsGeom(data):
    """
    Creates an ensemble using the geometric mean results of several CV folds.
    
    Parameters
    ----------
    data: List
        The data from which the mean is to be found.
        
    Returns
    -------
    List
        The predictions made by the ensemble for each image.
    """
    
    return stats.mstats.gmean(data).tolist()

def createModel(imgRows, imgCols, colourType=1):
    """
    Creates a convolutional neural network for training. The parameters of that
    network are intended to be tweaked through alterations to this function.
    
    Parameters
    ----------
    imgRows: int
        The number of rows that each image is expected to have.
    imgCols: int
        The number of columns that each image is expected to have.
    colourType: int
        The number of colour channels each image is expected to have.
        
    Returns
    -------
    Model
        A Keras model containing a convolutional neural network ready for 
        training.
    """
    
    # The number of convolutional filters to use in each layer.
    numFilters = [8, 16, 32, 64]
    # The size of the pooling areas for max pooling.
    sizePool = [2, 2, 2, 2]
    # The convolution kernel size.
    sizeConv = [3, 3, 3, 3]
    # The number of neurons in the dense layer.
    denseNeurons = 64
    # The dropout after each max pool.
    poolDropout = 0.0
    # The dropout after each dense layer.
    denseDropout = 0.0
    # The type of initialisation to use.
    initialisation = 'he_normal'
    # The type of activation function to used on non classifcation layers.
    activ = 'relu'
    
    # Initialise the model.
    model = Sequential()
    
    # Set up conv layer 1.
    if yuv:
        model.add(ZeroPadding2D((1, 1), input_shape=(4, imgRows, imgCols/2)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(3, imgRows, imgCols)))
    model.add(Convolution2D(numFilters[0], sizeConv[0], sizeConv[0], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 1.
    if yuv:
        model.add(MaxPooling2D(pool_size=(sizePool[0], sizePool[0]/2)))
    else:
        model.add(MaxPooling2D(pool_size=(sizePool[0], sizePool[0])))
    model.add(Dropout(poolDropout))
    
    # Set up conv layer 2.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[1], sizeConv[1], sizeConv[1], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 2.
    model.add(MaxPooling2D(pool_size=(sizePool[1], sizePool[1])))
    model.add(Dropout(poolDropout))
    
    # Set up conv layer 3.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[2], sizeConv[2], sizeConv[2], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 3.
    model.add(MaxPooling2D(pool_size=(sizePool[2], sizePool[2])))
    model.add(Dropout(poolDropout))
    
    # Set up conv layer 4.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[3], sizeConv[3], sizeConv[3], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 4.
    model.add(MaxPooling2D(pool_size=(sizePool[3], sizePool[3])))
    model.add(Dropout(denseDropout))

    # Set up the dense layers.
    model.add(Flatten())
    model.add(Dense(denseNeurons, W_regularizer=l2(0.01), init=initialisation, 
                                                              activation=activ))
    model.add(Dropout(denseDropout))
    
    # Finally add the classification layer
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    # Default optimiser is 'adadelta'.
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', 
                                                           metrics=['accuracy'])
    return model


def runCrossValidation(nFolds=10, loadModel = False, startFold=0):
    """
    Creates an trains a convolutional neural network to solve the classification
    task. Validataion is done through cross validation.
    
    Parameters
    ----------
    nFolds: int
        The number of cross validation folds to perform.
    loadModel: bool
        Whether the cached model should be loaded to continue training.
    """
    
    # If we are loading a model restore its predictions.
    if loadModel:
        fullTrainPred = restoreData(os.path.join('cache', 'savedModel', 
                                                               'fullTrainPred'))
        fullTestPred = restoreData(os.path.join('cache', 'savedModel', 
                                                                'fullTestPred'))
    else:
        fullTrainPred = dict()
        fullTestPred = []
    
    # Load all the images.
    trainDataFull, trainLabelsFull = readAndNormalizeLabelledData(trainPath, 
                                             imgRows, imgCols, colourTypeGlobal)
        
    # If needed read the test data.
    if prepSubmission:
        testData, testLabels = readAndNormalizeLabelledData(testPath, imgRows, 
                                                      imgCols, colourTypeGlobal)
    
    # Create a set of indexes for cross validation.
    kf = KFold(len(trainLabelsFull), n_folds=nFolds, shuffle=True, 
                                                       random_state=randomState)
                                                       
    # Perform each fold.
    numFold = 0
    for trainIndex, validationIndex in kf:
        
        # Skip folds that were completed on previous runs.
        if numFold >= startFold:

            # Report progress.
            print('Start KFold number {} from {}'.format(numFold, nFolds))
                    
            # Create the model.
            model = createModel(imgRows, imgCols, colourTypeGlobal)
            
            # Partition the drivers.
            trainData = trainDataFull[trainIndex]
            trainLabels = trainLabelsFull[trainIndex]
            validationData = trainDataFull[validationIndex]
            validationLabels = trainLabelsFull[validationIndex]
            
            """
            # Create morphed images.
            print('Creating transformed images...')
            newX = []
            newY = []
            transforms = [tf.SimilarityTransform(scale=1, rotation=math.pi/6, 
                          translation = (0,0)), tf.SimilarityTransform(scale=1, 
                                      rotation=-math.pi/6, translation = (0,0))]
            for index in range(len(trainData)):
                im = trainData[index]
                label = trainLabels[index]
                
                for transform in transforms:
                    newIm = []
                    #for i in range(3):
                    #    newIm.append(transform(im[i]))
                    newX.append(tf.warp(im, transform))
                    newY.append(label)
            
            # Add morphed images.
            trainData = np.concatenate((trainData, newX))
            trainLabels = np.concatenate((trainLabels, newY))
            """
            """
            # Create data generator.
            datagen = kIm.ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=20,
                width_shift_range=0.0,
                height_shift_range=0.0,
                horizontal_flip=False)
                
            # Should only really be needed if featurewise_center or 
            # featurewise_std_normalization is on.
            datagen.fit(trainData)
            """

            # Make the fit.
            print("Starting fit...")
            
            """
            # Make the fit.
            save_model(model, 'fold_' + str(numFold))
            model.fit(trainData, trainLabels, batch_size=batchSize, 
                      nb_epoch=numEpoch, verbose=1, 
                      validation_data=(validationData, validationLabels),
                      callbacks=[ModelCheckpoint(os.path.join('cache', 
                      'savedModel', 'fold_'+str(numFold)+'_model_weights.h5'),
                                               save_best_only=True, verbose=1)])
            """
            
            # Do epoch separately.
            score = 1000000
            predictionsValid = []
            for epoch in range(numEpoch):
            
                print('Beginning epoch ' +str(epoch+1) + ' of ' + str(numEpoch))
            
                # The different ways of making the fit.
                #model.fit_generator(datagen.flow(trainData, trainLabels, 
                #                    batch_size=batchSize), 
                #                    samples_per_epoch=len(trainData), 
                #                    verbose=1, validation_data=(validationData,
                #                                              validationLabels))
                #model.fit(trainData, trainLabels, batch_size=batchSize, 
                #          verbose=1, validation_data=(validationData, 
                #                                              validationLabels))
                model.fit(trainData, trainLabels, batch_size=batchSize, 
                                                          nb_epoch=1, verbose=1)
                
                # Validate the model.
                tempPredictions = model.predict(validationData, 
                                                batch_size=batchSize, verbose=1)
                tempAcc = model.evaluate(validationData, validationLabels,
                                                batch_size=batchSize, verbose=1)
                tempScore = log_loss(validationLabels, tempPredictions)
                print('Validation log_loss: ' + str(tempScore))
                print('Accuracy: ' + str(tempAcc[1]))
                
                # Keep the best model.
                if tempScore < score:
                    score = tempScore
                    saveModel(model, os.path.join('savedFolds', 'fold_' + 
                                                                  str(numFold)))
                    predictionsValid = tempPredictions
            
            # Restore the best model.
            model = readModel( os.path.join('savedFolds', 'fold_' + 
                                                                  str(numFold)))
            model.compile(optimizer='adadelta', loss='categorical_crossentropy')
            
            # Gather predictions for the best model.
            predictionsValid = model.predict(validationData, 
                                               batch_size=batchSize, verbose=1)
            score = log_loss(validationLabels, predictionsValid)
            print('Best score log_loss: ', score)
            
            # Store valid predictions
            for i in range(len(validationIndex)):
                fullTrainPred[validationIndex[i]] = predictionsValid[i]
                
            # Store test predictions
            if prepSubmission:
                testPredictions = model.predict(test_data, 
                                                batch_size=batchSize, 
                                                                      verbose=1)
                fullTestPred.append(test_prediction)
    
            # Save the data.
            cacheData(fullTrainPred, os.path.join('cache', 'savedModel', 
                                                               'fullTrainPred'))
            cacheData(fullTestPred, os.path.join('cache', 'savedModel', 
                                                                'fullTestPred'))
        
        # If skipping, say so.
        else:
            print('Skipped fold ' + str(numFold))
        
        # Update the fold counter.
        numFold += 1
    
    # Calculate overall log loss of the system.
    score = log_loss(trainLabelsFull, dictToList(fullTrainPred))
    print('Final log_loss: {}, rows: {} cols: {} nFolds: {} epoch: {}'.format(
                                     score, imgRows, imgCols, nFolds, numEpoch))
    infoString = 'loss_' + str(score) \
                    + '_r_' + str(imgRows) \
                    + '_c_' + str(imgCols) \
                    + '_folds_' + str(nFolds) \
                    + '_ep_' + str(numEpoch)

    # Create a final submission if appropriate.
    if prepSubmission:
	    test_res = mergeSeveralFoldsMean(fullTestPred, nFolds)
	    create_submission(test_res, testID, infoString)   
      
runCrossValidation(10000, False, 0)










































