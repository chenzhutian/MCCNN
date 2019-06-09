'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ShapeNet.py

    \brief Code to train a segmentation network on the ShapeNet dataset.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import time
import argparse
import importlib
import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from PyUtils import visualize_progress
from ShapeNetDataSet import ShapeNetDataSet

current_milli_time = lambda: time.time() * 1000.0

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
] 

# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []

        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def create_loss(logits, labels, weigthDecay):
    labels = tf.to_int64(tf.reshape(labels, [-1]))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    xentropyloss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    regularizer = tf.contrib.layers.l2_regularizer(scale=weigthDecay)
    regVariables = tf.get_collection('weight_decay_loss')
    regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
    return xentropyloss, regTerm


def create_accuracy(logits, labels, scope):
    _, logitsIndexs = tf.nn.top_k(logits)
    with tf.variable_scope(scope):
        return tf.metrics.accuracy(labels, logitsIndexs)


def create_trainning(lossGraph, learningRate, maxLearningRate, learningDecayFactor, learningRateDecay, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningRateDecay, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph, global_step=global_step)
    return train_op, learningRateExp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train MCCNN for segmentation tasks (ShapeNet)')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--model', default='MCSeg', help='model (default: MCSeg)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size  (default: 32)')
    parser.add_argument('--maxEpoch', default=201, type=int, help='Max Epoch  (default: 201)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDeacyFactor', default=0.2, type=float, help='Learning deacy factor (default: 0.2)')
    parser.add_argument('--learningDecayRate', default=15, type=int, help='Learning decay rate  (default: 15 Epochs)')
    parser.add_argument('--maxLearningRate', default=0.00001, type=float, help='Maximum Learning rate (default: 0.00001)')
    parser.add_argument('--useDropOut', action='store_true', help='Use drop out  (default: True)')
    parser.add_argument('--dropOutKeepProb', default=0.5, type=float, help='Keep neuron probabillity drop out  (default: 0.5)')
    parser.add_argument('--useDropOutConv', action='store_true', help='Use drop out in convolution layers (default: False)')
    parser.add_argument('--dropOutKeepProbConv', default=0.8, type=float, help='Keep neuron probabillity drop out in convolution layers (default: 0.8)')
    parser.add_argument('--weightDecay', default=0.0, type=float, help='Weight decay (default: 0.0)')
    parser.add_argument('--ptDropOut', default=0.8, type=float, help='Point drop out (default: 0.8)')
    parser.add_argument('--augment', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--nonunif', action='store_true', help='Train on non-uniform (default: False)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    # parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    args = parser.parse_args()

    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp ../models/%s.py %s' % (args.model, args.logFolder))
    os.system('cp ShapeNet.py %s' % (args.logFolder))
    logFile = args.logFolder+"/log.txt"

    #Write execution info.
    with open(logFile, "a") as myFile:
        myFile.write("Model: "+args.model+"\n")
        myFile.write("Grow: "+str(args.grow)+"\n")
        myFile.write("BatchSize: "+str(args.batchSize)+"\n")
        myFile.write("MaxEpoch: "+str(args.maxEpoch)+"\n")
        myFile.write("InitLearningRate: "+str(args.initLearningRate)+"\n")
        myFile.write("LearningDeacyFactor: "+str(args.learningDeacyFactor)+"\n")
        myFile.write("LearningDecayRate: "+str(args.learningDecayRate)+"\n")
        myFile.write("MaxLearningRate: "+str(args.maxLearningRate)+"\n")
        myFile.write("UseDropOut: "+str(args.useDropOut)+"\n")
        myFile.write("DropOutKeepProb: "+str(args.dropOutKeepProb)+"\n")
        myFile.write("UseDropOutConv: "+str(args.useDropOutConv)+"\n")
        myFile.write("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv)+"\n")
        myFile.write("WeightDecay: "+str(args.weightDecay)+"\n")
        myFile.write("ptDropOut: "+str(args.ptDropOut)+"\n")
        myFile.write("Augment: "+str(args.augment)+"\n")
        myFile.write("Nonunif: "+str(args.nonunif)+"\n")

    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("BatchSize: "+str(args.batchSize))
    print("MaxEpoch: "+str(args.maxEpoch))
    print("InitLearningRate: "+str(args.initLearningRate))
    print("LearningDeacyFactor: "+str(args.learningDeacyFactor))
    print("LearningDecayRate: "+str(args.learningDecayRate))
    print("MaxLearningRate: "+str(args.maxLearningRate))
    print("UseDropOut: "+str(args.useDropOut))
    print("DropOutKeepProb: "+str(args.dropOutKeepProb))
    print("UseDropOutConv: "+str(args.useDropOutConv))
    print("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv))
    print("WeightDecay: "+str(args.weightDecay))
    print("ptDropOut: "+str(args.ptDropOut))
    print("Augment: "+str(args.augment))
    print("Nonunif: "+str(args.nonunif))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    allowedSamplingsTrain=[]
    allowedSamplingsTest=[]
    if args.nonunif:
        allowedSamplingsTrain = [1, 2, 3, 4]
        allowedSamplingsTest = [0, 1, 2, 3, 4]
    else:
        allowedSamplingsTrain = [0]
        allowedSamplingsTest = [0]
    
    mTrainDataSet = ShapeNetDataSet(True, args.batchSize, args.ptDropOut, 
        allowedSamplingsTrain, args.augment)
    mTestDataSet = ShapeNetDataSet(False, 1, 1.0,
        allowedSamplingsTest, False)
    
    numTrainModels = mTrainDataSet.get_num_models()
    numBatchesXEpoch = numTrainModels/args.batchSize
    if numTrainModels%args.batchSize != 0:
        numBatchesXEpoch = numBatchesXEpoch + 1
    numTestModels = mTestDataSet.get_num_models()

    cat = mTrainDataSet.get_categories()
    segClasses = mTrainDataSet.get_categories_seg_parts()
    print(segClasses)
    print("Train models: " + str(numTrainModels))
    print("Test models: " + str(numTestModels))



    #Create variable and place holders
    global_step = tf.Variable(0, name='global_step', trainable=False)
    inPts = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None, 1])
    inFeatures = tf.placeholder(tf.float32, [None, 1])
    inCatLabels = tf.placeholder(tf.int32, [None, 1])
    inLabels = tf.placeholder(tf.int32, [None, 1])

    isTraining = tf.placeholder(tf.bool)
    keepProbConv = tf.placeholder(tf.float32)
    keepProbFull = tf.placeholder(tf.float32)
    iouVal = tf.placeholder(tf.float32)

    # create optimizer
    learningRateExp = tf.train.exponential_decay(
        args.initLearningRate, 
        global_step, args.learningDecayRate*numBatchesXEpoch, 
        args.learningDeacyFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, args.maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)

    num_gpus = 2
    inPts_batch = tf.split(inPts, num_gpus, name='input_xyz')
    inBatchIds_batch = tf.split(inBatchIds, num_gpus, name='input_batch_id')
    inFeatures_batch = tf.split(inFeatures, num_gpus, name='input_features')
    inLabels_batch = tf.split(inLabels, num_gpus, name='input_label')
    inCatLabels_batch = tf.split(inCatLabels, num_gpus, name='input_cat_labels')
    tower_grads = []
    tower_pred = []
    tower_loss = []
    tower_xentropyLoss = []
    tower_regularizationLoss = []
    tower_train_summary = []

    #Create the network
    with tf.variable_scope(tf.get_variable_scope()) as outter_scope:
        for i in range(num_gpus):
            with tf.device(assign_to_device('/gpu:%d'%(i), "/cpu:0")), tf.name_scope('gpu_%d' % (i)):
                
                logits = model.create_network(
                    inPts_batch[i], inBatchIds_batch[i], inFeatures_batch[i], inCatLabels_batch[i], 
                    1, len(cat), 50, args.batchSize, args.grow, 
                    isTraining, keepProbConv, keepProbFull, args.useDropOutConv, args.useDropOut)

                #Create loss
                xentropyLoss, regularizationLoss = create_loss(logits, inLabels_batch[i], args.weightDecay)
                loss = xentropyLoss + regularizationLoss

                #Create sumaries
                tower_train_summary.extend([
                    tf.summary.scalar('loss', loss),
                    tf.summary.scalar('loss_XEntropy', xentropyLoss),
                    tf.summary.scalar('loss_Regularization', regularizationLoss)
                ])
                # lossSummary =
                # xEntropyLossSummary = 
                # regularizationLossSummary = 

                with tf.name_scope("compute_gradients"):
                    tower_grads.append(optimizer.compute_gradients(loss))

                # collect
                tower_pred.append(logits)
                tower_xentropyLoss.append(xentropyLoss)
                tower_regularizationLoss.append(regularizationLoss)
                tower_loss.append(loss)
            outter_scope.reuse_variables()
    
    #Create training
    with tf.name_scope("apply_gradients"), tf.device("/cpu:0"):
        grads = average_gradients(tower_grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
        # NOTE: get this
        total_xentropyLoss = tf.reduce_mean(tower_xentropyLoss)
        total_regularizationLoss = tf.reduce_mean(tower_regularizationLoss)
        total_loss = tf.reduce_mean(tower_loss)

    # trainning, learningRateExp = create_trainning(loss, 
    #     args.initLearningRate, args.maxLearningRate, args.learningDeacyFactor, 
    #     args.learningDecayRate*numBatchesXEpoch, global_step)

    #Create predict labels
    predictedLabels = tf.argmax(logits, 1)

    #Create accuracy metric
    accuracyVal, accuracyAccumOp = create_accuracy(logits, inLabels, 'metrics')
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

    learningRateSumm = tf.summary.scalar('learninRate', learningRateExp)
    tower_train_summary.append(learningRateSumm)
    trainingSummary = tf.summary.merge(tower_train_summary)
    metricsSummary = tf.summary.scalar('accuracy', accuracyVal)
    metricsTestSummary = tf.summary.merge([tf.summary.scalar('Tes_Accuracy', accuracyVal), tf.summary.scalar('Test_IoU', iouVal)], name='TestMetrics')

    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver()
    
    #Create session
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #Create the summary writer
    summary_writer = tf.summary.FileWriter(args.logFolder, sess.graph)
    summary_writer.add_graph(sess.graph)
    
    #Init variables
    sess.run(init)
    sess.run(initLocal)
    step = 0
    epochStep = 0
    np.random.seed(int(time.time()))
    
    #Train
    for epoch in range(args.maxEpoch):

        startEpochTime = current_milli_time()
        startTrainTime = current_milli_time()

        epochStep = 0
        lossInfoCounter = 0
        lossAccumValue = 0.0

        sess.run(resetMetrics)

        #Iterate over all the train files
        mTrainDataSet.start_iteration()
        while mTrainDataSet.has_more_batches():

            _, points, batchIds, features, labels, catLabels, _ = mTrainDataSet.get_next_batch()
    
            _, lossRes, xentropyLossRes, regularizationLossRes, trainingSummRes, _ = \
                sess.run([train_op, total_loss, total_xentropyLoss, total_regularizationLoss, trainingSummary, accuracyAccumOp], {
                    inPts: points, 
                    inBatchIds: batchIds, 
                    inFeatures: features, 
                    inLabels: labels, 
                    inCatLabels: catLabels, 
                    isTraining: True, 
                    keepProbConv: args.dropOutKeepProbConv, 
                    keepProbFull: args.dropOutKeepProb
                })

            summary_writer.add_summary(trainingSummRes, step)

            lossAccumValue += lossRes
            lossInfoCounter += 1

            if lossInfoCounter == 10:
                endTrainTime = current_milli_time()                   
                currAccuracy, metricsSummRes = sess.run([accuracyVal, metricsSummary])
                summary_writer.add_summary(metricsSummRes, step)

                visualize_progress(epochStep, numBatchesXEpoch, "Loss: %.6f | Accuracy: %.4f | Time: %.4f" % (
                    lossAccumValue/10.0, currAccuracy*100.0, (endTrainTime-startTrainTime)/1000.0))

                with open(logFile, "a") as myfile:
                    myfile.write("Step: %6d (%4d) | Loss: %.6f | Accuracy: %.4f\n" % (step, epochStep, lossAccumValue/10.0, currAccuracy*100.0))

                sess.run(resetMetrics)
                lossInfoCounter = 0
                lossAccumValue = 0.0
                startTrainTime = current_milli_time()

            step += 1
            epochStep += 1

        endEpochTime = current_milli_time()   
        print("Epoch %3d  train time: %.4f" %(epoch, (endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch %3d  train time: %.4f \n" %(epoch, (endEpochTime-startEpochTime)/1000.0))

        if epoch%10==0:
            saver.save(sess, args.logFolder+"/model.ckpt")

        #Test data
        it = 0
        accumTestLoss = 0.0
        sess.run(resetMetrics)
        IoUxCat = [[] for i in range(len(cat))]
        mTestDataSet.start_iteration()
        while mTestDataSet.has_more_batches():

            _, points, batchIds, features, labels, catLabels, _ = mTestDataSet.get_next_batch()

            lossRes, predictedLabelsRes, _ = sess.run([total_loss, predictedLabels, accuracyAccumOp], 
                    {
                        inPts: points, 
                        inBatchIds: batchIds, 
                        inFeatures: features, 
                        inCatLabels: catLabels, 
                        inLabels: labels, 
                        isTraining: False, 
                        keepProbConv: 1.0, 
                        keepProbFull: 1.0
                    })

            accumTestLoss = accumTestLoss + lossRes
            
            #Compute IoU
            numParts = len(segClasses[cat[catLabels[0][0]][0]])
            accumIoU = 0.0
            for j in range(numParts):
                intersection = 0.0
                union = 0.0
                currLabel = segClasses[cat[catLabels[0][0]][0]][j]
                for k in range(len(labels)):
                    if labels[k] == predictedLabelsRes[k] and labels[k] == currLabel:
                        intersection = intersection + 1.0
                    if labels[k] == currLabel or predictedLabelsRes[k] == currLabel:
                        union = union + 1.0
                if union > 0.0:
                    accumIoU = accumIoU + intersection/union
                else:
                    accumIoU = accumIoU + 1.0
            accumIoU = accumIoU/float(numParts)
            IoUxCat[catLabels[0][0]].append(accumIoU)
            
            if it%100 == 0:
                visualize_progress(it, numTestModels)

            it += 1

        #Compute mean IoU
        meanIoUxCat = 0.0
        for i in range(len(IoUxCat)):
            currMean = 0.0
            for currVal in IoUxCat[i]:
                currMean = currMean + currVal
            currMean = currMean / float(len(IoUxCat[i]))
            print("Mean IoU category "+cat[i][0]+": "+str(currMean))
            meanIoUxCat = meanIoUxCat + currMean*float(len(IoUxCat[i]))
        meanIoUxCat = meanIoUxCat / float(numTestModels)

        totalAccuracy, metricsTestSummRes = sess.run([accuracyVal, metricsTestSummary], {iouVal: meanIoUxCat})
        accumTestLoss = accumTestLoss/float(numTestModels)
        summary_writer.add_summary(metricsTestSummRes, step)

        print("Loss: %.6f | Test accuracy: %.4f | Test IoU %.4f" % (accumTestLoss, totalAccuracy*100.0, meanIoUxCat*100.0))
        with open(logFile, "a") as myfile:
            myfile.write("Loss: %.6f | Test accuracy: %.4f | Test IoU %.4f\n" % (accumTestLoss, totalAccuracy*100.0, meanIoUxCat*100.0))
