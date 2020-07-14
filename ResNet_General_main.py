import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ResNet_general import ResNet
import h5py

np.random.seed(1)

def load_dataset():
    train_dataset = h5py.File('train_signs.h5',"r")
    test_dataset = h5py.File('test_signs.h5',"r")

    train_set_x = np.array(train_dataset["train_set_x"][:]) #train set features
    train_set_y = np.array(train_dataset["train_set_y"][:]) #train set labels

    test_set_x = np.array(test_dataset["test_set_x"][:]) #test set features
    test_set_y = np.array(test_dataset["test_set_y"][:]) #test set labels

    classes = np.array(test_dataset["list_classes"][:]) #list of classes

    return train_set_x, train_set_y, test_set_x, test_set_y, classes

def create_one_hot(Y,C):
    """
    Y: labels
    C: number of classes
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def resnet_util(resnet_type):
    resnet_dict = {'resnet_18':{'building_block':'basic_block', 'repetitions':[2,2,2,2]},
                   'resnet_34':{'building_block':'basic_block', 'repetitions':[3,4,6,3]},
                   'resnet_50':{'building_block':'bottleneck', 'repetitions':[3,4,6,3]},
                   'resnet_101':{'building_block':'bottleneck', 'repetitions':[3,4,23,3]},
                   'resnet_152':{'building_block':'bottleneck', 'repetitions':[3,8,36,3]}}
    if resnet_type not in resnet_dict.keys():
        raise Exception('Not a defined resnet architecture')
    return resnet_dict[resnet_type]

def main():
    X_train, y_train, X_test, y_test, classes = load_dataset()
    X_train = X_train/255
    X_test = X_test/255

    num_classes = len(classes)
    y_train = create_one_hot(y_train, num_classes).T
    y_test = create_one_hot(y_test, num_classes).T

    modelname = 'resnet_34' #set resnet architecture name here

    resnet_architecture = resnet_util(modelname) #returns a dictionary of resnet architecture details
    building_block = resnet_architecture['building_block']
    repetitions = resnet_architecture['repetitions']

    resnet_network = ResNet(X_train, y_train, X_test, y_test, num_classes, modelname, building_block, repetitions)
    model, history = resnet_network.train(num_epochs = 70, batch_size = 32, learning_rate = 0.001)
    test_accuracy = resnet_network.evaluate(model)

    print('Accuracy on test set: {:.3f} %'.format(test_accuracy*100))

    plt.plot(history.history['loss'])
    plt.title('Training error')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()

main()
