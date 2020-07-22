import numpy as np
from tensorflow import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import he_normal
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def first_conv_block(X, filters, kernel_size, strides,):
    """
    Implements the following
    Conv2D -> BatchNormalization -> ReLu Activation
    """

    X = Conv2D(filters, kernel_size = kernel_size, strides = strides, kernel_initializer = he_normal(seed = 0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    return X

def residual_block(X, block_function, filters, repetitions, is_first_layer = False):

    """
    Builds a residual block by repeating convolutional and identity blocks
    """

    for i in range(repetitions):
        init_strides = (1,1)
        if i==0 and not is_first_layer:
            init_strides = (2,2)
        X = block_function(X, filters = filters, init_strides = init_strides, is_first_block_of_first_layer = (is_first_layer and i==0))

    return X

def basic_block(X, filters, init_strides = (1,1), is_first_block_of_first_layer= False):
    input = X
    if is_first_block_of_first_layer:
        X = Conv2D(filters = filters, kernel_size = (3,3), strides = init_strides, padding = 'same', kernel_initializer = he_normal(seed = 0))(X)
    else:
        X = bn_relu_conv(X, filters = filters, kernel_size = (3,3), strides = init_strides)

    X = bn_relu_conv(X, filters=filters, kernel_size = (3,3), strides = (1,1))
    BatchNormalization(axis = 3)(X)
    residual = X
    return shortcut(residual, input)

def bottleneck(X, filters, init_strides = (1,1), is_first_block_of_first_layer = False):
    input = X
    if is_first_block_of_first_layer:
        X = Conv2D(filters = filters, kernel_size = (1,1), strides = init_strides, padding = 'same', kernel_initializer = he_normal(seed = 0))(X)
    else:
        X = bn_relu_conv(X, filters = filters, kernel_size = (1,1), strides = init_strides)
    X = bn_relu_conv(X, filters = filters, kernel_size=(3,3), strides = (1,1))
    X = bn_relu_conv(X, filters = filters*4, kernel_size = (1,1), strides = (1,1))
    # X = BatchNormalization(axis = 3)(X)
    residual = X
    return shortcut(residual, input)

def bn_relu_conv(X, filters, kernel_size, strides):
    X = BatchNormalization(axis = 3)(X) #bn
    X = Activation('relu')(X) #relu
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same', kernel_initializer = he_normal(seed =0 ))(X)
    return X

def shortcut(residual, input):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)

    stride_width = int(round(input_shape[1]/residual_shape[1]))
    stride_height = int(round(input_shape[2]/residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    if stride_width>1 or stride_height>1 or not equal_channels:
        short_cut = Conv2D(filters = residual_shape[3], kernel_size = (1,1),
                    strides = (stride_width, stride_height), padding = 'valid',
                    kernel_initializer = he_normal(seed = 0))(input)
    else:
        short_cut = input
    return Add()([short_cut, residual])


class ResNet(object):

    def __init__(self, X_train, y_train, X_test, y_test, num_classes, modelname, building_block, repetitions, init_num_filters, conv1_f):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        assert(len(self.X_train)==len(self.y_train))
        assert(len(self.X_test)==len(self.y_test))

        self.num_classes = num_classes
        self.block_optns = {'basic_block': basic_block, 'bottleneck':bottleneck}
        (m, n_H0, n_W0, n_C0) = self.X_train.shape
        self.input_shape = (n_H0, n_W0, n_C0)

        self.modelname = modelname
        self.building_block = building_block
        self.repetitions = repetitions
        self.init_num_filters = init_num_filters
        self.conv1_f = conv1_f

    def build(self):
        """
        Returns the keras 'Model'
        """

        if len(self.input_shape) != 3:
            raise Exception("Input shape should be a tuple of (n_H, n_W, n_C)")

        X_input = Input(self.input_shape)

        # Stage 1: conv -> BN -> ReLu -> MaxPool
        f = self.conv1_f
        X = first_conv_block(X_input, filters = self.init_num_filters, kernel_size = (f, f), strides = (2,2))
        X = MaxPooling2D((3,3), strides = (2,2), padding = 'same')(X)

        #Stages 2-5: conv block -> identity blocks -> batch norm -> relu
        num_filters = self.init_num_filters
        block_fn = self.block_optns[self.building_block]
        for i, r in enumerate(self.repetitions):
            X = residual_block(X, block_fn, filters=num_filters, repetitions=r, is_first_layer = (i==0))
            num_filters *= 2

        # Last activation
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        block_shape = K.int_shape(X)

        #Average pooling
        X = AveragePooling2D(pool_size = (block_shape[1], block_shape[2]), strides = (1,1))(X)

        #Flatten
        X = Flatten()(X)
        X = Dense(self.num_classes, activation = 'softmax', kernel_initializer= he_normal(seed = 0))(X)

        print('Creating '+self.modelname+'....')

        model = Model(inputs = X_input, outputs = X, name = 'ResNet')
        return model

    def train_data_preprocess(self):

      datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                                   vertical_flip =False, zoom_range=0.2, fill_mode='nearest')
      return datagen


    def train(self, num_epochs = 50, batch_size = 32, data_aug = False, lr_schedule = True):

      assert(num_epochs>0 and batch_size>0)

      # Using Callback API for early stopping
      early_stopper = EarlyStopping(min_delta = 0.0005, patience = 10)
      #Learning rate scheduling can also be done using Callback API
      # lr_reducer = ReduceLROnPlateau(factor=0.01, cooldown=0, patience=5, min_lr=0.5e-6)

      model = self.build() #builds resnet forward model
      if lr_schedule:
        print('Using Learning Rate Scheduling....')
        learning_rate_init = 0.1
        opt = keras.optimizers.SGD(learning_rate=learning_rate_init,momentum = 0.9)

      else:
        print('Using fixed learning rate...')
        learning_rate = 0.001
        opt = keras.optimizers.SGD(learning_rate = learning_rate)

      #Train model
      print('Training model......')

      if data_aug:
        print('Using Data Augmentation....')
        datagen = self.train_data_preprocess()
        datagen.fit(self.X_train)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
                            validation_data=(self.X_test, self.y_test), epochs= num_epochs, steps_per_epoch = self.X_train.shape[0] // batch_size,
                            shuffle = True, callbacks=[early_stopper], verbose = 1)
      else:
        print('Not using Data Augmentation....')
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(self.X_train, self.y_train, batch_size = batch_size, validation_data = (self.X_test, self.y_test),
                            epochs = num_epochs, shuffle = True, callbacks = [early_stopper], verbose = 1)


      print('Training complete.......')

      return model, history

    def evaluate(self, model):
        preds = model.evaluate(self.X_test, self.y_test)

        return preds[1]

def load_dataset():
    """
    Loads cifar10 dataset as numpy array
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize data
    X_train = X_train/255
    X_test = X_test/255

    classes = np.unique(y_train)

    return X_train, y_train, X_test, y_test, classes

def create_one_hot(Y, C):

    """
    Converts labels to one-hot vector
    """

    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def resnet_util(resnet_type):
    resnet_dict = {'resnet_18':{'building_block':'basic_block', 'repetitions':[2,2,2,2],'init_num_filters':64,'conv1_kernel':7},
                   'resnet_34':{'building_block':'basic_block', 'repetitions':[3,4,6,3],'init_num_filters':64,'conv1_kernel':7},
                   'resnet_50':{'building_block':'bottleneck', 'repetitions':[3,4,6,3],'init_num_filters':64,'conv1_kernel':7},
                   'resnet_101':{'building_block':'bottleneck', 'repetitions':[3,4,23,3],'init_num_filters':64,'conv1_kernel':7},
                   'resnet_152':{'building_block':'bottleneck', 'repetitions':[3,8,36,3],'init_num_filters':64,'conv1_kernel':7},
                   'resnet_32':{'building_block':'basic_block', 'repetitions':[5,5,5],'init_num_filters':16,'conv1_kernel':3}}
    if resnet_type not in resnet_dict.keys():
        raise Exception('Not a defined resnet architecture')
    return resnet_dict[resnet_type]

def main():
    X_train, y_train, X_test, y_test, classes = load_dataset()

    num_classes = len(classes)
    y_train = create_one_hot(y_train, num_classes).T
    y_test = create_one_hot(y_test, num_classes).T



    modelname = 'resnet_18' #set resnet architecture name here

    resnet_architecture = resnet_util(modelname) #returns a dictionary of resnet architecture details

    #Unpack parameters specific to an architecture
    building_block = resnet_architecture['building_block']
    repetitions = resnet_architecture['repetitions']
    init_num_filters = resnet_architecture['init_num_filters']
    conv1_f = resnet_architecture['conv1_kernel']

    resnet_network = ResNet(X_train, y_train, X_test, y_test, num_classes, modelname,
                            building_block, repetitions, init_num_filters, conv1_f)

    model, history = resnet_network.train(num_epochs = 100, batch_size = 32, data_aug = True, lr_schedule = True)
    test_accuracy = resnet_network.evaluate(model)

    print('Accuracy on test set: {:.3f} %'.format(test_accuracy*100))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Performance')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper left')
    plt.show()

main()
