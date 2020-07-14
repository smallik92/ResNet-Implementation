import numpy as np
from tensorflow import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras import backend as K

def first_conv_block(X, filters, kernel_size, strides,):
    """
    Implements the following
    Conv2D -> BatchNormalization -> ReLu Activation
    """

    X = Conv2D(filters, kernel_size = kernel_size, strides = strides, kernel_initializer = glorot_uniform(seed = 0))(X)
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
        X = Conv2D(filters = filters, kernel_size = (3,3), strides = init_strides, padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    else:
        X = bn_relu_conv(X, filters = filters, kernel_size = (3,3), strides = init_strides)

    X = bn_relu_conv(X, filters=filters, kernel_size = (3,3), strides = (1,1))
    BatchNormalization(axis = 3)(X)
    residual = X
    return shortcut(residual, input)

def bottleneck(X, filters, init_strides = (1,1), is_first_block_of_first_layer = False):
    input = X
    if is_first_block_of_first_layer:
        X = Conv2D(filters = filters, kernel_size = (1,1), strides = init_strides, padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
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
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same', kernel_initializer = glorot_uniform(seed =0 ))(X)
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
                    kernel_initializer = glorot_uniform(seed = 0))(input)
        # short_cut = BatchNormalization(axis = 3)(short_cut)
    else:
        short_cut = input
    return Add()([short_cut, residual])


class ResNet(object):

    def __init__(self, X_train, y_train, X_test, y_test, num_classes, modelname, building_block, repetitions):
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

    def build(self):
        """
        Returns the keras 'Model'
        """

        if len(self.input_shape) != 3:
            raise Exception("Input shape should be a tuple of (n_H, n_W, n_C)")

        X_input = Input(self.input_shape)

        # Stage 1: conv -> BN -> ReLu -> MaxPool
        X = first_conv_block(X_input, filters = 64, kernel_size = (7,7), strides = (2,2))
        X = MaxPooling2D((3,3), strides = (2,2), padding = 'same')(X)

        #Stages 2-5: conv block -> identity blocks -> batch norm -> relu
        num_filters = 64
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
        X = Dense(self.num_classes, activation = 'softmax', kernel_initializer= glorot_uniform(seed = 0))(X)

        print('Creating '+self.modelname+'....')

        model = Model(inputs = X_input, outputs = X, name = 'ResNet')
        return model

    def train(self, num_epochs = 70, batch_size = 32, learning_rate = 0.001):

        assert(num_epochs>0 and batch_size>0 and learning_rate>0)

        model = self.build() #builds resnet forward model
        opt = keras.optimizers.SGD(learning_rate = learning_rate)

         #Train model
        print('Training model......')
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(self.X_train, self.y_train, epochs = num_epochs, batch_size = batch_size)
        return model, history

    def evaluate(self, model):
        preds = model.evaluate(self.X_test, self.y_test)

        return preds[1]
