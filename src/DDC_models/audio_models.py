from keras import backend as K
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Flatten, Conv2D
from keras.layers import LeakyReLU, SpatialDropout2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.applications.xception import Xception, preprocess_input
from keras.applications.nasnet import NASNetMobile

from DDC_models.efficientNet_model import EfficientNetB3



# This is a VGG-style network that I made by 'dumbing down' @keunwoochoi's compact_cnn code
# I have not attempted much optimization, however it *is* fairly understandable
def panotti_model(X_shape=(128,173,1), nb_classes=2, nb_layers=4):
    '''
    Based on
    https://github.com/drscotthawley/panotti/blob/master/panotti/models.py
    Author: Scott Hawley
    Where we'll put various NN models.
    MyCNN:  This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
      and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    # Inputs:
    #    X_shape = [ # spectrograms per batch, # spectrogram freq bins, # spectrogram time bins ]
    #    nb_classes = number of output n_classes
    #    nb_layers = number of conv-pooling sets in the CNN
    '''

    K.set_image_data_format('channels_last')                   # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last

    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5    # conv. layer dropout
    dl_dropout = 0.6    # dense layer dropout

    input_shape = X_shape
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, padding='same', input_shape=input_shape, name="Input"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Activation('relu'))        # Leave this relu & BN here.  ELU is not good here (my experience)
    model.add(BatchNormalization(axis=-1))  # axis=1 for 'channels_first'; but tensorflow preferse channels_last (axis=-1)

    for layer in range(nb_layers-1):   # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size, padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Activation('elu'))
        model.add(Dropout(cl_dropout))
        #model.add(BatchNormalization(axis=-1))  # ELU authors reccommend no BatchNorm. I confirm.

    model.add(Flatten())
    model.add(Dense(128))            # 128 is 'arbitrary' for now
    #model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
    model.add(Activation('elu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax",name="Output"))

    return model

def urban_audio_classifier_model(X_shape=(128,173,1), num_labels=2, spatial_dropout_rate_1=0.07, spatial_dropout_rate_2=0.14, l2_rate=0.0005):
    '''
    Based on 
    https://github.com/GorillaBus/urban-audio-classifier/blob/master/4-cnn-model-mel_spec.ipynb
    '''

    # Create a secquential object
    model = Sequential()

    # Conv 1
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate), 
                     input_shape=X_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    # Max Pooling #1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_2))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3,3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    
    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())
    
    # Softmax output
    model.add(Dense(num_labels, activation='softmax'))
    
    return model

def Xception_model(input_shape=(128, 173, 3), num_classes=2):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)

    return model

def mobilenet_model(input_shape=(128, 173, 3), num_classes=2):
    base_model = NASNetMobile(include_top=False, weights='imagenet', input_shape=input_shape)

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='elu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)

    return model

def EfficientNet_model(shape=(128, 173, 3)):

    image_input = Input(shape=shape)
    
    cnn = EfficientNetB3(include_top=False, 
                         weights='imagenet',
                         pooling='avg',
                         backend = keras.backend, 
                         layers = keras.layers, 
                         models = keras.models, 
                         utils = keras.utils)(image_input)

    d1 = Dense(512)(cnn)
    d1 = Dropout(0.5)(d1)
  
    output = Dense(2, activation='softmax')(d1)
    
    model = Model(inputs=image_input, outputs=output)

    return model


def audio_model(name):
    if name == 'panotti':
        return panotti_model()
    elif name == 'urban_audio_classifier':
        return urban_audio_classifier_model()
    elif name == 'Xception':
        return Xception_model()
    elif name == 'mobilenet':
        return mobilenet_model()
    elif name == 'EfficientNet':
        return EfficientNet_model()

