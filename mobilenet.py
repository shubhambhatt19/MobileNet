import tensorflow as tf
from tensorflow.keras import Model
import config as cfg
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D,Conv2D, BatchNormalization, Activation, Dense, Dropout


num_classes = cfg.NO_OF_CLASSES
alpha = cfg.alpha
img_size = cfg.inp_image_size[0]
rho = cfg.rho

def Conv(x, filter, stride, padding='same'):
    filter = int(filter * alpha)
    x = Conv2D(filters=filter, kernel_size=3, strides=stride, padding=padding,input_shape=[int(img_size * rho), int(img_size * rho), 3])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Depthwise_conv(x, stride, padding='same'):
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Pointwise_Conv(x, filter, stride):
    filter = int(filter * alpha)
    x = Conv2D(filters=filter, kernel_size=1, strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x



def mobilenet():

	inputs = Input((224, 224,3))

	x = Conv(inputs, 32, 2)
	x = Depthwise_conv(x, 1)
	x = Pointwise_Conv(x, 64, 1)
	
	x = Depthwise_conv(x, 2)
	x = Pointwise_Conv(x, 128, 1)
	
	x = Depthwise_conv(x, 1)
	x = Pointwise_Conv(x, 128, 1)
	
	x = Depthwise_conv(x, 2)
	x = Pointwise_Conv(x, 256, 1)
	
	x = Depthwise_conv(x, 1)
	x = Pointwise_Conv(x, 256, 1)
	
	x = Depthwise_conv(x, 2)
	x = Pointwise_Conv(x, 512, 1)
	
	for i in range(5):
	    x = Depthwise_conv(x, 1)
	    x = Pointwise_Conv(x, 512, 1)
	
	x = Depthwise_conv(x, 2)
	x = Pointwise_Conv(x, 1024, 1)
	
	x = ZeroPadding2D(padding=4)(x)
	x = Depthwise_conv(x, 2, padding='valid')
	x = Pointwise_Conv(x, 1024, 1) 

	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.2)(x)
	x = Dense(units=num_classes, activation='softmax')(x)
	
	model = Model(inputs = inputs , outputs = x)

	return model

model = mobilenet()

model.summary()