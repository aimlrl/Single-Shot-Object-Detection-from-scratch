#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import resnet
from keras.layers import Dense,Conv2D
from keras.layers import BatchNormalization,Activation
from keras.layers import AveragePooling2D,Input
from keras.layers import Add
from keras.regularizers import l2
from keras.models import Model


# In[2]:


def convolutional_layer(num_filters=8,ksize=3,padding="same",stride=1):
    
    """This function creates a convolutional layer for use in the residual blocks of Resnet.

    Parameters:
                  num_filters (int) : Number of filters to use in Convolutional layer
                  ksize (int) : Kernel size of the filters to be used in Convolutional layer
                  padding (string) : Whether to pad or not

    Returning: 
                  A convolutional layer
    """

    conv_layer = Conv2D(filters=num_filters,kernel_size=ksize,padding=padding,kernel_initializer="he_normal",kernel_regularizer=l2(0.001),strides=stride)
    
    return conv_layer


# In[3]:


def residual_layer(layer_input,num_filters=8,ksize=3,activation="relu",conv_first=True,padding="same",stride=1):

    """This function will create a residual layer of a residual block used in Resnet

    Parameters:
                  layer_input (tensor) : Input tensor from the output of the previous layer
                  num_filters (int) : Number of filters to be used in convolutional layer inside residual 
                                      layer
                  kernel_size (int, usually odd) : Number of Rows as well as columns in the Kernel of 
                                                   filters used in convolutional layers
                  activation (string) : Specifies which activation function layer should be used after 
                                        applying batch normalization
                  conv_first (bool) : Whether to use Conv2D-BN-Relu (if conv_first is True) or 
                                      BN-Relu-Conv2D

    Returning:
                  layer_output (tensor) : Output of the layer as a tensor
    """

    conv_layer = convolutional_layer(num_filters,ksize,padding=padding,stride=stride)

    if conv_first == True:

        layer_output = conv_layer(layer_input)
        layer_output = BatchNormalization()(layer_output)
        layer_output = Activation(activation)(layer_output)

    else:
        layer_output = BatchNormalization()(layer_input)
        layer_output = Activation(activation)(layer_output)
        layer_output = conv_layer(layer_output)
        
    return layer_output


# In[4]:


def residual_block(residual_block_input,stack_number,res_block_number,num_residual_layers=2,num_filters=8,padding="same"):

    """This function creates a residual block to be used in Resnet.

    Parameters:
                  residual_block_input (tensor) : Input to the residual block
                  num_residual_layers (int) : Number of residual layers in a residual block

    Returning:
                  A residual block having residual layers
    """

    if stack_number > 0 and res_block_number == 0:
        stride = 2
    else:
        stride = 1

    residual_layer_output = residual_layer(residual_block_input,num_filters=num_filters,stride=stride)
    residual_layer_output = residual_layer(residual_layer_output,num_filters=num_filters,activation=None)

    if stack_number > 0 and res_block_number == 0:
        residual_block_input = residual_layer(residual_block_input,num_filters=num_filters,ksize=1,stride=stride,activation=None) 

    residual_layer_output = Add()([residual_block_input,residual_layer_output])
    residual_block_output = Activation("relu")(residual_layer_output)
    
    return residual_block_output


# In[7]:


def custom_resnet_convolutional_base(input_shape,num_extra_layers=2,num_residual_blocks=3,num_stacks=4,num_filters=8):

    """This function creates a Resnet having specific number of Residual blocks.

    Parameters:
                  input_shape (tuple) : Shape of the input tensor which will be feed into the Resnet
                  num_extra_layers (int) : Number of extra layers which you want to add into Resnet 
                                           depending upon the number of scales at which the detection
                                           needs to be performed
                  num_residual_blocks (int) : Number of residual blocks which need to be added into the 
                                              convolutional base of Resnet
                  num_stacks (int) : Number of stacks of residual blocks

    Returning: 
                  A Resnet model based on functional API
    """

    input_image = Input(shape=input_shape)
    transformed_image = convolutional_layer(padding="valid",ksize=1,stride=2)(input_image)
    transformed_image = BatchNormalization()(transformed_image)
    residual_block_output = Activation("relu")(transformed_image)

    for i in range(num_stacks):
        for j in range(num_residual_blocks):
            residual_block_output = residual_block(residual_block_output,num_filters=num_filters,
                                                   stack_number=i,res_block_number=j)
        num_filters = num_filters * 2


    resnet_conv_base_output = AveragePooling2D(pool_size=2)(residual_block_output)
    resnet_outputs = [resnet_conv_base_output]
    previous_layer_output = resnet_conv_base_output
    
    num_filters = num_filters/2

    for _ in range(num_extra_layers):
        current_layer_output = residual_layer(previous_layer_output,num_filters=num_filters,
                                              activation="elu",padding="valid",stride=2)
        resnet_outputs.append(current_layer_output)
        previous_layer_output = current_layer_output
        num_filters = num_filters * 2

    resnet = Model(inputs=input_image,outputs=resnet_outputs,name="Custom")
    
    return resnet


# In[6]:


def prebuilt_resnet_convolutional_base(input_shape,include_pretrained_weights,num_extra_layers=2):

    """This function creates a convolutional base of prebuilt residual network from keras.applications.

    Parameters:
                  input_shape (tuple) : Shape of the input which is to be fed into the prebuilt Resnet
                  include_pretrained_weights (bool) : If true then the convolutional base of this network 
                                                      will be instantiated with pretrained weights from 
                                                      Imagenet
                  num_extra_layers (int) : Number of extra layers which you want to add into Resnet 
                                           depending upon the number of scales at which the detection
                                           needs to be performed 
                                           
    Returning: 
                  A prebuilt Resnet convolutional base model based on functional API.
    """

    input_image = Input(shape=input_shape)

    prebuilt_resnet_conv_base = resnet.ResNet50(include_top=False,input_shape=input_shape,
                                                input_tensor=input_image)

    resnet_outputs = [prebuilt_resnet_conv_base.output]
    previous_layer_output = prebuilt_resnet_conv_base.output

    num_filters = 2048

    for _ in range(num_extra_layers):
        current_layer_output = residual_layer(previous_layer_output,num_filters=num_filters,
                                              activation="elu",padding="valid",stride=2)
        resnet_outputs.append(current_layer_output)
        previous_layer_output = current_layer_output
        num_filters = num_filters * 2
        
    prebuilt_resnet = Model(inputs=input_image,outputs=resnet_outputs,name="Prebuilt")
    
    return prebuilt_resnet


# In[8]:


def create_resnet(input_shape,is_prebuilt,is_pretrained=False,num_extra_layers=2):

    """This function creates a resnet based on the user preferences.

    Parameters:
                  input_shape (tuple) : Shape of the input which is to be fed into the user created Resnet
                  num_extra_layers (int) : Number of extra layers based on user preference depending upon 
                                           at how many different scales a user wants to build a residual 
                                           network to perform detection
                  is_prebuilt (bool) : Whether user wants to select a prebuilt version of a resnet
                  is_pretrained (bool) : Whether user wants to select a pretrained version of a prebuilt 
                                         resnet
                    
    Returning:
              A resnet model based function API and based on user preferences. 
    """

    if is_prebuilt == True:
        user_resnet = prebuilt_resnet_convolutional_base(input_shape=input_shape,
                                                         include_pretrained_weights=is_pretrained)
    else:
        user_resnet = custom_resnet_convolutional_base(input_shape=input_shape)

    return user_resnet

