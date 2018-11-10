from typing import Dict, Optional

from keras import Model, Input
from keras.engine import Layer
# from keras.layers import Conv2D, Dropout, MaxPool2D, concatenate, Conv2DTranspose, ThresholdedReLU
from keras.layers import Conv3D, Dropout, MaxPool3D, concatenate, Conv3DTranspose, ThresholdedReLU
import keras.backend as K


def custom_unet(config) -> Model:
    """
    Create a custom U-Net Keras model based on model settings defined in the configuration section.

    Due to the non-sequential nature of a U-Net, each layer is given a custom name as well as added to a dictionary
    that is used to to connect the encoding and decoding blocks together.

    Both the encoding and decoding blocks contain 2D Convolution layers with a certain number of filters associated
    with them.  The number of filters is determined by the depth of the block.  The first encoding block (and final
    decoding block) contain INITIAL_FILTERS number of filters.  Each encoding block after the first has twice as
    many filters as the previous block.  For decoding blocks, the pattern runs in reverse until the final decoding
    block has only INITIAL_FILTERS number of filters.

    Layer Naming Conventions:
        All layers are given custom names so that they can be easily identified in the Keras model.  The layer names
        are also used in the layers dictionary so that proper encoder-decoder connections can be made dynamically.
        The convention used separates identifying parts with underscores.
        Each hidden layer starts with a four letter prefix that identifies what type of layer it is.
            CONV - 2D Convolution
            TRAN - 2D Transposed Convolution
            POOL - 2D Max Pooling
            DROP - Dropout
        Following the prefix is a three letter identifier to indicate what block the layer is in.
            ENC - Encoding Block
            DEC - Decoding Block
            BTM - Bottom Block
        Next is the letter D followed by a number.  The D indicates that the following number refers to the depth
        that the layer is in.  The number is between 1 and DEPTH (defined in the configuration section).
        In some cases there will be a letter S followed by a number.  This indicates what segmentation levels the
        layer works on.  This number is between 1 and SEGMENTATION_LEVELS, which is defined in the configuration
        section.
        Finally, there may be the letter L at the end of the name.  If any of the blocks contain a LeakyReLU
        activation layer, they will be denoted by the letter L as a suffix.  In the case that this layer is used,
        the previous layer is given no activation function (linear activation) so that the LeakyReLU layer does
        not receive any altered output values.

    This function is divided into several parts that each generate a section, or block, of the U-Net model.  These
    blocks are:
        Encoding - The descending part of the UNet that takes the inputs and runs it through convolution filters.
                   Each encoding block contains at least one 2D Convolution layer and is determined by the value
                   assigned to SEGMENTATION_LEVELS.  If DROPOUT_RATE is not set to None, then there will be a
                   Dropout layer after the first Convolution layer of each encoding block.  If SEGMENTATION_LEVELS
                   is set to a number grater than 1, then 1 - SEGMENTATION_LEVELS will be created after the Dropout
                   layer.  At the end of each encoding block is a 2D Max Pooling layer that has a size defined by
                   POOL_SIZE in the configuration section.

        Decoding - The ascending part of the UNet that processes the convoluted data through deconvolution and
                   concatenates that data with data from the encoding blocks.  The decoding blocks contain a
                   concatenation layers and as many 2D Convolution layers following the concatenation layer as there
                   are SEGMENTATION_LEVELS.

          Bottom - The middle of the "U" in the UNet architecture is slightly different than the Encoding blocks,
                   thus it is given it's own block function.  This block is similar to the Encoding blocks, except
                   it does not contain any Max Pooling layers.  It is simply a combination of 2D Convolution layer(s)
                   and an optional Dropout layer.  The configuration section has special settings for this block in
                   case it is determined that the Convolution layers need to be tweaked independently of the Encoding
                   blocks settings.
    """
    # Let keras know the order of the tuples.  It is not recommended to change this, as this file is hard-coded in many
    # places to handle numpy arrays where the channel dimension is last.
    K.set_image_data_format('channels_last')

    # Keep track of the layers so the encoding modules can be connected to the decoding modules
    layers: Dict[str, Layer] = {}
    prev_layer: Optional[Layer] = None

    def record_layer(l_name: str, layer: Layer) -> None:
        """
        Prints out information about the layer currently being generated as well as adding it to the layer
        dictionary.  This function will also assign the layer passed into it as the prev_layer (previous layer)
        so that the following layer can connect directly to it if needed.

        :rtype: None
        :param l_name: The custom layer name, this is used to get the layer from the layers dictionary.
        :param layer: The keras layer object created.
        """
        global prev_layer
        print("Adding layer {} to model".format(l_name))
        layers[l_name] = layer
        prev_layer = layer

    def get_activation(defined_activation):
        if type(defined_activation) == str:
            return defined_activation
        elif isinstance(defined_activation(), Layer):
            return 'linear'
        else:
            raise TypeError("Bad activation type {}".format(defined_activation))

    def create_encoding_block(curr_depth: int) -> None:
        """
        Creates an encoding block for the custom UNet.

        See documentation for custom_unet for information about Encoding blocks.

        :rtype: None
        :param curr_depth: The current depth level of the UNet.  This is used to determine the number of filters
        for convolution layers as well as the layer name.
        """
        global prev_layer
        n_filters = config['initial_filters'] * (2 ** curr_depth)  # Calculate the number of filters for this block

        activation = get_activation(config['encoder_activation'])

        l_name = "CONV_ENC_D{}_S{}".format(curr_depth, 1)
        conv = Conv3D(filters=n_filters, kernel_size=config['encoder_kernel_size'], activation=activation,
                      padding=config['padding_mode'], kernel_initializer=config['encoder_kernel_initializer'],
                      kernel_regularizer=config['encoder_kernel_regularizer'], name=l_name)(prev_layer)
        record_layer(l_name, conv)

        if activation == 'linear':
            l_name = l_name + "_A"
            conv = config['encoder_activation'](name=l_name)(prev_layer)
            record_layer(l_name, conv)

        if config['dropout_rate'] is not None:
            l_name = "DROP_D{}_S{}".format(curr_depth, 0)
            drop = Dropout(config['dropout_rate'], name=l_name)(prev_layer)
            record_layer(l_name, drop)

        for seg_level in range(1, config['segmentation_levels']):
            l_name = "CONV_ENC_D{}_S{}".format(curr_depth, seg_level + 1)
            conv = Conv3D(filters=n_filters, kernel_size=config['encoder_kernel_size'], activation=activation,
                          padding=config['padding_mode'], kernel_initializer=config['encoder_kernel_initializer'],
                          kernel_regularizer=config['encoder_kernel_regularizer'], name=l_name)(prev_layer)
            record_layer(l_name, conv)

            if activation == 'linear':
                l_name = l_name + "_A"
                conv = config['encoder_activation'](name=l_name)(prev_layer)
                record_layer(l_name, conv)

            if config['dropout_rate'] is not None:
                l_name = "DROP_D{}_S{}".format(curr_depth, seg_level)
                drop = Dropout(config['dropout_rate'], name=l_name)(prev_layer)
                record_layer(l_name, drop)

        l_name = "POOL_D{}".format(curr_depth)
        pool = MaxPool3D(pool_size=config['pooling_size'], name=l_name)(prev_layer)
        record_layer(l_name, pool)

    def create_decoding_block(curr_depth: int) -> None:
        """
        Creates a decoding block for the custom UNet.

        See documentation for custom_unet for information about Decoding blocks.

        :rtype: None
        :param curr_depth: The current depth level of the UNet.  This is used to determine the number of filters
        for convolution layers, what layer the concatenation layer needs to connect to, as well as the layer name.
        """
        global prev_layer
        n_filters = config['initial_filters'] * (2 ** curr_depth)  # Calculate the number of filters for this block

        activation = get_activation(config['decoder_activation'])

        # Ensure that the concatenation layer is connecting to the correct encoding layer.  We need to tell it to
        # look at the non-leaky layer    if we do not use a Leaky ReLU activation layer in the encoder blocks.
        if type(config['decoder_activation']) == str:
            connected_layer = "CONV_ENC_D{}_S{}"
        elif isinstance(config['encoder_activation'](), Layer):
            connected_layer = "CONV_ENC_D{}_S{}_A"
        else:
            raise TypeError("Encoder activation must be a string alias of a keras layer or a keras Layer object")

        l_name = "CONC_DEC_D{}".format(curr_depth)
        merge = concatenate([Conv3DTranspose(filters=n_filters, kernel_size=config['decoder_kernel_size'],
                                             strides=config['decoder_strides'], padding=config['padding_mode'],
                                             name="TRAN_DEC_D{}".format(curr_depth))(prev_layer),
                             layers[connected_layer.format(curr_depth, config['segmentation_levels'])]], axis=4,
                            name=l_name)
        record_layer(l_name, merge)

        for s in range(config['segmentation_levels']):
            l_name = "CONV_DEC_D{}_S{}".format(curr_depth, s + 1)
            conv = Conv3D(filters=n_filters, kernel_size=config['decoder_kernel_size'], activation=activation,
                          padding=config['padding_mode'], kernel_initializer=config['decoder_kernel_initializer'],
                          kernel_regularizer=None, name=l_name)(prev_layer)
            record_layer(l_name, conv)

            if activation == 'linear':
                l_name = l_name + "_A"
                conv = config['decoder_activation'](name=l_name)(prev_layer)
                record_layer(l_name, conv)

    def create_bottom_block() -> None:
        """
        Creates a decoding block for the custom UNet.

        See documentation for custom_unet for information about Decoding blocks.

        :rtype: None
        """
        global prev_layer
        n_filters = config['initial_filters'] * (2 ** config['depth'])

        activation = get_activation(config['bottom_activation'])

        for seg_level in range(config['segmentation_levels']):
            l_name = "CONV_BTM_D{}_S{}".format(config['depth'], seg_level + 1)
            conv = Conv3D(filters=n_filters, kernel_size=config['bottom_kernel_size'], activation=activation,
                          padding=config['padding_mode'], kernel_initializer=config['bottom_kernel_initializer'],
                          kernel_regularizer=config['bottom_kernel_regularizer'], name=l_name)(prev_layer)
            record_layer(l_name, conv)

            if activation == 'linear':
                l_name = l_name + "_A"
                conv = config['bottom_activation'](name=l_name)(prev_layer)
                record_layer(l_name, conv)

        if config['dropout_rate'] is not None:
            l_name = "DROP_BTM_D{}".format(config['depth'])
            drop = Dropout(config['dropout_rate'], name=l_name)(prev_layer)
            record_layer(l_name, drop)

    def create_final_block() -> None:
        global prev_layer

        activation = get_activation(config['final_activation'])

        l_name = "CONV_FINAL"
        conv = Conv3D(filters=1, kernel_size=config['final_kernel_size'], activation=activation,
                      padding=config['padding_mode'], kernel_initializer=config['final_kernel_initializer'],
                      kernel_regularizer=None, name=l_name)(prev_layer)
        record_layer(l_name, conv)

        if activation == 'linear':
            l_name = l_name + "_A"
            conv = config['final_activation'](name=l_name)(prev_layer)
            record_layer(l_name, conv)

        if config['labels'] > 1:
            # TODO: Allow for multiple output labels
            pass

        if config['theta_cutoff'] is not None:
            l_name = "OUT"
            out = ThresholdedReLU(theta=config['theta_cutoff'], name=l_name)(prev_layer)
            record_layer(l_name, out)

    # Create input layer
    name = "INPUT"
    inputs = Input(shape=(config['input_size'][0], config['input_size'][1], config['input_size'][2], config['input_channels']), name=name)
    record_layer(name, inputs)

    # Create descending encoding layers
    for dep in range(1, config['depth']):
        create_encoding_block(dep)

    # Create bottom layers of model
    create_bottom_block()

    # Create ascending decoding layers
    for dep in range(config['depth']-1 , 0, -1):
        create_decoding_block(dep)

    # Create final layer of model
    create_final_block()

    if config['theta_cutoff'] is None:
        if type(config['final_activation']) == str:
            my_model = Model(inputs=layers['INPUT'], outputs=layers['CONV_FINAL'])
        elif isinstance(config['final_activation'](), Layer):
            my_model = Model(input=layers['INPUT'], output=layers['CONV_FINAL_A'])
        else:
            raise TypeError("Final activation must be a string alias of a keras layer or a keras Layer object")
    else:
        my_model = Model(inputs=layers['INPUT'], outputs=layers['OUT'])

    my_model.compile(optimizer=config['optimizer'], loss=config['loss_function'], metrics=config['metrics'])
    my_model.summary()

    return my_model
