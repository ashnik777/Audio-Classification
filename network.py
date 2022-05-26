import tensorflow as tf
from config import config
import tensorflow_addons as tfa
import custom_layers

config_matchbox = config['matchbox']

# This file scan be empty

class MatchBox(tf.keras.Model):
    def __init__(self, feature_layer):
        super(MatchBox, self).__init__()
        # this is used in the callbacks to choose the best model
        self.val_accuracy_moving_mean_max = tf.Variable(initial_value=config_matchbox['val_accuracy_intial_max_value'],
                                                        trainable=False,
                                                        name='val_accuracy_max_value')
        self.val_accuracy_moving_mean = tf.Variable(initial_value=0.0, trainable=False,
                                              name='val_accuracy_moving_mean')
        self.feature_layer = feature_layer
        if config_matchbox['data_normalization']:
            self.data_normalization = custom_layers.DataNormalization()
        self.conv0 = SeparableConvModule(filters=128, kernel_size=11, strides=1)
        self.B = config_matchbox['B']
        self.C = config_matchbox['C']
        list_of_kernel_sizes = config_matchbox['kernel_sizes']
        self.block_list = []
        for i in range(self.B):
            self.block_list.append(Block(filters=self.C, kernel_size=list_of_kernel_sizes[i]))

        self.conv1 = SeparableConvModule(filters=128, kernel_size=29)
        self.conv2 = RegularConvModule(filters=128, kernel_size=1, dilation_rate=2)
        self.decoder_linear = tf.keras.layers.Dense(35, use_bias=True)
        self.global_pool = tfa.layers.AdaptiveAveragePooling1D(output_size=1)
    # @tf.function
    def call(self, inputs, training):  # , training=False):
        inputs = self.feature_layer(inputs)
        # print(inputsz.shape)
        if config_matchbox['data_normalization']:
            inputs = self.data_normalization(inputs)
        # print(inputs.shape)
        x = self.conv0(inputs, training=training)
        # print(x.shape,1)
        for i in range(self.B):
           x = self.block_list[i](x, training=training)  
        # print(x.shape,2)
        x = self.conv1(x, training)
        # print(x.shape,3)
        x = self.conv2(x)
        # print(x.shape)
        x = self.decoder_linear(x)
        # print(x.shape)
        x = self.global_pool(x)
        # print(x.shape)
        x = tf.squeeze(x)
        # print(x.shape)
        x = tf.nn.softmax(x)
        # print(x)
        return x





class SeparableConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, dropout_rate=config_matchbox['dropout'],
                 strides=1):
        super(SeparableConvModule, self).__init__()
        self.conv = tf.keras.layers.SeparableConv1D(filters, kernel_size, dilation_rate=dilation_rate, strides=strides,
                                                    padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, input_tensor, training):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return x


class RegularConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, dropout_rate=config_matchbox['dropout'],
                 strides=1):
        super(RegularConvModule, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, strides=strides,
                                           padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, input_tensor, training):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Block, self).__init__()
        self.R = config_matchbox['R'] - 1
        self.conv_list = []
        for i in range(self.R):
            self.conv_list.append(SeparableConvModule(filters, kernel_size))
        self.last_block_conv = tf.keras.layers.SeparableConv1D(filters, kernel_size,
                                                               padding='same', use_bias=False)
        self.last_block_bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.pointwise_skip = tf.keras.layers.Conv1DTranspose(filters, kernel_size=1,
                                                              use_bias=False)
        self.bn_skip = tf.keras.layers.BatchNormalization(axis=-1)
        self.last_dropout = tf.keras.layers.Dropout(rate=config_matchbox['dropout'])

    def call(self, inputs, training):
        if self.R > 0:
            x = self.conv_list[0](inputs)
            for i in range(1, self.R):
                x = self.conv_list[i](x, training=training)
        else:
            x = inputs

        x = self.last_block_conv(x, training=training)
        x = self.last_block_bn(x, training=training)
        # skip branch
        y = self.pointwise_skip(inputs)
        y = self.bn_skip(y, training=training)
        # connection
        x = tf.nn.relu(y + x)
        x = self.last_dropout(x, training=training)
        return x
