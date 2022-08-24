from tensorflow.python.keras.layers.convolutional import Conv3D, ZeroPadding3D, Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from tensorflow.python.keras.layers.wrappers import Bidirectional, TimeDistributed
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras.layers import BatchNormalization, Input, GlobalAveragePooling2D, Reshape, multiply
from tensorflow.python.keras.models import Model
from lipnet.core.layers import CTC
from tensorflow.python.keras import backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.n_res_blocks = 10
        self.build()

    def adaptive_global_average_pool_3d(self, x):
        """
        In the paper, using gap which output size is 1, so i just gap func :)
        :param x: 5d-tensor, (batch_size, frame, height, width, channel)
        :return: 5d-tensor, (batch_size, frame, 1, 1, channel)
        """
        f = x.get_shape()[1]
        c = x.get_shape()[-1]
        return tf.reshape(tf.reduce_mean(x, axis=[2, 3]), (-1, f, 1, 1, c))


    def channel_attention(self, x, f, name):
        skip_conn = tf.identity(x, name=name + '_identity')
        x = self.adaptive_global_average_pool_3d(x)


        x = Conv3D(filters=32, kernel_size=1, name=name + "_conv3d-1")(x)
        x = Activation('relu')(x)

        x = Conv3D(filters=32, kernel_size=1, name=name + "_conv3d-2")(x)

        x = tf.nn.sigmoid(x)

        return tf.multiply(skip_conn, x)

    def residual_channel_attention_block(self, x, use_bn, name):
        skip_conn = tf.identity(x, name=name + '_identity')

        x = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=1, padding="SAME", name=name + "_conv3d-1")(x)
        x = BatchNormalization()(x) if use_bn else x
        x = Activation('relu')(x)

        x = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=1, padding="SAME", name=name + "_conv3d-2")(x)
        x = BatchNormalization()(x) if use_bn else x

        x = self.channel_attention(x, 32, name= name + "/CA")

        return x + skip_conn


    def residual_group(self, x, use_bn, name):

        skip_conn = tf.identity(x, name= name + "_identity")

        for i in range(self.n_res_blocks):
            x = self.residual_channel_attention_block(x, use_bn, name= name + "/RCAB_" + str(i))

        x = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=1, padding="SAME", name= name + "_rg-conv-1")(x)


        return x + skip_conn


    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero0 = ZeroPadding3D(padding=(1, 2, 2), name='zero0')(self.input_data)
        self.conv0 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv0')(self.zero0)

        self.RG0 = self.residual_group(self.conv0, use_bn=False, name = "RG0")

        self.conv1 = Conv3D(32, (1, 3, 3), strides=1, padding='SAME', kernel_initializer='he_normal', name='conv1')(self.RG0)
        self.batc1 = BatchNormalization(name='batc1')(self.conv1)
        self.actv1 = Activation('relu', name='actv1')(self.batc1)
        self.drop1 = SpatialDropout3D(0.5)(self.actv1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.drop1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.maxp1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.batc2 = BatchNormalization(name='batc2')(self.conv2)
        self.actv2 = Activation('relu', name='actv2')(self.batc2)
        self.drop2 = SpatialDropout3D(0.5)(self.actv2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.drop2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.maxp2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.batc3 = BatchNormalization(name='batc3')(self.conv3)
        self.actv3 = Activation('relu', name='actv3')(self.batc3)
        self.drop3 = SpatialDropout3D(0.5)(self.actv3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.drop3)

        self.resh1 = TimeDistributed(Flatten())(self.maxp3)

        self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh1)
        self.gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc',[self.y_pred, self.labels, self.input_length, self.label_length])
        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        with open('RG1_RCAB10.txt', 'w') as fh:
            Model(inputs=self.input_data, outputs=self.y_pred).summary(print_fn=lambda x: fh.write(x + '\n'))

    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred])
