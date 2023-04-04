from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Add
import tensorflow as tf


class NestedUnet:
    def __init__(self, config):
        super()
        self.config = config

    def _conv_block(self, x, n_filters, n_convs, residual=False):
        out = tf.identity(x)
        for i in range(n_convs):
            out = Conv2D(n_filters, kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                         kernel_initializer=self.config['initializer'])(out)
            out = BatchNormalization()(out)
            out = Activation('relu')(out)

        if residual:
            shortcut = Conv2D(n_filters, kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                              kernel_initializer=self.config['initializer'])(x)
            shortcut = BatchNormalization()(shortcut)
            out = Add()([shortcut, out])
        return out

    def _downsample_block(self, x, n_filters, n_convs, residual=False):
        f = self._conv_block(x, n_filters, n_convs, residual)
        p = MaxPooling2D(2)(f)
        p = Dropout(self.config['dropout'])(p)
        return f, p

    def _upsample_block(self, x, conv_features, n_filters, n_convs, residual=False):
        x = Conv2DTranspose(n_filters, 2, 2, padding=self.config['padding'])(x)
        x = concatenate([x, *conv_features])
        x = Dropout(self.config['dropout'])(x)
        x = self._conv_block(x, n_filters, n_convs, residual)
        return x

    def create_model(self):
        inputs1 = Input(shape=(
            self.config['image_size'], self.config['image_size'], self.config['channels']))

        inputs2 = Input(
            shape=(self.config['image_size'] // 2,
                   self.config['image_size'] // 2, 3)
        )

        conv = self._conv_block(inputs2, self.config['filters'], 2, False)

        # encoder
        # 1 - downsample
        conv1_1, pool1 = self._downsample_block(
            inputs1, self.config['filters'], 2, False)
        # 2 - downsample
        conv2_1, pool2 = self._downsample_block(concatenate(
            [conv, pool1]), self.config['filters'] * 2, 2, False)

        conv1_2 = self._upsample_block(
            conv2_1, [conv1_1], self.config['filters'], 2, False)

        # 3 - downsample
        conv3_1, pool3 = self._downsample_block(
            pool2, self.config['filters'] * 4, 2, False)

        conv2_2 = self._upsample_block(
            conv3_1, [conv2_1], self.config['filters'] * 2, 2, False)
        conv1_3 = self._upsample_block(
            conv2_2, [conv1_1, conv1_2], self.config['filters'], 2, False)

        # 4 - downsample
        conv4_1, pool4 = self._downsample_block(
            pool3, self.config['filters'] * 8, 2, False)

        conv3_2 = self._upsample_block(
            conv4_1, [conv3_1], self.config['filters'] * 4, 2, False)
        conv2_3 = self._upsample_block(
            conv3_2, [conv2_1, conv2_2], self.config['filters'] * 2, 2, False)
        conv1_4 = self._upsample_block(
            conv2_3, [conv1_1, conv1_2, conv1_3], self.config['filters'], 2, False)

        # 5 - bottleneck
        conv5_1 = self._conv_block(
            pool4, self.config['filters'] * 16, 2, False)

        conv4_2 = self._upsample_block(
            conv5_1, [conv4_1], self.config['filters'] * 8, 2, False)
        conv3_3 = self._upsample_block(
            conv4_2, [conv3_1, conv3_2], self.config['filters'] * 4, 2, False)
        conv2_4 = self._upsample_block(
            conv3_3, [conv2_1, conv2_2, conv2_3], self.config['filters'] * 2, 2, False)
        conv1_5 = self._upsample_block(
            conv2_4, [conv1_1, conv1_2, conv1_3, conv1_4], self.config['filters'], 2, False)

        # outputs
        outputs = Conv2D(
            len(self.config['classes']),
            1,
            padding=self.config['padding'],
            activation=self.config['activation']
        )(conv1_5)

        model = Model([inputs1, inputs2], outputs, name="NestedUNet")

        return model
