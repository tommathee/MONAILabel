from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, concatenate, Conv2D, AveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout


class BasicConv(Model):
    def __init__(
            self,
            n_filters,
            kernel_size,
            padding,
            initializer,
            dilation_rate=1,
            batch_norm=False,
            dropout=None,
            residual=False,
            activation=None
    ):
        super(BasicConv, self).__init__()

        self.conv = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer,
            dilation_rate=dilation_rate,
            activation=activation
        )
        self.batch_norm = BatchNormalization() if batch_norm else None
        self.relu = Activation('relu')
        self.dropout = Dropout(dropout) if dropout else None
        self.residual = residual

    def call(self, input, training=False):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)

        if self.residual:
            x += input

        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)

        return x


class DeepLabV3Plus:
    def __init__(self, config):
        super(DeepLabV3Plus, self)
        self.config = config

    def dilated_spatial_pyramid_pooling(self, inputs):
        dims = inputs.shape

        x = AveragePooling2D(
            pool_size=(dims[-3], dims[-2])
        )(inputs)

        x = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer']
        )(x)

        out_pool = UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        out_1 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=1
        )(inputs)
        out_6 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=6
        )(inputs)
        out_12 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=12
        )(inputs)
        out_18 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=18
        )(inputs)

        x = concatenate([out_pool, out_1, out_6, out_12, out_18], axis=-1)
        output = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
        )(x)

        return output

    def create_model(self):
        input = Input(
            shape=(self.config['image_size'], self.config['image_size'], self.config['channels'])
        )
        resnet50 = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.dilated_spatial_pyramid_pooling(x)

        input1 = UpSampling2D(
            size=(self.config['image_size'] // 4 // x.shape[1], self.config['image_size'] // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)

        input2 = resnet50.get_layer("conv2_block3_2_relu").output
        input2 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
        )(input2)

        x = concatenate([input1, input2], axis=-1)
        x = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
        )(x)
        x = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
        )(x)
        x = UpSampling2D(
            size=(self.config['image_size'] // x.shape[1], self.config['image_size'] // x.shape[2]),
            interpolation="bilinear",
        )(x)

        output = Conv2D(
            len(self.config['final_classes']),
            kernel_size=1,
            padding=self.config['padding'],
            kernel_initializer=self.config['initializer'],
            activation=self.config['activation']
        )(x)

        model = Model(input, output, name="DeepLabV3Plus")
        return model
