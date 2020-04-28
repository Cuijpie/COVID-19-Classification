from keras.layers import Input, \
                            Convolution2D, \
                            MaxPooling2D, \
                            BatchNormalization, \
                            AveragePooling2D, \
                            Concatenate, \
                            Flatten, \
                            Dense, \
                            Dropout
from keras.models import Model


class GoogleNet(object):
    def __init__(self) -> None:
        self.__inception_3a = [64, (96, 128), (16, 32), 32]
        self.__inception_3b = [128, (128, 192), (32, 96), 64]
        self.__inception_4a = [192, (96, 208), (16, 48), 64]
        self.__inception_4b = [160, (112, 224), (24, 64), 64]
        self.__inception_4c = [128, (128, 256), (24, 64), 64]
        self.__inception_4d = [112, (144, 288), (32, 128), 64]
        self.__inception_4e = [256, (160, 320), (32, 128), 128]
        self.__inception_5a = [256, (160, 320), (32, 128), 128]
        self.__inception_5b = [384, (192, 384), (48, 128), 128]

        self.__create_model()

    def __create_model(self) -> None:
        input_layer = Input(shape=(224, 224, 3))

        layer = Convolution2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu')(input_layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
        layer = BatchNormalization()(layer)

        layer = Convolution2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(layer)
        layer = Convolution2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu')(layer)
        layer = BatchNormalization()(layer)

        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
        layer = self.__inception_layer(layer, self.__inception_3a)
        layer = self.__inception_layer(layer, self.__inception_3b)

        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
        layer = self.__inception_layer(layer, self.__inception_4a)

        softmax_0 = self.__softmax_layer(layer, "softmax_0")

        layer = self.__inception_layer(layer, self.__inception_4b)
        layer = self.__inception_layer(layer, self.__inception_4c)
        layer = self.__inception_layer(layer, self.__inception_4d)

        softmax_1 = self.__softmax_layer(layer, "softmax_1")
        layer = self.__inception_layer(layer, self.__inception_4e)

        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
        layer = self.__inception_layer(layer, self.__inception_5a)
        layer = self.__inception_layer(layer, self.__inception_5b)
        layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(layer)
        layer = Flatten()(layer)
        layer = Dropout(0.4)(layer)
        softmax_2 = Dense(units=1000, activation='softmax', name="softmax_2")(layer)

        self.model = Model(input_layer, [softmax_0, softmax_1, softmax_2])

    def __inception_layer(self, prev_layer, filters) -> Concatenate:
        conv1x1 = Convolution2D(filters[0], kernel_size=1, strides=1, padding='same', activation='relu')(prev_layer)

        conv1x1_conv3x3 = Convolution2D(filters[1][0], kernel_size=1, strides=1, padding='same', activation='relu')(prev_layer)
        conv1x1_conv3x3 = Convolution2D(filters[1][0], kernel_size=3, strides=1, padding='same', activation='relu')(conv1x1_conv3x3)

        conv1x1_conv5x5 = Convolution2D(filters[2][0], kernel_size=1, strides=1, padding='same', activation='relu')(prev_layer)
        conv1x1_conv5x5 = Convolution2D(filters[2][1], kernel_size=5, strides=1, padding='same', activation='relu')(conv1x1_conv5x5)

        max3x3_conv1x1 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(prev_layer)
        max3x3_conv1x1 = Convolution2D(filters[3], kernel_size=1, strides=1, padding='same', activation='relu')(max3x3_conv1x1)

        return Concatenate(axis=-1)([conv1x1, conv1x1_conv3x3, conv1x1_conv5x5, max3x3_conv1x1])

    def __softmax_layer(self, prev_layer, name) -> Dense:
        layer = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(prev_layer)
        layer = Convolution2D(filters=128, kernel_size=1, strides=1, padding='same', activation='relu')(layer)
        layer = Flatten()(layer)
        layer = Dense(units=1024, activation='relu')(layer)
        layer = Dropout(0.7)(layer)
        layer = Dense(units=1000, activation='softmax', name=name)(layer)
        return layer



