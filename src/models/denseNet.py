from keras.layers import Input, \
                            Convolution2D, \
                            BatchNormalization, \
                            Activation, \
                            Concatenate, \
                            AveragePooling2D, \
                            GlobalAveragePooling2D, \
                            Dense
from keras.models import Model


class DenseNet(object):
    """"
        Implementation based on: https://arxiv.org/pdf/1608.06993.pdf
        DenseNet with both bottleneck and transition layers with Theta < 1
        This is also referred to as DenseNet-BC
    """
    def __init__(self, dense_blocks=3, growth_rate=12, bottleneck=False, depth=40) -> None:
        self.__growth_rate = growth_rate
        self.__dense_blocks = dense_blocks
        self.__bottleneck = bottleneck
        self.__depth = depth
        self.__compression_factor = 0.5
        self.__dense_layers = self.__compute_dense_layers(bottleneck, depth, dense_blocks)

        self.model = self.__create_model()

    def __compute_dense_layers(self, bottleneck, depth, dense_blocks):
        if bottleneck:
            dense_layers = (depth - (dense_blocks - 1)) / dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1)) // dense_blocks

        return [int(dense_layers) for _ in range(dense_blocks)]

    def __create_model(self) -> Model:
        input_layer = Input(shape=(224, 224, 3))
        number_of_filters = self.__growth_rate * 2

        layer = Convolution2D(filters=number_of_filters, kernel_size=7, strides=2, padding='same')(input_layer)
        layer = BatchNormalization()(layer)

        for block in range(self.__dense_blocks):
            layer, number_of_filters = self.__dense_block(layer, self.__dense_layers[block], number_of_filters)

            if block < self.__dense_blocks - 1:
                layer = self.__transition_layer(layer, number_of_filters)
                number_of_filters = int(number_of_filters * self.__compression_factor)

        output_layer = self.__classification_layer(layer)

        return Model(input_layer, output_layer)

    def __classification_layer(self, prev_layer):
        layer = BatchNormalization()(prev_layer)
        layer = Activation('relu')(layer)
        layer = GlobalAveragePooling2D()(layer)

        layer = Dense(units=1000, activation='softmax', name='output')(layer)

        return layer

    def __transition_layer(self, prev_layer, number_of_filters):
        layer = BatchNormalization()(prev_layer)
        layer = Activation('relu')(layer)
        layer = Convolution2D(filters=int(number_of_filters * self.__compression_factor), kernel_size=1, padding='same')(layer)
        layer = AveragePooling2D(pool_size=(2, 2), strides=2)(layer)

        return layer

    def __composite_layer(self, prev_layer, number_of_filters) -> Convolution2D:
        # Bottleneck

        layer = BatchNormalization()(prev_layer)
        layer = Activation('relu')(layer)
        layer = Convolution2D(filters=number_of_filters, kernel_size=3, padding='same')(layer)

        #if there is a dropout its going to be here

        return layer

    def __dense_block(self, layer, dense_layers, number_of_filters):
        layers = [layer]

        for i in range(dense_layers):
            dense_layer = self.__composite_layer(layer, number_of_filters)
            layers.append(dense_layer)
            layer = Concatenate(axis=-1)(layers)
            number_of_filters += self.__growth_rate

        return layer, number_of_filters
