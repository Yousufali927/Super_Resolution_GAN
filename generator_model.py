import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model

def pixel_shuffle(scale):
    def _pixel_shuffle(x):
        return tf.nn.depth_to_space(x, scale)
    
    def _output_shape(input_shape):
        batch_size, height, width, channels = input_shape
        height = height * scale if height is not None else None
        width = width * scale if width is not None else None
        channels = channels // (scale ** 2)
        return (batch_size, height, width, channels)
    
    return Lambda(_pixel_shuffle, output_shape=_output_shape)

def build_main_generator(input_shape=(None, None, 3)):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), padding='same')(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)
    
    residual = x
    for _ in range(16):
        res = Conv2D(64, (3, 3), padding='same')(residual)
        res = BatchNormalization(momentum=0.8)(res)
        res = PReLU(shared_axes=[1, 2])(res)
        res = Conv2D(64, (3, 3), padding='same')(res)
        res = BatchNormalization(momentum=0.8)(res)
        residual = Add()([residual, res])
    
    x = Conv2D(64, (3, 3), padding='same')(residual)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, residual])
    
    for _ in range(2):
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = pixel_shuffle(scale=2)(x)
        x = PReLU(shared_axes=[1, 2])(x)
    
    output_layer = Conv2D(3, (9, 9), padding='same')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

# Load the generator model
generator = build_main_generator()
generator.load_weights('bad_generator_weights.h5')
