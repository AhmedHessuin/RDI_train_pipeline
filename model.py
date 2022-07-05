from tensorflow.keras import layers as Layer
import tensorflow as tf
from logger import logger

def model_builder(input_size:int = 150,weight_decay:float=1e-5):
    '''
    build vanilia model using the input size parameter and the weight decay value
    :param input_size: int, contain the input shape width = height
    :param weight_decay: weight decay parameter
    :return:
    '''
    logger.debug("building the model")
    input = Layer.Input((input_size,input_size,3),name="input")
    rescaled=Layer.experimental.preprocessing.Rescaling(1. / 255)(input)
    conv_1=Layer.Conv2D(256,3,3,activation=tf.nn.relu,name="conv1",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(rescaled)
    conv_2=Layer.Conv2D(128,3,3,activation=tf.nn.relu,name="conv2",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(conv_1)
    conv_3=Layer.Conv2D(64,3,3,activation=tf.nn.relu,name="conv3",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(conv_2)
    flatter_1= Layer.Flatten(name="flatter")(conv_3)
    output_layer= Layer.Dense(2,activation=tf.nn.softmax,name="output")(flatter_1)

    logger.debug("layer structure done")
    return tf.keras.Model(inputs=[input],outputs=[output_layer])


def compile_model(model:tf.keras.Model,learning_rate:float=0.001):
    logger.debug("compile the model")
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 0.0001
    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.categorical_accuracy])
    model.summary()
    logger.info(f"compiled done using, optimizer={opt},"
                 f" loss={tf.keras.losses.CategoricalCrossentropy()}"
                 f", metrics={tf.keras.metrics.categorical_accuracy}")
    return model

