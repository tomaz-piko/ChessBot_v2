import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Conv2D,
    Flatten,
    ReLU,
    Add,
)
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.optimizers import SGD
import keras.backend as K

from configs import modelConfig
from configs import selfplayConfig as config

K.set_image_data_format("channels_last")

R = config["history_repetition_planes"]
T = config["history_steps"]
NUM_PLANES = (6 * 2 + R) * T + 5 

CONV_FILTERS = modelConfig["conv_filters"]
NUM_RESIDUAL_BLOCKS = modelConfig["residual_blocks"]
CONV_KERNEL_INITIALIZER = config.get(modelConfig["conv_kernel_initializer"], None)
USE_BIAS_ON_OUTPUTS = modelConfig["use_bias_on_output"]
VALUE_HEAD_FILTERS = modelConfig["value_head_filters"]
VALUE_HEAD_DENSE = modelConfig["value_head_dense"]
POLICY_HEAD_FILTERS = modelConfig["policy_head_filters"]
POLICY_HEAD_DENSE = modelConfig["policy_head_dense"]
POLICY_HEAD_LOSS_WEIGHT = modelConfig["policy_head_loss_weight"]
VALUE_HEAD_LOSS_WEIGHT = modelConfig["value_head_loss_weight"]
L2_REG = modelConfig["l2_regularization"]
SGD_MOMENTUM = modelConfig["sgd_momentum"]
SGD_NESTEROV = modelConfig["sgd_nesterov"]

@tf.function
def predict_fn(trt_func, images):
    predictions = trt_func(images)
    policy_logits = predictions["policy_head"]
    value = predictions["value_head"]
    return value, policy_logits

@tf.function
def predict_model(model, images):
    #image = tf.expand_dims(image, axis=0)
    values, policy_logits = model(images)
    return values, policy_logits

def reshape_planes(planes):
    x = planes[:, :-1]
    shape = x.get_shape()
    x = tf.expand_dims(x, -1)
    mask = tf.bitwise.left_shift(tf.ones([], dtype=tf.int64), tf.range(64, dtype=tf.int64))
    x = tf.bitwise.bitwise_and(x, mask)
    x = tf.cast(x, tf.bool)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, [-1, int(shape[1]), 8, 8])

    y = tf.cast(planes[:, -1:], tf.float32)[:, :, tf.newaxis, tf.newaxis] / tf.fill([1, 1, 8, 8], 99.0)
    return tf.concat([x, y], axis=1)

def generate_model():
    # Define the input layer
    input_layer = Input(shape=(NUM_PLANES), name="input_layer", dtype=tf.int64)
    x = reshape_planes(input_layer)

    # Define the body
    x = Conv2D(filters=CONV_FILTERS , kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(L2_REG), name="Body-Conv2D")(x)
    x = BatchNormalization(name="Body-BatchNorm", axis=1)(x)
    x = ReLU(name="Body-ReLU")(x)

    # Create the residual blocks tower
    for i in range(NUM_RESIDUAL_BLOCKS):
        block = Conv2D(filters=CONV_FILTERS, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(L2_REG), name=f"ResBlock_{i}-Conv2D_1")(x)
        block = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_1", axis=1)(block)
        block = ReLU(name=f"ResBlock_{i}-ReLU_1")(block)
        block = Conv2D(filters=CONV_FILTERS, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(L2_REG), name=f"ResBlock_{i}-Conv2D_2")(block)
        block = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_2", axis=1)(block)
        block = Add(name=f"ResBlock_{i}-SkipCon")([x, block])
        x = ReLU(name=f"ResBlock_{i}-ReLU_2")(block)

    value_head = Conv2D(filters=VALUE_HEAD_FILTERS, kernel_size=1, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(L2_REG), name=f"ValueHead-Conv2D")(x)
    value_head = BatchNormalization(name=f"ValueHead-BatchNorm", axis=1)(value_head)
    value_head = ReLU(name=f"ValueHead-ReLU")(value_head)
    value_head = Flatten(name=f"ValueHead-Flatten", data_format="channels_first")(value_head)
    value_head = Dense(VALUE_HEAD_DENSE, activation='relu', use_bias=USE_BIAS_ON_OUTPUTS, kernel_regularizer=l2(L2_REG), name=f"ValueHead-DenseReLU")(value_head)
    value_head = Dense(1, activation="tanh", use_bias=USE_BIAS_ON_OUTPUTS, kernel_regularizer=l2(L2_REG), name="value_head")(value_head)

    # Define the policy head and value head
    policy_head = Conv2D(filters=POLICY_HEAD_FILTERS, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(L2_REG), name="PolicyHead-Conv2D")(x)
    policy_head = BatchNormalization(name="PolicyHead-BatchNorm", axis=1)(policy_head)
    policy_head = ReLU(name="PolicyHead-ReLU")(policy_head)
    policy_head = Flatten(name="PolicyHead-Flatten", data_format="channels_first")(policy_head)
    policy_head = Dense(POLICY_HEAD_DENSE, activation='linear', use_bias=USE_BIAS_ON_OUTPUTS, kernel_regularizer=l2(L2_REG), name="policy_head")(policy_head)

    # Define the optimizer
    optimizer = SGD(learning_rate=0.2, nesterov=SGD_NESTEROV, momentum=SGD_MOMENTUM)

    # Define the model
    model = Model(inputs=input_layer, outputs=[value_head, policy_head])
    model.compile(
        optimizer=optimizer,
        loss={
            "value_head": "mean_squared_error",
            "policy_head": CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            "value_head": VALUE_HEAD_LOSS_WEIGHT,
            "policy_head": POLICY_HEAD_LOSS_WEIGHT
        },
        metrics=["accuracy"]
    )
    return model


def update_trt_model(config, model_version: str = "latest", precision_mode: str = "FP32", build_model: bool = True):
    """ Save new model as a TensorRT model or update the existing one.

    Args:
        config (dict): A dictionary containing the configuration for the project.
        model_version (str, optional): How to label the directory containing models. Defaults to "latest".
        precision_mode (str, optional): Precision mode. Defaults to "FP32".
        build_model (bool, optional): Whether to build model beforehand or after first use. Defaults to False.

    """
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    def input_fn():
        np_data = np.load(os.path.join(config['project_dir'], 'data', 'conversion_data', 'histories.npz'))
        histories = np_data["histories"]
        # Yield the histories in batches of size defaultConfig["batch_size"]
        for i in range(0, len(histories), config["num_vl_searches"]):
            yield (histories[i:i + config["num_vl_searches"]],)
    
    model_save_path = os.path.join(config['project_dir'], 'data', 'models', model_version, 'saved_model')

    tf_model = tf.keras.models.load_model(os.path.join(config['project_dir'], 'data', 'models', 'model.keras'))
    tf_model.save(model_save_path)

    conversion_params = trt.TrtConversionParams(
        precision_mode=precision_mode,
        use_calibration=True if precision_mode == "INT8" else False,
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_save_path,
        conversion_params=conversion_params,
    )

    if precision_mode == "INT8":
        converter.convert(calibration_input_fn=input_fn)
    else:
        converter.convert()

    if precision_mode != "INT8" and build_model:
        converter.build(input_fn=input_fn)

    converter.save(model_save_path)


def load_as_trt_model(model_version: str = "latest"):
    """ Load a TensorRT model.

    Args:
        model_version (str, optional): Name of directory that contains the model. Defaults to "latest".

    Raises:
        FileNotFoundError: Error if models is not found

    Returns:
        (func, obj): A tuple containing the TensorRT inference function and the loaded model.
    """
    model_save_path = os.path.join(config['project_dir'], 'data', 'models', model_version, 'saved_model')
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model not found at {model_save_path}")

    loaded_model = tf.saved_model.load(model_save_path)
    trt_func = loaded_model.signatures['serving_default']
    return trt_func, loaded_model