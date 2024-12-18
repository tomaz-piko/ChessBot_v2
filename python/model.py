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
from keras.optimizers import SGD, Adam
import keras.backend as K

from configs import defaultConfig, modelConfig

K.set_image_data_format("channels_last")

R = defaultConfig["history_repetition_planes"]
T = defaultConfig["history_steps"]
num_planes = (6 * 2 + R) * T + 5 

INPUT_SHAPE = (None, num_planes)
CONV_FILTERS = modelConfig["conv_filters"]
NUM_RESIDUAL_BLOCKS = modelConfig["num_residual_blocks"]
CONV_KERNEL_INITIALIZER = modelConfig["conv_kernel_initializer"] if modelConfig["conv_kernel_initializer"] else None #"he_normal"
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
LEARNING_RATE = modelConfig["learning_rate"]

def generate_model():
    # Define the input layer
    input_layer = Input(shape=INPUT_SHAPE, name="input_layer")

    # Define the body
    x = Conv2D(filters=CONV_FILTERS , kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, kernel_regularizer=l2(L2_REG), name="Body-Conv2D")(input_layer)
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
    optimizer = SGD(learning_rate=LEARNING_RATE, nesterov=SGD_NESTEROV, momentum=SGD_MOMENTUM)

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