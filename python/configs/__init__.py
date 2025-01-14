import configparser
import os

current_dir = os.path.dirname(__file__)

_ints = [
    # Config
    "history_steps",
    "history_repetition_planes",
    "actionspace_size",
    "batch_size",
    "num_mcts_sims",
    "num_mcts_sampling_moves",
    "num_vl_searches",

    # Model
    "conv_filters",
    "residual_blocks",
    "policy_head_filters",
    "value_head_filters",
    "policy_head_dense",
    "value_head_dense",
]

_floats = [
    # Config
    "root_dirichlet_alpha",
    "root_exploration_fraction",
    "fpu_root",
    "fpu_leaf",
    "moves_softmax_temp",
    "policy_temp",
    "pb_c_base",
    "pb_c_factor",
    "pb_c_init",

    # Model
    "learning_rate",
    "l2_regularization",
    "policy_head_loss_weight",
    "value_head_loss_weight",
    "sgd_momentum",
]

_bools = [
    # Config
    "history_perspective_flip",
    "root_exploration_noise",

    # Model
    "use_bias_on_output",
    "sgd_nesterov",
]

def get_config(baseconfig, section: str):
    config = {}
    for key in baseconfig[section].keys():
        if key in _ints:
            config[key] = baseconfig[section].getint(key)
        elif key in _floats:
            config[key] = baseconfig[section].getfloat(key)
        elif key in _bools:
            config[key] = baseconfig[section].getboolean(key)
        else:
            config[key] = baseconfig[section].get(key)
    return config


_baseconfig = configparser.ConfigParser()
_baseconfig.read(f"{current_dir}/config.toml")

defaultConfig = get_config(_baseconfig, "DEFAULT")
selfplayConfig = get_config(_baseconfig, "selfplay")
trainingConfig = get_config(_baseconfig, "training")

_modelconfig = configparser.ConfigParser()
_modelconfig.read(f"{current_dir}/model.toml")
modelConfig = get_config(_modelconfig, defaultConfig["model_size"])