class BaseConfig:
    __history_steps = 8
    __history_perspective_flip = False
    __repetition_planes = 2
    __history_planes = __history_steps * (6 * 2 + __repetition_planes) + 5

    __image_shape = (__history_planes, 8, 8)
    __num_actions = 1858

    __root_dirichlet_alpha = 0.3
    __root_exploration_fraction = 0.25
    __pb_c_base = 19652

    __num_vl_searches = 16

    def __init__(self):
        self.type = 'BaseConfig'
        self.__pb_c_factor = (1.0, 1.0)
        self.__pb_c_init = 1.25
        self.__fpu = (1.0, 0.3)
        self.__softmax_temp = 1.0
        self.__policy_temp = 1.0
        self.__num_mcts_sims = 0
        self.__num_mcts_sampling_moves = 0
        self.__add_root_noise = True

    @property
    def history_steps(self):
        return self.__history_steps
    
    @property
    def history_perspective_flip(self):
        return self.__history_perspective_flip
    
    @property
    def repetition_planes(self):
        return self.__repetition_planes
    
    @property
    def history_planes(self):
        return self.__history_planes
    
    @property
    def image_shape(self):
        return self.__image_shape
    
    @property
    def num_actions(self):
        return self.__num_actions
    
    @property
    def root_dirichlet_alpha(self):
        return self.__root_dirichlet_alpha
    
    @property
    def root_exploration_fraction(self):
        return self.__root_exploration_fraction
    
    @property
    def pb_c_base(self):
        return self.__pb_c_base
    
    @property
    def pb_c_factor(self):
        return self.__pb_c_factor
    
    @property
    def pb_c_init(self):
        return self.__pb_c_init
    
    @property
    def fpu(self):
        return self.__fpu
    
    @property
    def softmax_temp(self):
        return self.__softmax_temp
    
    @property
    def policy_temp(self):
        return self.__policy_temp
    
    @property
    def add_root_noise(self):
        return self.__add_root_noise
    
    @property
    def num_vl_searches(self):
        return self.__num_vl_searches
    
    @property
    def num_mcts_sims(self):
        return self.__num_mcts_sims
    
    @property
    def num_mcts_sampling_moves(self):
        return self.__num_mcts_sampling_moves

class TestConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.type = 'TestConfig'
        self.__num_mcts_sims = 100
        self.__num_mcts_sampling_moves = 30
        self.__add_root_noise = True

class PlayConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.type = 'PlayConfig'
        self.__num_mcts_sims = None
        self.__num_mcts_sampling_moves = 0
        self.__add_root_noise = False

class SelfplayConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.type = 'SelfplayConfig'
        self.__num_mcts_sims = 800
        self.__num_mcts_sampling_moves = 30
        self.__add_root_noise = True