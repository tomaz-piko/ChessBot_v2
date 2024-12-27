cimport numpy as np

cpdef select_child(object node, unsigned int pb_c_base, float pb_c_init, float pb_c_factor, float fpu)

cpdef expand_node(object node, list legal_moves, bint to_play)

cpdef void evaluate_node(object node, float[:] policy_logits)

cpdef float value_to_01(float value)

cpdef float flip_value(float value)

cpdef void backup(list search_path, float value)

cpdef void add_vloss(list search_path)

cpdef np.ndarray calculate_search_statistics(object root, unsigned int num_actions)
