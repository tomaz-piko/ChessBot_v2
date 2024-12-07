import numpy as np

def convert_u64_to_np(x: list) -> np.array:
    x = np.array(x, dtype=np.uint64)
    shape = x.shape
    x = np.expand_dims(x, axis=-1)
    mask = np.left_shift(1, np.arange(64, dtype=np.uint64))
    x = np.bitwise_and(x, mask)
    x = x.astype(bool)
    x = x.astype(np.float32)
    return np.reshape(x, (-1, shape[-1], 8, 8))