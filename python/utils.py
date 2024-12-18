import numpy as np

def convert_u64_to_np(images: list) -> np.array:
    """Converts a list or a batch of lists containing u64 integers, to a numpy array.
       First 116 integers are converted to a 8x8x116 numpy array where a single integer is represented as a 8x8 binary matrix.
       The last integer represents the halfmove clock and is repeated to form a 8x8x1 matrix.

    Args:
        image (list): List of u64 integers to be converted to a numpy array

    Returns:
        np.array: Numpy array of shape (batch_size, 117, 8, 8) representing the board state        
    """
    x = np.array(images, dtype=np.uint64)
    if len(x.shape) == 1:
        x = np.array([x], dtype=np.uint64)
    h = x[:, -1]
    x = x[:, :-1]
    batch_size = x.shape[0]
    x = np.expand_dims(x, axis=-1)
    mask = np.left_shift(1, np.arange(64, dtype=np.uint64))
    x = np.bitwise_and(x, mask)
    x = x.astype(bool)
    x = x.astype(np.float32)
    x = np.reshape(x, (-1, 108, 8, 8))
    h = h.astype(np.float32)
    h = (np.ones((batch_size, 1, 8, 8)) * h[:, np.newaxis, np.newaxis, np.newaxis]) / 100.0
    x = np.concatenate([x, h], axis=1).astype(np.float32)
    return x