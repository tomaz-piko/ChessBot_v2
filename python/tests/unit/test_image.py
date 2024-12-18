import unittest
import numpy as np
from rchess import Board
from utils import convert_u64_to_np
from configs import defaultConfig

T = 8
M = (6 * 2 + defaultConfig["history_repetition_planes"])
flip = False
planes_count = T * M + 5

# Ruy lopez opening from whites perspective
# Turn 0: (e4) B    Turn 1: (e5) W    Turn 2: (Nf3) B   Turn 3: (Nc6) W   Turn 4: (Bb5) B   Turn 5: (a6)  W   Turn 6: (Ba4) B   Turn 7: (Nf6) W       
# r n b q k b n r   r n b q k b n r   r n b q k b n r   r . b q k b n r   r . b q k b n r   r . b q k b n r   r . b q k b n r   r . b q k b . r
# p p p p p p p p   p p p p . p p p   p p p p . p p p   p p p p . p p p   p p p p . p p p   . p p p . p p p   . p p p . p p p   . p p p . p p p
# . . . . . . . .   . . . . . . . .   . . . . . . . .   . . n . . . . .   . . n . . . . .   p . n . . . . .   p . n . . . . .   p . n . . n . .
# . . . . . . . .   . . . . p . . .   . . . . p . . .   . . . . p . . .   . B . . p . . .   . B . . p . . .   . . . . p . . .   . . . . p . . .
# . . . . P . . .   . . . . P . . .   . . . . P . . .   . . . . P . . .   . . . . P . . .   . . . . P . . .   B . . . P . . .   B . . . P . . .
# . . . . . . . .   . . . . . . . .   . . . . . N . .   . . . . . N . .   . . . . . N . .   . . . . . . . .   . . . . . N . .   . . . . . N . . 
# P P P P . P P P   P P P P . P P P   P P P P . P P P   P P P P . P P P   P P P P . P P P   P P P P . P P R   P P P P . P P P   P P P P . P P P
# R N B Q K B N R   R N B Q K B N R   R N B Q K B . R   R N B Q K B . R   R N B Q K . . R   R N B Q K . . R   R N B Q K . . R   R N B Q K . . R

# Including starting position 7 -> 8 time steps
ruy_lopez = Board() # W
ruy_lopez.push_uci("e2e4") # B
ruy_lopez.push_uci("e7e5") # W
ruy_lopez.push_uci("g1f3") # B
ruy_lopez.push_uci("b8c6") # W
ruy_lopez.push_uci("f1b5") # B
ruy_lopez.push_uci("a7a6") # W
ruy_lopez.push_uci("b5a4") # B
ruy_lopez.push_uci("g8f6") # W (If we create image here its white's turn)

# Ruy lopez opening from blacks perspective
# Turn 1:           Turn 2:           Turn 3:           Turn 4:           Turn 5:           Turn 6:           Turn 7:           Turn 8:
# R N B K Q B N R   R . B K Q B N R   R . B K Q B N R   R . . K Q B N R   R . . K Q B N R   R . . K Q B N R   R . . K Q B N R   . K R . Q B N R
# P P P . P P P P   P P P . P P P P   P P P . P P P P   P P P . P P P P   P P P . P P P R   P P P . P P P P   P P P . P P P P   P P P P . P P P
# . . . . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .
# . . . P . . . .   . . . P . . . .   . . . P . . . .   . . . P . . . .   . . . P . . . .   . . . P . . . B   . . . P . . . B   . . . . P . . B
# . . . p . . . .   . . . p . . . .   . . . p . . . .   . . . p . . B .   . . . p . . B .   . . . p . . . .   . . . p . . . .   . . . . p . . .
# . . . . . . . .   . . . . . . . .   . . . . . n . .   . . . . . n . .   . . . . . n . p   . . . . . n . p   . . n . . n . p   . . n . . n . .
# p p p . p p p p   p p p . p p p p   p p p . p p p p   p p p . p p p p   p p p . p p p .   p p p . p p p .   p p p . p p p .   p p p p . p p p
# r n b k q b n r   r n b k q b n r   r n b k q b . r   r n b k q b . r   r n b k q b . r   r n b k q b . r   r . b k q b . r   r . b k q b . r

ruy_lopez2 = ruy_lopez.clone()
ruy_lopez2.push_uci("e1g1") # B

piece = {"pawn": 0, "knight": 1, "bishop": 2, "rook": 3, "queen": 4, "king": 5}

class TestGameImage(unittest.TestCase):
    def test_image_shape(self):
        board = Board()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :]
        self.assertEqual(image.shape, (planes_count, 8, 8))

    def test_image_dtype(self):
        board = Board()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :]
        self.assertEqual(image.dtype, np.float32)

    def test_half_move_clock_is_last(self):
        board = Board()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        half_move_plane_idx = -1
        self.assertTrue(np.all(image[0, half_move_plane_idx] == 0))

    def test_half_move_clock(self):
        board = Board()
        board.push_uci("g1f3")
        board.push_uci("g8f6")
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        half_move_plane_idx = -1
        self.assertTrue(np.all(image[0, half_move_plane_idx] == 2 / 100))

    def test_ruy_lopez_white_t0(self): # Checking black and white pieces but from white perspective
        t = 0
        board = ruy_lopez.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts0_w = image[t*M:t*M+6]
        ts0_b = image[t*M+6:t*M+12]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
     
        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts0_w[piece["pawn"]], pawns_w), f"PawnsW0:\nExpected:\n{pawns_w}\n\nGot:\n{ts0_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["pawn"]], pawns_b), f"PawnsB0:\nExpected:\n{pawns_b}\n\nGot:\n{ts0_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["knight"]], knights_w), f"KnightsW0:\nExpected:\n{knights_w}\n\nGot:\n{ts0_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["knight"]], knights_b), f"KnightsB0:\nExpected:\n{knights_b}\n\nGot:\n{ts0_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["bishop"]], bishops_w), f"BishopsW0:\nExpected:\n{bishops_w}\n\nGot:\n{ts0_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["bishop"]], bishops_b), f"BishopsB0:\nExpected:\n{bishops_b}\n\nGot:\n{ts0_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["rook"]], rooks_w), f"RooksW0:\nExpected:\n{rooks_w}\n\nGot:\n{ts0_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["rook"]], rooks_b), f"RooksB0:\nExpected:\n{rooks_b}\n\nGot:\n{ts0_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["queen"]], queens_w), f"QueenW0:\nExpected:\n{queens_w}\n\nGot:\n{ts0_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["queen"]], queens_b), f"QueenB0:\nExpected:\n{queens_b}\n\nGot:\n{ts0_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["king"]], kings_w), f"KingW0:\nExpected:\n{kings_w}\n\nGot:\n{ts0_w[piece['king']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["king"]], kings_b), f"KingB0:\nExpected:\n{kings_b}\n\nGot:\n{ts0_b[piece['king']]}")


    def test_ruy_lopez_white_t1(self): # Checking black and white pieces but from white perspective
        t = 1
        board = ruy_lopez.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)
        ts1_b = image[t*M:t*M+6]
        ts1_w = image[t*M+6:t*M+12]
        
        # At this point its blacks turn
        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])

        knights_w = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # This step is only needed on turns with an odd index: e.g. t1, t3, ...
        # Flipped perspective used to be default in the past
        # This is the reason the next part of code is a bit counter intuitive
        # If perspective flipping is disabled we flip the flipped perspective to get the correct perspective
        if not flip:
            ts1_b, ts1_w = ts1_w, ts1_b
            pawns_w, pawns_b = np.flip(pawns_w, axis=0), np.flip(pawns_b, axis=0)
            knights_w, knights_b = np.flip(knights_w, axis=0), np.flip(knights_b, axis=0)
            bishops_w, bishops_b = np.flip(bishops_w, axis=0), np.flip(bishops_b, axis=0)
            rooks_w, rooks_b = np.flip(rooks_w, axis=0), np.flip(rooks_b, axis=0)
            queens_w, queens_b = np.flip(queens_w, axis=0), np.flip(queens_b, axis=0)
            kings_w, kings_b = np.flip(kings_w, axis=0), np.flip(kings_b, axis=0)

        self.assertTrue(np.array_equal(ts1_w[piece["pawn"]], pawns_w), f"PawnsW1:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["pawn"]], pawns_b), f"PawnsB1:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["knight"]], knights_w), f"KnightsW1:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["knight"]], knights_b), f"KnightsB1:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["bishop"]], bishops_w), f"BishopsW1:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["bishop"]], bishops_b), f"BishopsB1:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["rook"]], rooks_w), f"RooksW1:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["rook"]], rooks_b), f"RooksB1:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["queen"]], queens_w), f"QueenW1:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["queen"]], queens_b), f"QueenB1:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["king"]], kings_w), f"KingW1:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["king"]], kings_b), f"KingB1:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[piece['king']]}")

    def test_ruy_lopez_white_t2(self): # Checking black and white pieces but from white perspective
        t=2
        board = ruy_lopez.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts2_w = image[t*M:t*M+6]
        ts2_b = image[t*M+6:t*M+12]
        
        # At this point its whites turn
        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts2_w[piece["pawn"]], pawns_w), f"PawnsW2:\nExpected:\n{pawns_w}\n\nGot:\n{ts2_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["pawn"]], pawns_b), f"PawnsB2:\nExpected:\n{pawns_b}\n\nGot:\n{ts2_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["knight"]], knights_w), f"KnightsW2:\nExpected:\n{knights_w}\n\nGot:\n{ts2_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["knight"]], knights_b), f"KnightsB2:\nExpected:\n{knights_b}\n\nGot:\n{ts2_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["bishop"]], bishops_w), f"BishopsW2:\nExpected:\n{bishops_w}\n\nGot:\n{ts2_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["bishop"]], bishops_b), f"BishopsB2:\nExpected:\n{bishops_b}\n\nGot:\n{ts2_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["rook"]], rooks_w), f"RooksW2:\nExpected:\n{rooks_w}\n\nGot:\n{ts2_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["rook"]], rooks_b), f"RooksB2:\nExpected:\n{rooks_b}\n\nGot:\n{ts2_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["queen"]], queens_w), f"QueenW2:\nExpected:\n{queens_w}\n\nGot:\n{ts2_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["queen"]], queens_b), f"QueenB2:\nExpected:\n{queens_b}\n\nGot:\n{ts2_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["king"]], kings_w), f"KingW2:\nExpected:\n{kings_w}\n\nGot:\n{ts2_w[piece['king']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["king"]], kings_b), f"KingB2:\nExpected:\n{kings_b}\n\nGot:\n{ts2_b[piece['king']]}")

    def test_ruy_lopez_white_t3(self): # Checking black and white pieces but from white perspective
        t=3
        board = ruy_lopez.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts3_b = image[t*M:t*M+6]
        ts3_w = image[t*M+6:t*M+12]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])

        knights_w = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # This step is only needed on turns with an odd index: e.g. t1, t3, ...
        # Flipped perspective used to be default in the past
        # This is the reason the next part of code is a bit counter intuitive
        # If perspective flipping is disabled we flip the flipped perspective to get the correct perspective
        if not flip:
            ts3_b, ts3_w = ts3_w, ts3_b
            pawns_w, pawns_b = np.flip(pawns_w, axis=0), np.flip(pawns_b, axis=0)
            knights_w, knights_b = np.flip(knights_w, axis=0), np.flip(knights_b, axis=0)
            bishops_w, bishops_b = np.flip(bishops_w, axis=0), np.flip(bishops_b, axis=0)
            rooks_w, rooks_b = np.flip(rooks_w, axis=0), np.flip(rooks_b, axis=0)
            queens_w, queens_b = np.flip(queens_w, axis=0), np.flip(queens_b, axis=0)
            kings_w, kings_b = np.flip(kings_w, axis=0), np.flip(kings_b, axis=0)

        self.assertTrue(np.array_equal(ts3_w[piece["pawn"]], pawns_w), f"PawnsW3:\nExpected:\n{pawns_w}\n\nGot:\n{ts3_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["pawn"]], pawns_b), f"PawnsB3:\nExpected:\n{pawns_b}\n\nGot:\n{ts3_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["knight"]], knights_w), f"KnightsW3:\nExpected:\n{knights_w}\n\nGot:\n{ts3_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["knight"]], knights_b), f"KnightsB3:\nExpected:\n{knights_b}\n\nGot:\n{ts3_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["bishop"]], bishops_w), f"BishopsW3:\nExpected:\n{bishops_w}\n\nGot:\n{ts3_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["bishop"]], bishops_b), f"BishopsB3:\nExpected:\n{bishops_b}\n\nGot:\n{ts3_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["rook"]], rooks_w), f"RooksW3:\nExpected:\n{rooks_w}\n\nGot:\n{ts3_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["rook"]], rooks_b), f"RooksB3:\nExpected:\n{rooks_b}\n\nGot:\n{ts3_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["queen"]], queens_w), f"QueenW3:\nExpected:\n{queens_w}\n\nGot:\n{ts3_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["queen"]], queens_b), f"QueenB3:\nExpected:\n{queens_b}\n\nGot:\n{ts3_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["king"]], kings_w), f"KingW3:\nExpected:\n{kings_w}\n\nGot:\n{ts3_w[piece['king']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["king"]], kings_b), f"KingB3:\nExpected:\n{kings_b}\n\nGot:\n{ts3_b[piece['king']]}")


    def test_ruy_lopez_black_t0(self):
        t = 0
        board = ruy_lopez2.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts0_b = image[t*M:t*M+6]
        ts0_w = image[t*M+6:t*M+12]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts0_b[piece["pawn"]], pawns_b), f"PawnsB0:\nExpected:\n{pawns_b}\n\nGot:\n{ts0_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["pawn"]], pawns_w), f"PawnsW0:\nExpected:\n{pawns_w}\n\nGot:\n{ts0_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["knight"]], knights_b), f"KnightsB0:\nExpected:\n{knights_b}\n\nGot:\n{ts0_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["knight"]], knights_w), f"KnightsW0:\nExpected:\n{knights_w}\n\nGot:\n{ts0_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["bishop"]], bishops_b), f"BishopsB0:\nExpected:\n{bishops_b}\n\nGot:\n{ts0_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["bishop"]], bishops_w), f"BishopsW0:\nExpected:\n{bishops_w}\n\nGot:\n{ts0_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["rook"]], rooks_b), f"RooksB0:\nExpected:\n{rooks_b}\n\nGot:\n{ts0_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["rook"]], rooks_w), f"RooksW0:\nExpected:\n{rooks_w}\n\nGot:\n{ts0_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["queen"]], queens_b), f"QueenB1:\nExpected:\n{queens_b}\n\nGot:\n{ts0_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["queen"]], queens_w), f"QueenW1:\nExpected:\n{queens_w}\n\nGot:\n{ts0_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts0_b[piece["king"]], kings_b), f"KingB1:\nExpected:\n{kings_b}\n\nGot:\n{ts0_b[piece['king']]}")
        self.assertTrue(np.array_equal(ts0_w[piece["king"]], kings_w), f"KingW1:\nExpected:\n{kings_w}\n\nGot:\n{ts0_w[piece['king']]}")

    def test_ruy_lopez_black_t1(self):
        t=1
        board = ruy_lopez2.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts1_w = image[t*M:t*M+6]
        ts1_b = image[t*M+6:t*M+12]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # This step is only needed on turns with an odd index: e.g. t1, t3, ...
        # Flipped perspective used to be default in the past
        # This is the reason the next part of code is a bit counter intuitive
        # If perspective flipping is disabled we flip the flipped perspective to get the correct perspective
        if not flip:
            ts1_b, ts1_w = ts1_w, ts1_b
            pawns_w, pawns_b = np.flip(pawns_w, axis=0), np.flip(pawns_b, axis=0)
            knights_w, knights_b = np.flip(knights_w, axis=0), np.flip(knights_b, axis=0)
            bishops_w, bishops_b = np.flip(bishops_w, axis=0), np.flip(bishops_b, axis=0)
            rooks_w, rooks_b = np.flip(rooks_w, axis=0), np.flip(rooks_b, axis=0)
            queens_w, queens_b = np.flip(queens_w, axis=0), np.flip(queens_b, axis=0)
            kings_w, kings_b = np.flip(kings_w, axis=0), np.flip(kings_b, axis=0)

        self.assertTrue(np.array_equal(ts1_b[piece["pawn"]], pawns_b), f"PawnsB2:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["pawn"]], pawns_w), f"PawnsW2:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["knight"]], knights_b), f"KnightsB2:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["knight"]], knights_w), f"KnightsW2:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["bishop"]], bishops_b), f"BishopsB2:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["bishop"]], bishops_w), f"BishopsW2:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["rook"]], rooks_b), f"RooksB2:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["rook"]], rooks_w), f"RooksW2:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["queen"]], queens_b), f"QueenB2:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["queen"]], queens_w), f"QueenW2:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[piece["king"]], kings_b), f"KingB2:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[piece["king"]], kings_w), f"KingW2:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[piece['king']]}")

    def test_ruy_lopez_black_t2(self):
        t=2
        board = ruy_lopez2.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts2_b = image[t*M:t*M+6]
        ts2_w = image[t*M+6:t*M+12]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])

        knights_w = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts2_b[piece["pawn"]], pawns_b), f"PawnsB3:\nExpected:\n{pawns_b}\n\nGot:\n{ts2_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["pawn"]], pawns_w), f"PawnsW3:\nExpected:\n{pawns_w}\n\nGot:\n{ts2_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["knight"]], knights_b), f"KnightsB3:\nExpected:\n{knights_b}\n\nGot:\n{ts2_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["knight"]], knights_w), f"KnightsW3:\nExpected:\n{knights_w}\n\nGot:\n{ts2_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["bishop"]], bishops_b), f"BishopsB3:\nExpected:\n{bishops_b}\n\nGot:\n{ts2_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["bishop"]], bishops_w), f"BishopsW3:\nExpected:\n{bishops_w}\n\nGot:\n{ts2_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["rook"]], rooks_b), f"RooksB3:\nExpected:\n{rooks_b}\n\nGot:\n{ts2_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["rook"]], rooks_w), f"RooksW3:\nExpected:\n{rooks_w}\n\nGot:\n{ts2_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["queen"]], queens_b), f"QueenB3:\nExpected:\n{queens_b}\n\nGot:\n{ts2_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["queen"]], queens_w), f"QueenW3:\nExpected:\n{queens_w}\n\nGot:\n{ts2_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts2_b[piece["king"]], kings_b), f"KingB3:\nExpected:\n{kings_b}\n\nGot:\n{ts2_b[piece['king']]}")
        self.assertTrue(np.array_equal(ts2_w[piece["king"]], kings_w), f"KingW3:\nExpected:\n{kings_w}\n\nGot:\n{ts2_w[piece['king']]}")

    def test_ruy_lopez_black_t3(self):
        t=3
        board = ruy_lopez2.clone()
        hist, _ = board.history(flip)
        image = convert_u64_to_np(hist)
        image = image[0, :, :, :].astype(np.uint8)

        ts3_w = image[t*M:t*M+6]
        ts3_b = image[t*M+6:t*M+12]
        
        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # This step is only needed on turns with an odd index: e.g. t1, t3, ...
        # Flipped perspective used to be default in the past
        # This is the reason the next part of code is a bit counter intuitive
        # If perspective flipping is disabled we flip the flipped perspective to get the correct perspective
        if not flip:
            ts3_b, ts3_w = ts3_w, ts3_b
            pawns_w, pawns_b = np.flip(pawns_w, axis=0), np.flip(pawns_b, axis=0)
            knights_w, knights_b = np.flip(knights_w, axis=0), np.flip(knights_b, axis=0)
            bishops_w, bishops_b = np.flip(bishops_w, axis=0), np.flip(bishops_b, axis=0)
            rooks_w, rooks_b = np.flip(rooks_w, axis=0), np.flip(rooks_b, axis=0)
            queens_w, queens_b = np.flip(queens_w, axis=0), np.flip(queens_b, axis=0)
            kings_w, kings_b = np.flip(kings_w, axis=0), np.flip(kings_b, axis=0)

        self.assertTrue(np.array_equal(ts3_b[piece["pawn"]], pawns_b), f"PawnsB4:\nExpected:\n{pawns_b}\n\nGot:\n{ts3_b[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["pawn"]], pawns_w), f"PawnsW4:\nExpected:\n{pawns_w}\n\nGot:\n{ts3_w[piece['pawn']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["knight"]], knights_b), f"KnightsB4:\nExpected:\n{knights_b}\n\nGot:\n{ts3_b[piece['knight']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["knight"]], knights_w), f"KnightsW4:\nExpected:\n{knights_w}\n\nGot:\n{ts3_w[piece['knight']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["bishop"]], bishops_b), f"BishopsB4:\nExpected:\n{bishops_b}\n\nGot:\n{ts3_b[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["bishop"]], bishops_w), f"BishopsW4:\nExpected:\n{bishops_w}\n\nGot:\n{ts3_w[piece['bishop']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["rook"]], rooks_b), f"RooksB4:\nExpected:\n{rooks_b}\n\nGot:\n{ts3_b[piece['rook']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["rook"]], rooks_w), f"RooksW4:\nExpected:\n{rooks_w}\n\nGot:\n{ts3_w[piece['rook']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["queen"]], queens_b), f"QueenB4:\nExpected:\n{queens_b}\n\nGot:\n{ts3_b[piece['queen']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["queen"]], queens_w), f"QueenW4:\nExpected:\n{queens_w}\n\nGot:\n{ts3_w[piece['queen']]}")
        self.assertTrue(np.array_equal(ts3_b[piece["king"]], kings_b), f"KingB4:\nExpected:\n{kings_b}\n\nGot:\n{ts3_b[piece['king']]}")
        self.assertTrue(np.array_equal(ts3_w[piece["king"]], kings_w), f"KingW4:\nExpected:\n{kings_w}\n\nGot:\n{ts3_w[piece['king']]}")
        

if __name__ == '__main__':
    unittest.main()