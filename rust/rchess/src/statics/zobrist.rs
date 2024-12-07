use crate::statics::prng::Prng;
use crate::types::piece::Piece;
use crate::types::square::Square;
use crate::types::castling_rights::CastlingRights;

static mut ZOBRIST_TABLE: [[u64; 64]; 6] = [[0; 64]; 6];
static mut ZOBRIST_BLACK: u64 = 0;
static mut ZOBRIST_EP: [u64; 8] = [0; 8];
static mut ZOBRIST_CASTLING: [u64; 16] = [0; 16];

#[inline(always)]
pub fn zobrist_piece(piece: Piece, square: Square) -> u64 {
    unsafe { ZOBRIST_TABLE[piece as usize][square as usize] }
}

#[inline(always)]
pub fn zobrist_black() -> u64 {
    unsafe { ZOBRIST_BLACK }
}

#[inline(always)]
pub fn zobrist_ep(ep: Square) -> u64 {
    unsafe { ZOBRIST_EP[ep.file_idx()] }
}

#[inline(always)]
pub fn zobrist_castling(castling: CastlingRights) -> u64 {
    unsafe { ZOBRIST_CASTLING[castling] }
}

#[cold]
pub fn init_zobrist() {
    unsafe {
        generate_zobrist();
    }
}

#[cold]
unsafe fn generate_zobrist() {
    let mut rng = Prng::init(70026072);
    for piece in 0..6 {
        for sq in 0..64 {
            ZOBRIST_TABLE[piece][sq] = rng.rand();
        }
    }
    ZOBRIST_BLACK = rng.rand();
    for i in 0..8 {
        ZOBRIST_EP[i] = rng.rand();
    }
    for i in 0..16 {
        ZOBRIST_CASTLING[i] = rng.rand();
    }
}
