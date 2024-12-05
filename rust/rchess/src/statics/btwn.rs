use crate::bitboard::Bitboard;
use crate::bitboard::masks::BB_EMPTY;
use crate::types::square::{Square, SQUARES};

static mut SQUARES_BETWEEN: [[Bitboard; 64]; 64] = [[BB_EMPTY; 64]; 64];

#[inline]
pub fn squares_between(sq1: Square, sq2: Square) -> Bitboard {
    unsafe {
        SQUARES_BETWEEN[sq1 as usize][sq2 as usize]
    }
}

#[cold]
pub fn init_squares_between() {
    unsafe {
        generate_squares_between();
    }
}

#[cold]
unsafe fn generate_squares_between() {
    for sq1 in SQUARES {
        for sq2 in SQUARES {
            SQUARES_BETWEEN[sq1 as usize][sq2 as usize] = Bitboard::squares_between(sq1, sq2);
        }
    }
}