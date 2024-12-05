use crate::bitboard::Bitboard;
use crate::bitboard::masks::BB_EMPTY;
use crate::types::square::{Square, SQUARES};

static mut LINES: [[Bitboard; 64]; 64] = [[BB_EMPTY; 64]; 64];

#[inline]
pub fn line_through(sq1: Square, sq2: Square) -> Bitboard {
    unsafe {
        LINES[sq1 as usize][sq2 as usize]
    }
}

#[cold]
pub fn init_lines_between() {
    unsafe {
        generate_lines_through();
    }
}

#[cold]
unsafe fn generate_lines_through() {
    for sq1 in SQUARES {
        for sq2 in SQUARES {
            LINES[sq1 as usize][sq2 as usize] = Bitboard::line_through(sq1, sq2);
        }
    }
}