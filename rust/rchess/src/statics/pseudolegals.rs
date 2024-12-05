use crate::bitboard::Bitboard;
use crate::bitboard::masks::{BB_EMPTY, BB_KNIGHT_ATTACKS, BB_KING_ATTACKS, BB_PAWN_ATTACKS};
use crate::types::color::Color;
use crate::types::piece::{Piece, BISHOP, KING, KNIGHT, QUEEN, ROOK};
use crate::types::square::{Square, SQUARES};

static mut PAWN_ATTACKS: [[Bitboard; 64]; 2] = [[BB_EMPTY; 64]; 2];
static mut PIECE_PSEUDO_LEGALS_ATTACKS: [[Bitboard; 64]; 5] = [[BB_EMPTY; 64]; 5];

#[inline]
pub fn pawn_pseudo_legal(square: Square, color: Color) -> Bitboard {
    unsafe {
        PAWN_ATTACKS[color as usize][square as usize]
    }
}

#[inline]
pub fn piece_pseudo_legal(square: Square, piece: Piece) -> Bitboard {
    unsafe {
        PIECE_PSEUDO_LEGALS_ATTACKS[piece as usize - 1][square as usize]
    }
}

#[cold]
pub fn init_pseudolegals() {
    unsafe {
        generate_pawn_attacks();
        generate_piece_pseudo_legals();
    }
}

#[cold]
unsafe fn generate_pawn_attacks() {
    PAWN_ATTACKS = BB_PAWN_ATTACKS;
}

#[cold]
unsafe fn generate_piece_pseudo_legals() {
    PIECE_PSEUDO_LEGALS_ATTACKS[KING as usize - 1] = BB_KING_ATTACKS;
    PIECE_PSEUDO_LEGALS_ATTACKS[KNIGHT as usize - 1] = BB_KNIGHT_ATTACKS;
    for sq in SQUARES.iter() {
        let rook_att = Bitboard::calc_orth_attacks(*sq, BB_EMPTY);
        let bishop_att = Bitboard::calc_diag_attacks(*sq, BB_EMPTY);
        PIECE_PSEUDO_LEGALS_ATTACKS[BISHOP as usize - 1][*sq as usize] = bishop_att;
        PIECE_PSEUDO_LEGALS_ATTACKS[ROOK as usize - 1][*sq as usize] = rook_att;
        PIECE_PSEUDO_LEGALS_ATTACKS[QUEEN as usize - 1][*sq as usize] = rook_att | bishop_att;
    }
}