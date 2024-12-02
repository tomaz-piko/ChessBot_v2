pub mod magics;
mod btwn;
mod lines;
mod pseudolegals;
mod zobrist;

use std::sync::Once;
use crate::bitboard::Bitboard;
use crate::board::CastlingRights;
use crate::color::Color;
use crate::piece::Piece;
use crate::square::Square;

static INIT: Once = Once::new();

#[cold]
fn init_statics() {
    INIT.call_once(|| {
        magics::init_magics();
        btwn::init_squares_between();
        lines::init_lines_between();
        pseudolegals::init_pseudolegals();
        zobrist::init_zobrist();
    });
}

#[derive(Copy, Clone)]
pub struct Lookups {}

unsafe impl Send for Lookups {}
unsafe impl Sync for Lookups {}

impl Default for Lookups {
    fn default() -> Self {
        Lookups {}
    }
}

impl Lookups {
    pub fn new() -> Lookups {
        init_statics();
        Lookups {}
    }

    #[inline(always)]
    pub fn bishop_attacks(&self, sq: Square, occ: Bitboard) -> Bitboard {
        magics::bishop_attacks(sq as usize, occ)
    }

    #[inline(always)]
    pub fn rook_attacks(&self, sq: Square, occ: Bitboard) -> Bitboard {
        magics::rook_attacks(sq as usize, occ)
    }

    #[inline(always)]
    pub fn squares_between(&self, sq1: Square, sq2: Square) -> Bitboard {
        btwn::squares_between(sq1, sq2)
    }

    #[inline(always)]
    pub fn line_through(&self, sq1: Square, sq2: Square) -> Bitboard {
        lines::line_through(sq1, sq2)
    }

    #[inline(always)]
    pub fn pawn_pseudo_legal(&self, sq: Square, color: Color) -> Bitboard {
        pseudolegals::pawn_pseudo_legal(sq, color)
    }

    #[inline(always)]
    pub fn piece_pseudo_legal(&self, sq: Square, piece: Piece) -> Bitboard {
        pseudolegals::piece_pseudo_legal(sq, piece)
    }

    #[inline(always)]
    pub fn zobrist_piece(&self, piece: Piece, sq: Square) -> u64 {
        zobrist::zobrist_piece(piece, sq)
    }

    #[inline(always)]
    pub fn zobrist_black(&self) -> u64 {
        zobrist::zobrist_black()
    }

    #[inline(always)]
    pub fn zobrist_ep(&self, ep: Square) -> u64 {
        zobrist::zobrist_ep(ep)
    }

    #[inline(always)]
    pub fn zobrist_castling(&self, castling: CastlingRights) -> u64 {
        zobrist::zobrist_castling(castling)
    }
}
