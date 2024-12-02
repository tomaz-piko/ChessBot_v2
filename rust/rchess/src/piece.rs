#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use std::str::FromStr;
use crate::bitboard::Bitboard;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl FromStr for Piece {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "p" => Ok(Piece::Pawn),
            "n" => Ok(Piece::Knight),
            "b" => Ok(Piece::Bishop),
            "r" => Ok(Piece::Rook),
            "q" => Ok(Piece::Queen),
            "k" => Ok(Piece::King),
            _ => Err(()),
        }
    }
}

impl Index<Piece> for [Bitboard; 6] {
    type Output = Bitboard;

    fn index(&self, index: Piece) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<Piece> for [Bitboard; 6] {
    fn index_mut(&mut self, index: Piece) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

pub const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

pub const PAWN: Piece = Piece::Pawn;
pub const KNIGHT: Piece = Piece::Knight;
pub const BISHOP: Piece = Piece::Bishop;
pub const ROOK: Piece = Piece::Rook;
pub const QUEEN: Piece = Piece::Queen;
pub const KING: Piece = Piece::King;
