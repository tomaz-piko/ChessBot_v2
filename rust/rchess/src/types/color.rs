use std::ops::{Index, IndexMut, Not};
use crate::bitboard::Bitboard;
use crate::types::square::Square;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    Black = 0,
    White = 1,
}

pub const WHITE: Color = Color::White;
pub const BLACK: Color = Color::Black;

impl Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        match self {
            WHITE => BLACK,
            BLACK => WHITE,
        }
    }
}

impl Index<Color> for [Bitboard; 2] {
    type Output = Bitboard;

    fn index(&self, index: Color) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<Color> for [Bitboard; 2] {
    fn index_mut(&mut self, index: Color) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

impl Index<Color> for [Square; 2] {
    type Output = Square;

    fn index(&self, index: Color) -> &Self::Output {
        &self[index as usize]
    }
}

#[test]
fn test_color_not() {
    assert_eq!(WHITE, !BLACK);
    assert_eq!(BLACK, !WHITE);
}

#[test]
fn test_index() {
    let bbs: [Bitboard; 2] = [Bitboard(1), Bitboard(0)];
    assert_eq!(Bitboard(0), bbs[Color::White as usize]); // White == 1 == true
    assert_eq!(Bitboard(1), bbs[Color::Black as usize]); // Black == 0 == false
}