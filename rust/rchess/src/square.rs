use crate::bitboard::Bitboard;
use crate::color::Color;
use std::fmt::Display;
use std::ops::{Add, BitAnd, Index, Sub};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub enum Square {
    A1 = 0,
    B1,
    C1,
    D1,
    E1,
    F1,
    G1,
    H1,
    A2,
    B2,
    C2,
    D2,
    E2,
    F2,
    G2,
    H2,
    A3,
    B3,
    C3,
    D3,
    E3,
    F3,
    G3,
    H3,
    A4,
    B4,
    C4,
    D4,
    E4,
    F4,
    G4,
    H4,
    A5,
    B5,
    C5,
    D5,
    E5,
    F5,
    G5,
    H5,
    A6,
    B6,
    C6,
    D6,
    E6,
    F6,
    G6,
    H6,
    A7,
    B7,
    C7,
    D7,
    E7,
    F7,
    G7,
    H7,
    A8,
    B8,
    C8,
    D8,
    E8,
    F8,
    G8,
    H8,
}

pub const SQUARES: [Square; 64] = [
    A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, C2, D2, E2, F2, G2, H2, A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4, A5, B5, C5, D5, E5, F5, G5, H5, A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7, A8, B8, C8, D8, E8, F8, G8, H8,
];

pub const SQUARES_REV: [Square; 64] = [
    A8, B8, C8, D8, E8, F8, G8, H8, A7, B7, C7, D7, E7, F7, G7, H7, A6, B6, C6, D6, E6, F6, G6, H6,
    A5, B5, C5, D5, E5, F5, G5, H5, A4, B4, C4, D4, E4, F4, G4, H4, A3, B3, C3, D3, E3, F3, G3, H3,
    A2, B2, C2, D2, E2, F2, G2, H2, A1, B1, C1, D1, E1, F1, G1, H1,
];

impl Square {
    pub fn file_idx(&self) -> usize {
        *self as usize % 8
    }

    pub fn rank_idx(&self) -> usize {
        *self as usize / 8
    }

    pub fn diag_l_idx(&self) -> usize {
        self.file_idx() + self.rank_idx()
    }

    pub fn diag_r_idx(&self) -> usize {
        7 - self.file_idx() + self.rank_idx()
    }
}

impl From<u16> for Square {
    fn from(value: u16) -> Self {
        assert!(value < 64);
        SQUARES[value as usize]
    }
}

impl Add<i8> for Square {
    type Output = Square;

    fn add(self, rhs: i8) -> Self::Output {
        let idx = self as i8 + rhs;
        assert!((0..64).contains(&idx), "Invalid square index {}", idx);
        SQUARES[idx as usize]
    }
}

impl Sub<i8> for Square {
    type Output = Square;

    fn sub(self, rhs: i8) -> Self::Output {
        let idx = self as i8 - rhs;
        assert!((0..64).contains(&idx), "Invalid square index {}", idx);
        SQUARES[idx as usize]
    }
}

impl Index<Square> for [Bitboard; 64] {
    type Output = Bitboard;

    fn index(&self, index: Square) -> &Self::Output {
        &self[index as usize]
    }
}

impl BitAnd<Color> for Square {
    type Output = Square;

    fn bitand(self, rhs: Color) -> Self::Output {
        match rhs {
            Color::White => self,
            Color::Black => SQUARES_REV[self as usize],
        }
    }
}

#[test]
fn test_square_and_color() {
    assert_eq!(A1 & Color::White, A1);
    assert_eq!(A1 & Color::Black, A8);
    assert_eq!(H1 & Color::White, H1);
    assert_eq!(H1 & Color::Black, H8);
    assert_eq!(E1 & Color::Black, E8);
    assert_eq!(C1 & Color::Black, C8);
}

pub const A1: Square = Square::A1;
pub const B1: Square = Square::B1;
pub const C1: Square = Square::C1;
pub const D1: Square = Square::D1;
pub const E1: Square = Square::E1;
pub const F1: Square = Square::F1;
pub const G1: Square = Square::G1;
pub const H1: Square = Square::H1;
pub const A2: Square = Square::A2;
pub const B2: Square = Square::B2;
pub const C2: Square = Square::C2;
pub const D2: Square = Square::D2;
pub const E2: Square = Square::E2;
pub const F2: Square = Square::F2;
pub const G2: Square = Square::G2;
pub const H2: Square = Square::H2;
pub const A3: Square = Square::A3;
pub const B3: Square = Square::B3;
pub const C3: Square = Square::C3;
pub const D3: Square = Square::D3;
pub const E3: Square = Square::E3;
pub const F3: Square = Square::F3;
pub const G3: Square = Square::G3;
pub const H3: Square = Square::H3;
pub const A4: Square = Square::A4;
pub const B4: Square = Square::B4;
pub const C4: Square = Square::C4;
pub const D4: Square = Square::D4;
pub const E4: Square = Square::E4;
pub const F4: Square = Square::F4;
pub const G4: Square = Square::G4;
pub const H4: Square = Square::H4;
pub const A5: Square = Square::A5;
pub const B5: Square = Square::B5;
pub const C5: Square = Square::C5;
pub const D5: Square = Square::D5;
pub const E5: Square = Square::E5;
pub const F5: Square = Square::F5;
pub const G5: Square = Square::G5;
pub const H5: Square = Square::H5;
pub const A6: Square = Square::A6;
pub const B6: Square = Square::B6;
pub const C6: Square = Square::C6;
pub const D6: Square = Square::D6;
pub const E6: Square = Square::E6;
pub const F6: Square = Square::F6;
pub const G6: Square = Square::G6;
pub const H6: Square = Square::H6;
pub const A7: Square = Square::A7;
pub const B7: Square = Square::B7;
pub const C7: Square = Square::C7;
pub const D7: Square = Square::D7;
pub const E7: Square = Square::E7;
pub const F7: Square = Square::F7;
pub const G7: Square = Square::G7;
pub const H7: Square = Square::H7;
pub const A8: Square = Square::A8;
pub const B8: Square = Square::B8;
pub const C8: Square = Square::C8;
pub const D8: Square = Square::D8;
pub const E8: Square = Square::E8;
pub const F8: Square = Square::F8;
pub const G8: Square = Square::G8;
pub const H8: Square = Square::H8;

// TODO: Switch to try_from & implement and error type
impl From<&str> for Square {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "a1" => Square::A1,
            "b1" => Square::B1,
            "c1" => Square::C1,
            "d1" => Square::D1,
            "e1" => Square::E1,
            "f1" => Square::F1,
            "g1" => Square::G1,
            "h1" => Square::H1,
            "a2" => Square::A2,
            "b2" => Square::B2,
            "c2" => Square::C2,
            "d2" => Square::D2,
            "e2" => Square::E2,
            "f2" => Square::F2,
            "g2" => Square::G2,
            "h2" => Square::H2,
            "a3" => Square::A3,
            "b3" => Square::B3,
            "c3" => Square::C3,
            "d3" => Square::D3,
            "e3" => Square::E3,
            "f3" => Square::F3,
            "g3" => Square::G3,
            "h3" => Square::H3,
            "a4" => Square::A4,
            "b4" => Square::B4,
            "c4" => Square::C4,
            "d4" => Square::D4,
            "e4" => Square::E4,
            "f4" => Square::F4,
            "g4" => Square::G4,
            "h4" => Square::H4,
            "a5" => Square::A5,
            "b5" => Square::B5,
            "c5" => Square::C5,
            "d5" => Square::D5,
            "e5" => Square::E5,
            "f5" => Square::F5,
            "g5" => Square::G5,
            "h5" => Square::H5,
            "a6" => Square::A6,
            "b6" => Square::B6,
            "c6" => Square::C6,
            "d6" => Square::D6,
            "e6" => Square::E6,
            "f6" => Square::F6,
            "g6" => Square::G6,
            "h6" => Square::H6,
            "a7" => Square::A7,
            "b7" => Square::B7,
            "c7" => Square::C7,
            "d7" => Square::D7,
            "e7" => Square::E7,
            "f7" => Square::F7,
            "g7" => Square::G7,
            "h7" => Square::H7,
            "a8" => Square::A8,
            "b8" => Square::B8,
            "c8" => Square::C8,
            "d8" => Square::D8,
            "e8" => Square::E8,
            "f8" => Square::F8,
            "g8" => Square::G8,
            "h8" => Square::H8,
            _ => panic!("Invalid square name {}", value),
        }
    }
}

impl Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let file = self.file_idx();
        let rank = self.rank_idx() + 1;
        write!(f, "{}{}", (b'a' + file as u8) as char, rank)
    }
}
