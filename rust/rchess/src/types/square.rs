use crate::bitboard::Bitboard;
use crate::types::color::Color;
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

impl TryFrom<&str> for Square {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "a1" => Ok(Square::A1),
            "b1" => Ok(Square::B1),
            "c1" => Ok(Square::C1),
            "d1" => Ok(Square::D1),
            "e1" => Ok(Square::E1),
            "f1" => Ok(Square::F1),
            "g1" => Ok(Square::G1),
            "h1" => Ok(Square::H1),
            "a2" => Ok(Square::A2),
            "b2" => Ok(Square::B2),
            "c2" => Ok(Square::C2),
            "d2" => Ok(Square::D2),
            "e2" => Ok(Square::E2),
            "f2" => Ok(Square::F2),
            "g2" => Ok(Square::G2),
            "h2" => Ok(Square::H2),
            "a3" => Ok(Square::A3),
            "b3" => Ok(Square::B3),
            "c3" => Ok(Square::C3),
            "d3" => Ok(Square::D3),
            "e3" => Ok(Square::E3),
            "f3" => Ok(Square::F3),
            "g3" => Ok(Square::G3),
            "h3" => Ok(Square::H3),
            "a4" => Ok(Square::A4),
            "b4" => Ok(Square::B4),
            "c4" => Ok(Square::C4),
            "d4" => Ok(Square::D4),
            "e4" => Ok(Square::E4),
            "f4" => Ok(Square::F4),
            "g4" => Ok(Square::G4),
            "h4" => Ok(Square::H4),
            "a5" => Ok(Square::A5),
            "b5" => Ok(Square::B5),
            "c5" => Ok(Square::C5),
            "d5" => Ok(Square::D5),
            "e5" => Ok(Square::E5),
            "f5" => Ok(Square::F5),
            "g5" => Ok(Square::G5),
            "h5" => Ok(Square::H5),
            "a6" => Ok(Square::A6),
            "b6" => Ok(Square::B6),
            "c6" => Ok(Square::C6),
            "d6" => Ok(Square::D6),
            "e6" => Ok(Square::E6),
            "f6" => Ok(Square::F6),
            "g6" => Ok(Square::G6),
            "h6" => Ok(Square::H6),
            "a7" => Ok(Square::A7),
            "b7" => Ok(Square::B7),
            "c7" => Ok(Square::C7),
            "d7" => Ok(Square::D7),
            "e7" => Ok(Square::E7),
            "f7" => Ok(Square::F7),
            "g7" => Ok(Square::G7),
            "h7" => Ok(Square::H7),
            "a8" => Ok(Square::A8),
            "b8" => Ok(Square::B8),
            "c8" => Ok(Square::C8),
            "d8" => Ok(Square::D8),
            "e8" => Ok(Square::E8),
            "f8" => Ok(Square::F8),
            "g8" => Ok(Square::G8),
            "h8" => Ok(Square::H8),
            _ => Err("Invalid square name"),
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
