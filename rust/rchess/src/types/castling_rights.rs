use std::fmt::Display;
use std::ops::Index;
use crate::types::color::Color;

pub const NONE: u8 = 0b0000;
pub const WHITE_OO: u8 = 0b1000;
pub const WHITE_OOO: u8 = 0b0100;
pub const BLACK_OO: u8 = 0b0010;
pub const BLACK_OOO: u8 = 0b0001;
pub const ALL: u8 = 0b1111;

pub enum CastlingSide {
    KingSide,
    QueenSide,
}

#[derive(Clone, Copy)]
pub struct CastlingRights(u8);

impl Default for CastlingRights {
    fn default() -> Self {
        CastlingRights(ALL)
    }
}

impl Index<CastlingRights> for [u64; 16] {
    type Output = u64;

    fn index(&self, index: CastlingRights) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl CastlingRights {
    pub fn new() -> Self {
        CastlingRights(NONE)
    }

    pub fn is_empty(&self) -> bool {
        self.0 == NONE
    }

    pub fn has_kingside_rights(&self, player: Color) -> bool {
        match player {
            Color::White => self.0 & WHITE_OO != 0,
            Color::Black => self.0 & BLACK_OO != 0,
        }
    }

    pub fn remove_kingside_rights(&mut self, player: Color) {
        match player {
            Color::White => self.0 &= !WHITE_OO,
            Color::Black => self.0 &= !BLACK_OO,
        }
    }

    pub fn has_queenside_rights(&self, player: Color) -> bool {
        match player {
            Color::White => self.0 & WHITE_OOO != 0,
            Color::Black => self.0 & BLACK_OOO != 0,
        }
    }

    pub fn remove_queenside_rights(&mut self, player: Color) {
        match player {
            Color::White => self.0 &= !WHITE_OOO,
            Color::Black => self.0 &= !BLACK_OOO,
        }
    }

    pub fn remove_both_rights(&mut self, player: Color) {
        match player {
            Color::White => self.0 &= BLACK_OO | BLACK_OOO,
            Color::Black => self.0 &= WHITE_OO | WHITE_OOO,
        }
    }

    pub fn add_rights(&mut self, player: Color, side: CastlingSide) {
        match player {
            Color::White => match side {
                CastlingSide::KingSide => self.0 |= WHITE_OO,
                CastlingSide::QueenSide => self.0 |= WHITE_OOO,
            },
            Color::Black => match side {
                CastlingSide::KingSide => self.0 |= BLACK_OO,
                CastlingSide::QueenSide => self.0 |= BLACK_OOO,
            },
        }
    }
}

impl From<&str> for CastlingRights {
    fn from(s: &str) -> Self {
        let mut cr = CastlingRights::new();
        for c in s.chars() {
            match c {
                'K' => cr.add_rights(Color::White, CastlingSide::KingSide),
                'Q' => cr.add_rights(Color::White, CastlingSide::QueenSide),
                'k' => cr.add_rights(Color::Black, CastlingSide::KingSide),
                'q' => cr.add_rights(Color::Black, CastlingSide::QueenSide),
                '-' => return cr,
                _ => panic!("Invalid castling rights string"),
            }
        }
        cr
    }
}

impl Display for CastlingRights {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = String::new();
        if self.0 == NONE {
            s.push('-');
            return write!(f, "{}", s);
        }
        if self.has_kingside_rights(Color::White) {
            s.push('K');
        }
        if self.has_queenside_rights(Color::White) {
            s.push('Q');
        }
        if self.has_kingside_rights(Color::Black) {
            s.push('k');
        }
        if self.has_queenside_rights(Color::Black) {
            s.push('q');
        }
        write!(f, "{}", s)
    }
}