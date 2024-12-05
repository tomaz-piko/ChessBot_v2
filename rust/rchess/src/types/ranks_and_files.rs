#![allow(dead_code)]
use std::ops::{BitAnd, Index};
use crate::bitboard::Bitboard;
use crate::types::color::Color;

pub enum Rank {
    RANK1 = 0,
    RANK2 = 1,
    RANK3 = 2,
    RANK4 = 3,
    RANK5 = 4,
    RANK6 = 5,
    RANK7 = 6,
    RANK8 = 7,
}

impl From<i8> for Rank {
    fn from(rank: i8) -> Self {
        match rank {
            0 => Rank::RANK1,
            1 => Rank::RANK2,
            2 => Rank::RANK3,
            3 => Rank::RANK4,
            4 => Rank::RANK5,
            5 => Rank::RANK6,
            6 => Rank::RANK7,
            7 | _ => Rank::RANK8,
        }
    }
}

impl BitAnd<Color> for Rank {
    type Output = Rank;

    fn bitand(self, rhs: Color) -> Self::Output {
        match rhs {
            Color::White => self,
            Color::Black => Rank::from(7 - self as i8),
        }
    }
}

pub const RANK1: Rank = Rank::RANK1;
pub const RANK2: Rank = Rank::RANK2;
pub const RANK3: Rank = Rank::RANK3;
pub const RANK4: Rank = Rank::RANK4;
pub const RANK5: Rank = Rank::RANK5;
pub const RANK6: Rank = Rank::RANK6;
pub const RANK7: Rank = Rank::RANK7;
pub const RANK8: Rank = Rank::RANK8;

impl Index<Rank> for [Bitboard; 8] {
    type Output = Bitboard;

    fn index(&self, index: Rank) -> &Self::Output {
        &self[index as usize]
    }
}

pub enum File {
    AFILE = 0,
    BFILE = 1,
    CFILE = 2,
    DFILE = 3,
    EFILE = 4,
    FFILE = 5,
    GFILE = 6,
    HFILE = 7,
}

impl From<i8> for File {
    fn from(file: i8) -> Self {
        match file {
            0 => File::AFILE,
            1 => File::BFILE,
            2 => File::CFILE,
            3 => File::DFILE,
            4 => File::EFILE,
            5 => File::FFILE,
            6 => File::GFILE,
            7 => File::HFILE,
            _ => panic!(),
        }
    }
}

impl BitAnd<Color> for File {
    type Output = File;

    fn bitand(self, rhs: Color) -> Self::Output {
        match rhs {
            Color::White => self,
            Color::Black => File::from(7 - self as i8),
        }
    }
}

pub const AFILE: File = File::AFILE;
pub const BFILE: File = File::BFILE;
pub const CFILE: File = File::CFILE;
pub const DFILE: File = File::DFILE;
pub const EFILE: File = File::EFILE;
pub const FFILE: File = File::FFILE;
pub const GFILE: File = File::GFILE;
pub const HFILE: File = File::HFILE;

impl Index<File> for [Bitboard; 8] {
    type Output = Bitboard;

    fn index(&self, index: File) -> &Self::Output {
        &self[index as usize]
    }
}