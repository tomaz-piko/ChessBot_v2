use std::fmt::Debug;
use std::ops::{BitAnd, Mul};

use crate::types::square::{Square, SQUARES_REV};

pub mod masks;
mod ops;

use crate::board::{AFILE, HFILE};
use crate::types::color::Color;
use masks::*;

#[derive(Debug, PartialEq)]
pub enum Direction {
    North = 8,
    North2x = 16,
    South = -8,
    South2x = -16,
    East = 1,
    West = -1,
    NorthEast = 9,
    NorthWest = 7,
    SouthEast = -7,
    SouthWest = -9,
}

impl From<i8> for Direction {
    fn from(val: i8) -> Self {
        match val {
            8 => Direction::North,
            16 => Direction::North2x,
            -8 => Direction::South,
            -16 => Direction::South2x,
            1 => Direction::East,
            -1 => Direction::West,
            9 => Direction::NorthEast,
            7 => Direction::NorthWest,
            -7 => Direction::SouthEast,
            -9 | _ => Direction::SouthWest,
        }
    }
}

impl BitAnd<Color> for Direction {
    type Output = Direction;

    fn bitand(self, rhs: Color) -> Self::Output {
        match rhs {
            Color::White => self,
            Color::Black => Direction::from(-(self as i8)),
        }
    }
}

impl BitAnd<&Color> for Direction {
    type Output = Direction;

    fn bitand(self, rhs: &Color) -> Self::Output {
        match rhs {
            Color::White => self,
            Color::Black => Direction::from(-(self as i8)),
        }
    }
}

impl Mul<Color> for Direction {
    type Output = i8;

    fn mul(self, rhs: Color) -> Self::Output {
        match rhs {
            Color::White => self as i8,
            Color::Black => -(self as i8),
        }
    }
}

impl Mul<&Color> for Direction {
    type Output = i8;

    fn mul(self, rhs: &Color) -> Self::Output {
        match rhs {
            Color::White => self as i8,
            Color::Black => -(self as i8),
        }
    }
}

pub const NORTH: Direction = Direction::North;
pub const NORTH2X: Direction = Direction::North2x;
pub const SOUTH: Direction = Direction::South;
pub const SOUTH2X: Direction = Direction::South2x;
pub const EAST: Direction = Direction::East;
pub const WEST: Direction = Direction::West;
pub const NORTH_EAST: Direction = Direction::NorthEast;
pub const NORTH_WEST: Direction = Direction::NorthWest;
pub const SOUTH_EAST: Direction = Direction::SouthEast;
pub const SOUTH_WEST: Direction = Direction::SouthWest;

#[cfg(test)]
mod direction_tests {
    use super::*;

    #[test]
    fn direction_bitand_color() {
        // Up & down
        assert_eq!(Direction::North & Color::White, Direction::North);
        assert_eq!(Direction::North & Color::Black, Direction::South);
        assert_eq!(Direction::South & Color::White, Direction::South);
        assert_eq!(Direction::South & Color::Black, Direction::North);
        // Left & right
        assert_eq!(Direction::East & Color::White, Direction::East);
        assert_eq!(Direction::East & Color::Black, Direction::West);
        assert_eq!(Direction::West & Color::White, Direction::West);
        assert_eq!(Direction::West & Color::Black, Direction::East);
        // Diagonals
        assert_eq!(Direction::NorthEast & Color::White, Direction::NorthEast);
        assert_eq!(Direction::NorthEast & Color::Black, Direction::SouthWest);
        assert_eq!(Direction::NorthWest & Color::White, Direction::NorthWest);
        assert_eq!(Direction::NorthWest & Color::Black, Direction::SouthEast);
        assert_eq!(Direction::SouthEast & Color::White, Direction::SouthEast);
    }

    #[test]
    fn direction_mul_color() {
        // Up & down
        assert_eq!(Direction::North * Color::White, 8);
        assert_eq!(Direction::North * Color::Black, -8);
        assert_eq!(Direction::South * Color::White, -8);
        assert_eq!(Direction::South * Color::Black, 8);
        // Left & right
        assert_eq!(Direction::East * Color::White, 1);
        assert_eq!(Direction::East * Color::Black, -1);
        assert_eq!(Direction::West * Color::White, -1);
        assert_eq!(Direction::West * Color::Black, 1);
        // Diagonals
        assert_eq!(Direction::NorthEast * Color::White, 9);
        assert_eq!(Direction::NorthEast * Color::Black, -9);
        assert_eq!(Direction::NorthWest * Color::White, 7);
        assert_eq!(Direction::NorthWest * Color::Black, -7);
    }
}

#[derive(Copy, Clone, PartialEq)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const fn init_const() -> Bitboard {
        Bitboard(0)
    }

    #[inline(always)]
    pub fn bsf(&self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let index = ((*self ^ (*self - 1)) * DEBRUIJN_MAGIC) >> 58;
        Some(Square::from(DEBRUIJN_64[index.0 as usize]))
    }

    #[inline(always)]
    pub fn mirror(&self) -> Bitboard {
        let mut m = self.0;
        m = ((m >> 1) & 0x5555_5555_5555_5555) | ((m & 0x5555_5555_5555_5555) << 1);
        m = ((m >> 2) & 0x3333_3333_3333_3333) | ((m & 0x3333_3333_3333_3333) << 2);
        m = ((m >> 4) & 0x0f0f_0f0f_0f0f_0f0f) | ((m & 0x0f0f_0f0f_0f0f_0f0f) << 4);
        Bitboard(m)
    }

    #[inline(always)]
    pub fn reverse(bb: Bitboard) -> Bitboard {
        let mut r = bb.0;
        r = (r & 0x5555555555555555) << 1 | ((r >> 1) & 0x5555555555555555);
        r = (r & 0x3333333333333333) << 2 | ((r >> 2) & 0x3333333333333333);
        r = (r & 0x0f0f0f0f0f0f0f0f) << 4 | ((r >> 4) & 0x0f0f0f0f0f0f0f0f);
        r = (r & 0x00ff00ff00ff00ff) << 8 | ((r >> 8) & 0x00ff00ff00ff00ff);
        r = (r << 48) | ((r & 0xffff0000) << 16) | ((r >> 16) & 0xffff0000) | (r >> 48);
        Bitboard(r)
    }

    #[inline(always)]
    pub fn shift(bb: Bitboard, dir: Direction) -> Bitboard {
        match dir {
            NORTH => bb << 8,
            SOUTH => bb >> 8,
            NORTH2X => bb << 16,
            SOUTH2X => bb >> 16,
            EAST => (bb & !BB_FILES[HFILE]) << 1,
            WEST => (bb & !BB_FILES[AFILE]) >> 1,
            NORTH_EAST => (bb & !BB_FILES[HFILE]) << 9,
            NORTH_WEST => (bb & !BB_FILES[AFILE]) << 7,
            SOUTH_EAST => (bb & !BB_FILES[HFILE]) >> 7,
            SOUTH_WEST => (bb & !BB_FILES[AFILE]) >> 9,
        }
    }

    #[inline(always)]
    pub fn squares_between(sq1: Square, sq2: Square) -> Bitboard {
        let btwn = (BB_FULL << sq1 as usize) ^ (BB_FULL << sq2 as usize);
        let file = Bitboard((sq2.file_idx().wrapping_sub(sq1.file_idx())) as u64);
        let rank = Bitboard((sq2.rank_idx().wrapping_sub(sq1.rank_idx())) as u64);
        let mut line = ((file & 7) - 1) & A2A7;
        line += 2 * (((rank & 7) - 1) >> 58);
        line += (((rank - file) & 15) - 1) & B2G7;
        line += (((rank + file) & 15) - 1) & H1B7;
        line *= btwn & (-btwn);
        line & btwn
    }

    #[inline(always)]
    pub fn line_through(sq1: Square, sq2: Square) -> Bitboard {
        let (sq1_rank, sq1_file) = (sq1.rank_idx(), sq1.file_idx());
        let (sq2_rank, sq2_file) = (sq2.rank_idx(), sq2.file_idx());
        if sq1_rank == sq2_rank || sq1_file == sq2_file {
            // Horizontal or vertical line
            (BB_RANKS[sq1_rank] | BB_FILES[sq1_file]) & (BB_RANKS[sq2_rank] | BB_FILES[sq2_file])
        } else {
            // Diagonal line
            (BB_DIAGONAL_L[sq1.diag_l_idx()] & BB_DIAGONAL_L[sq2.diag_l_idx()])
                | (BB_DIAGONAL_R[sq1.diag_r_idx()] & BB_DIAGONAL_R[sq2.diag_r_idx()])
        }
    }

    #[inline(always)]
    pub fn pop_count(&self) -> u32 {
        self.0.count_ones()
    }
}

impl Bitboard {
    #[inline(always)]
    pub fn sliding_attacks(sq: Square, occ: Bitboard, mask: Bitboard) -> Bitboard {
        (((occ & mask) - BB_SQUARES[sq] * 2)
            ^ Bitboard::reverse(
                Bitboard::reverse(occ & mask) - Bitboard::reverse(BB_SQUARES[sq]) * 2,
            ))
            & mask
    }

    #[inline(always)]
    pub fn calc_orth_attacks(sq: Square, occ: Bitboard) -> Bitboard {
        Bitboard::sliding_attacks(sq, occ, BB_RANKS[sq.rank_idx()])
            | Bitboard::sliding_attacks(sq, occ, BB_FILES[sq.file_idx()])
    }

    #[inline(always)]
    pub fn calc_diag_attacks(sq: Square, occ: Bitboard) -> Bitboard {
        Bitboard::sliding_attacks(sq, occ, BB_DIAGONAL_L[sq.diag_l_idx()])
            | Bitboard::sliding_attacks(sq, occ, BB_DIAGONAL_R[sq.diag_r_idx()])
    }
}

impl Default for Bitboard {
    fn default() -> Self {
        Bitboard(0)
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        return self.bsf().and_then(|sq| {
            self.0 &= self.0 - 1;
            Some(sq)
        });
    }
}

impl Debug for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut row: usize = 0;
        const ROWS_LABELS: [&str; 8] = [
            "8 | ", "7 | ", "6 | ", "5 | ", "4 | ", "3 | ", "2 | ", "1 | ",
        ];
        let mut board = String::new();
        board.push_str("- | --------------- |\n");
        for sq in SQUARES_REV.iter() {
            if *sq as i32 % 8 == 0 {
                board.push_str(ROWS_LABELS[row]);
                row += 1;
            }
            if *self & BB_SQUARES[*sq] != BB_EMPTY {
                board.push_str("1 ");
            } else {
                board.push_str(". ");
            }
            if *sq as i32 % 8 == 7 {
                board.push_str("|\n");
            }
        }
        board.push_str("- | --------------- |\n");
        board.push_str("  | a b c d e f g h |\n");
        write!(f, "{}", board)
    }
}

#[cfg(test)]
mod bitboard_tests {
    use super::*;
    use crate::board::File::{BFILE, GFILE};
    use crate::board::Rank::{RANK2, RANK3, RANK6, RANK7};
    use crate::types::color::{BLACK, WHITE};

    #[test]
    fn shift_north() {
        let bb = Bitboard(0x0000_0000_0000_00FF);
        let expected = Bitboard(0x0000_0000_0000_FF00);
        assert_eq!(Bitboard::shift(bb, NORTH & WHITE), expected);
        assert_eq!(bb, Bitboard(0x0000_0000_0000_00FF));
        let bb2 = Bitboard(0xFF00_0000_0000_0000);
        let expected = Bitboard(0x00FF_0000_0000_0000);
        assert_eq!(Bitboard::shift(bb2, NORTH & BLACK), expected);
        assert_eq!(bb2, Bitboard(0xFF00_0000_0000_0000));
    }

    #[test]
    fn shift_south() {
        let bb = Bitboard(0x0000_0000_0000_FF00);
        let expected = Bitboard(0x0000_0000_0000_00FF);
        assert_eq!(Bitboard::shift(bb, SOUTH & WHITE), expected);
        assert_eq!(bb, Bitboard(0x0000_0000_0000_FF00));
        let bb2 = Bitboard(0x00FF_0000_0000_0000);
        let expected = Bitboard(0xFF00_0000_0000_0000);
        assert_eq!(Bitboard::shift(bb2, SOUTH & BLACK), expected);
        assert_eq!(bb2, Bitboard(0x00FF_0000_0000_0000));
    }

    #[test]
    fn shift_east() {
        let a_file = BB_FILES[AFILE];
        let expected = BB_FILES[BFILE];
        assert_eq!(Bitboard::shift(a_file, EAST & WHITE), expected);
        let h_file = BB_FILES[HFILE];
        let expected = BB_FILES[GFILE];
        assert_eq!(Bitboard::shift(h_file, EAST & BLACK), expected);
    }

    #[test]
    fn shift_west() {
        let b_file = BB_FILES[BFILE];
        let expected = BB_FILES[AFILE];
        assert_eq!(Bitboard::shift(b_file, WEST & WHITE), expected);
        let g_file = BB_FILES[GFILE];
        let expected = BB_FILES[HFILE];
        assert_eq!(Bitboard::shift(g_file, WEST & BLACK), expected);
    }

    #[test]
    fn shift_north_east() {
        let starting_pawns = BB_RANKS[RANK2];
        let expected = BB_RANKS[RANK3] ^ BB_SQUARES[Square::A3];
        assert_eq!(
            Bitboard::shift(starting_pawns, NORTH_EAST & WHITE),
            expected
        );
        let starting_pawns = BB_RANKS[RANK7];
        let expected = BB_RANKS[RANK6] ^ BB_SQUARES[Square::H6];
        assert_eq!(
            Bitboard::shift(starting_pawns, NORTH_EAST & BLACK),
            expected
        );
    }

    #[test]
    fn shift_north_west() {
        let starting_pawns = BB_RANKS[RANK2];
        let expected = BB_RANKS[RANK3] ^ BB_SQUARES[Square::H3];
        assert_eq!(
            Bitboard::shift(starting_pawns, NORTH_WEST & WHITE),
            expected
        );
        let starting_pawns = BB_RANKS[RANK7];
        let expected = BB_RANKS[RANK6] ^ BB_SQUARES[Square::A6];
        assert_eq!(
            Bitboard::shift(starting_pawns, NORTH_WEST & BLACK),
            expected
        );
    }
}
