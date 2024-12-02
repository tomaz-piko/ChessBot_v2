#![allow(unused_variables)]
use crate::bitboard::masks::*;
use crate::bitboard::{Bitboard, NORTH, NORTH2X, NORTH_EAST, NORTH_WEST, SOUTH};
use crate::color::{Color, BLACK, WHITE};
use crate::piece::{Piece, PIECES};
use crate::r#move::{Flags, Move};
use crate::square::{Square, SQUARES, SQUARES_REV};
use crate::statics::Lookups;
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::ops::{BitAnd, Index};
use std::str::FromStr;

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

#[derive(Clone, Copy)]
pub enum Outcome {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    FiftyMoveRule,
    ThreeFoldRepetition,
}

pub enum CastlingSide {
    KingSide,
    QueenSide,
}

const NONE: u8 = 0b0000;
const WHITE_OO: u8 = 0b1000;
const WHITE_OOO: u8 = 0b0100;
const BLACK_OO: u8 = 0b0010;
const BLACK_OOO: u8 = 0b0001;
const ALL: u8 = 0b1111;

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
    fn new() -> Self {
        CastlingRights(NONE)
    }

    fn is_empty(&self) -> bool {
        self.0 == NONE
    }

    fn has_kingside_rights(&self, player: Color) -> bool {
        match player {
            Color::White => self.0 & WHITE_OO != 0,
            Color::Black => self.0 & BLACK_OO != 0,
        }
    }

    fn remove_kingside_rights(&mut self, player: Color) {
        match player {
            Color::White => self.0 &= !WHITE_OO,
            Color::Black => self.0 &= !BLACK_OO,
        }
    }

    fn has_queenside_rights(&self, player: Color) -> bool {
        match player {
            Color::White => self.0 & WHITE_OOO != 0,
            Color::Black => self.0 & BLACK_OOO != 0,
        }
    }

    fn remove_queenside_rights(&mut self, player: Color) {
        match player {
            Color::White => self.0 &= !WHITE_OOO,
            Color::Black => self.0 &= !BLACK_OOO,
        }
    }

    fn remove_both_rights(&mut self, player: Color) {
        match player {
            Color::White => self.0 &= BLACK_OO | BLACK_OOO,
            Color::Black => self.0 &= WHITE_OO | WHITE_OOO,
        }
    }

    fn add_rights(&mut self, player: Color, side: CastlingSide) {
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

const STARTING_POSITION: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";

#[derive(Clone)]
pub struct HistoryPlane {
    turn: Color,
    pieces: [Bitboard; 6],
    occupancy: [Bitboard; 2],
    repetition_count: u8,
}

impl HistoryPlane {
    pub fn color(&self) -> Color {
        self.turn
    }

    pub fn pieces_bb(&self, piece_type: Piece) -> Bitboard {
        self.pieces[piece_type]
    }

    pub fn occupancy_bb(&self, color: Color) -> Bitboard {
        self.occupancy[color]
    }

    pub fn repetition_count(&self) -> u8 {
        self.repetition_count
    }
}

pub struct Board {
    turn: Color,                            // Whose turn is it
    pieces_bb: [Bitboard; 6],               // Bitboards for each piece type
    occupancy_bb: [Bitboard; 2],            // Bitboards for each color
    pieces_list: [Option<Piece>; 64],       // Potential piece at each square
    piece_count: u8, // Number of pieces on the board. For draw by insufficient material and endgame tablebases
    castling_rights: CastlingRights, // Castling rights for each player
    ply: u16,        // Number of half-moves since start
    half_move_counter: u8, // Number of half moves since last capture or pawn move
    ep_sq: Option<Square>, // En-passant square
    history_planes: VecDeque<HistoryPlane>, // History planes
    zobrist_hash: u64, // Zobrist hash
    zobrist_history: Vec<u64>, // Zobrist history for repetition detection

    cached_legal_moves: Vec<Move>, // Legal moves for the current position
    cached_is_check: Option<bool>, // Is the current position a check

    outcome: Option<Outcome>, // Outcome of the game

    pub lookups: Lookups,
}

// // // // // // // // // // // // //
//                                  //
//   Setup & state vars & methods   //
//                                  //
// // // // // // // // // // // // //
impl Default for Board {
    fn default() -> Self {
        Board {
            pieces_bb: [BB_EMPTY; 6],
            occupancy_bb: [BB_EMPTY; 2],
            pieces_list: [None; 64],
            piece_count: 0,
            turn: Color::White,
            castling_rights: CastlingRights::default(),
            ply: 0,
            half_move_counter: 0,
            ep_sq: None,
            history_planes: VecDeque::with_capacity(7),
            zobrist_hash: 0,
            zobrist_history: Vec::new(),
            cached_legal_moves: Vec::new(),
            cached_is_check: None,
            outcome: None,
            lookups: Lookups::new(),
        }
    }
}

impl Clone for Board {
    fn clone(&self) -> Board {
        Board {
            turn: self.turn,
            pieces_bb: self.pieces_bb,
            occupancy_bb: self.occupancy_bb,
            pieces_list: self.pieces_list,
            piece_count: self.piece_count,
            castling_rights: self.castling_rights,
            ply: self.ply,
            half_move_counter: self.half_move_counter,
            ep_sq: self.ep_sq,
            history_planes: self.history_planes.clone(),
            zobrist_hash: self.zobrist_hash,
            zobrist_history: self.zobrist_history.clone(),
            cached_legal_moves: self.cached_legal_moves.clone(),
            cached_is_check: self.cached_is_check,
            outcome: None,
            lookups: self.lookups,
        }
    }
}

impl Board {
    pub fn new(fen: Option<&str>) -> Board {
        match fen {
            Some(fen) => Self::from_fen(fen),
            None => Self::from_fen(STARTING_POSITION),
        }
    }

    pub fn from_fen(fen: &str) -> Board {
        let mut board = Board::default();
        let fen_parts: Vec<&str> = fen.split_whitespace().collect();
        let pieces_setup = fen_parts[0];
        let rows: Vec<&str> = pieces_setup.split("/").collect();
        let mut square_idx: usize = 0;
        for row in rows.iter() {
            for c in row.chars() {
                if c.is_ascii_digit() {
                    let num: usize = c.to_digit(10).unwrap() as usize;
                    square_idx += num;
                    continue;
                }
                let color = if c.is_lowercase() { BLACK } else { WHITE };
                match Piece::from_str(c.to_string().as_str()) {
                    Ok(piece) => {
                        board.put_piece_at(SQUARES_REV[square_idx], piece, color);
                        square_idx += 1;
                        board.piece_count += 1;
                    }
                    Err(_) => {
                        println!("Error");
                    }
                }
            }
        }
        board.turn = match fen_parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            _ => WHITE,
        };
        board.castling_rights = CastlingRights::from(fen_parts[2]);
        board.ep_sq = match fen_parts[3] {
            "-" => None,
            _ => Some(Square::from(fen_parts[3])),
        };
        board.half_move_counter = fen_parts
            .get(4)
            .and_then(|x| x.parse::<u8>().ok())
            .unwrap_or(0);
        let full_move_counter = fen_parts
            .get(5)
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(1);
        board.ply = 2 * full_move_counter - if board.turn == BLACK { 1 } else { 2 };
        board
    }

    pub fn fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty_squares = 0;
            for file in 0..8 {
                let square = SQUARES[rank * 8 + file];
                match self.piece_at(square) {
                    Some(piece) => {
                        if empty_squares > 0 {
                            fen.push_str(&empty_squares.to_string());
                            empty_squares = 0;
                        }
                        let piece_char = match piece {
                            Piece::Pawn => 'p',
                            Piece::Knight => 'n',
                            Piece::Bishop => 'b',
                            Piece::Rook => 'r',
                            Piece::Queen => 'q',
                            Piece::King => 'k',
                        };
                        if self.occupancy_bb[WHITE] & BB_SQUARES[square] != BB_EMPTY {
                            fen.push(piece_char.to_ascii_uppercase());
                        } else {
                            fen.push(piece_char);
                        }
                    }
                    None => {
                        empty_squares += 1;
                    }
                }
            }
            if empty_squares > 0 {
                fen.push_str(&empty_squares.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push(' ');
        fen.push(match self.turn {
            WHITE => 'w',
            BLACK => 'b',
        });
        fen.push(' ');
        fen.push_str(&self.castling_rights.to_string());
        fen.push(' ');
        match self.ep_sq {
            Some(sq) => fen.push_str(&sq.to_string()),
            None => fen.push('-'),
        }
        fen.push(' ');
        fen.push_str(&self.half_move_counter.to_string());
        fen.push(' ');
        fen.push_str(&self.full_move_counter().to_string());
        fen
    }

    #[inline(always)]
    pub fn turn(&self) -> Color {
        self.turn
    }

    #[inline(always)]
    pub fn pieces_bb(&self, piece_type: Piece, color: Color) -> Bitboard {
        self.pieces_bb[piece_type] & self.occupancy_bb[color]
    }

    #[inline(always)]
    pub fn occupancy_bb(&self, color: Color) -> Bitboard {
        self.occupancy_bb[color]
    }

    #[inline(always)]
    pub fn pieces_list(&self) -> [Option<Piece>; 64] {
        self.pieces_list
    }

    #[inline(always)]
    pub fn piece_count(&self) -> u8 {
        self.piece_count
    }

    #[inline(always)]
    pub fn castling_rights(&self) -> String {
        self.castling_rights.to_string()
    }

    #[inline(always)]
    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.pieces_list[square as usize]
    }

    #[inline(always)]
    pub fn ply(&self) -> u16 {
        self.ply
    }

    #[inline(always)]
    pub fn full_move_counter(&self) -> u16 {
        self.ply / 2 + 1
    }

    #[inline(always)]
    pub fn half_move_counter(&self) -> u8 {
        self.half_move_counter
    }

    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep_sq
    }

    #[inline(always)]
    pub fn zobrist_hash(&self) -> u64 {
        self.zobrist_hash
    }

    #[inline(always)]
    pub fn history_planes(&self) -> &VecDeque<HistoryPlane> {
        &self.history_planes
    }

    #[inline(always)]
    pub fn make_image(&self) -> Vec<Bitboard> {
        let us: Color = self.turn;
        let mut image: Vec<Bitboard> = Vec::new();

        // Add current position to history planes
        let mut time_steps: VecDeque<HistoryPlane> = VecDeque::with_capacity(8);
        time_steps.push_front(self.to_history_plane());
        time_steps.extend(self.history_planes.iter().cloned());

        // Add bitboards to image
        for time_step in time_steps.iter() {
            for c in [us, !us] {
                for p in PIECES.iter() {
                    image.push(time_step.pieces_bb(*p) & time_step.occupancy_bb(c));
                }
            }
            for rc in [2, 3] {
                if time_step.repetition_count == rc {
                    image.push(BB_FULL);
                } else {
                    image.push(BB_EMPTY);
                }
            }
        }
        if time_steps.len() < 8 {
            image.resize(image.len() + 14 * (8 - time_steps.len()), BB_EMPTY);
        }
        for c in [us, !us] {
            if self.castling_rights.has_kingside_rights(c) {
                image.push(BB_FULL)
            } else {
                image.push(BB_EMPTY)
            }
            if self.castling_rights.has_queenside_rights(c) {
                image.push(BB_FULL)
            } else {
                image.push(BB_EMPTY)
            }
        }
        image.push(Bitboard(self.half_move_counter() as u64));

        image
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        // Always call this method to get legal moves, never legal_moves directly
        if self.cached_legal_moves.is_empty() {
            // A way to cache legal moves for a position
            let (moves, is_check) = self.generate_legal_moves();
            self.cached_legal_moves = moves;
            self.cached_is_check = Some(is_check);
        }
        self.cached_legal_moves.clone()
    }

    pub fn is_terminal(&mut self) -> (bool, Option<Color>) {
        if self.is_checkmate() {
            self.outcome = Some(Outcome::Checkmate);
            return (true, Some(!self.turn));
        }
        if self.is_stalemate() {
            self.outcome = Some(Outcome::Stalemate);
            return (true, None);
        }
        if self.is_draw_by_threefold_repetition() {
            self.outcome = Some(Outcome::ThreeFoldRepetition);
            return (true, None);
        }
        if self.is_draw_by_insufficient_material() {
            self.outcome = Some(Outcome::InsufficientMaterial);
            return (true, None);
        }
        if self.is_draw_by_50_move_rule() {
            self.outcome = Some(Outcome::FiftyMoveRule);
            return (true, None);
        }
        (false, None)
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    pub fn push(&mut self, r#move: &Move) -> Result<(), ()> {
        self.make_move(r#move);
        Ok(())
    }

    pub fn push_uci(&mut self, uci: &str) -> Result<(), ()> {
        let from = Square::from(uci.get(0..2).unwrap());
        let to: Square = Square::from(uci.get(2..4).unwrap());
        let promo: Option<char> = if uci.len() == 5 {
            Some(uci.chars().nth(4).unwrap())
        } else {
            None
        };
        let flag = self.get_flag_from_uci(from, to, promo);
        let r#move = Move::new(from, to, flag);
        self.push(&r#move)
    }

    fn is_checkmate(&mut self) -> bool {
        self.legal_moves().is_empty() && self.cached_is_check.unwrap_or(false)
    }

    fn is_stalemate(&mut self) -> bool {
        self.legal_moves().is_empty() && !self.cached_is_check.unwrap_or(false)
    }

    fn has_insufficient_material(&self, player: Color) -> bool {
        // King + rook, queen or pawn is not insufficient material
        let rqp = self.pieces_bb(Piece::Rook, player)
            | self.pieces_bb(Piece::Queen, player)
            | self.pieces_bb(Piece::Pawn, player);
        if rqp != BB_EMPTY {
            return false;
        }
        // A knight is not insufficient material if we have more than one
        let knights = self.pieces_bb(Piece::Knight, player);
        let knights_count = knights.pop_count();

        if knights_count > 1 {
            return false;
        } else if knights_count == 1 {
            // If opponent has more than one piece that is not a king or queen
            let opponent = !player;
            let non_king_queen = self.occupancy_bb(opponent)
                & !self.pieces_bb(Piece::King, opponent)
                & !self.pieces_bb(Piece::Queen, opponent);
            if non_king_queen.pop_count() > 1 {
                return false;
            }
        }
        // Bishops on opposite colors are not insufficient material
        let light_bishops = self.pieces_bb(Piece::Bishop, player) & BB_LIGHT_SQUARES;
        let dark_bishops = self.pieces_bb(Piece::Bishop, player) & BB_DARK_SQUARES;
        let bishop_count = light_bishops.pop_count() + dark_bishops.pop_count();
        if bishop_count > 1 {
            // With two bishops, we can always checkmate
            return false;
        } else if bishop_count == 1 {
            // With one bishop, we can only checkmate if the opponent has a knight or pawn or bishop of opposite color to block the king
            let opponent = !player;
            let opposite_bishop = if light_bishops != BB_EMPTY {
                self.pieces_bb(Piece::Bishop, opponent) & BB_DARK_SQUARES
            } else {
                self.pieces_bb(Piece::Bishop, opponent) & BB_LIGHT_SQUARES
            };
            let kpb = self.pieces_bb(Piece::Knight, opponent)
                | self.pieces_bb(Piece::Pawn, opponent)
                | opposite_bishop;
            if kpb != BB_EMPTY {
                return false;
            }
        }

        true
    }

    fn is_draw_by_insufficient_material(&self) -> bool {
        if self.piece_count > 5 {
            return false;
        }
        if self.piece_count == 2 {
            return true;
        }
        self.has_insufficient_material(self.turn) && self.has_insufficient_material(!self.turn)
    }

    fn is_draw_by_50_move_rule(&self) -> bool {
        self.half_move_counter >= 100
    }

    fn count_repetitions(&self, max: u8) -> u8 {
        let mut repetitions: u8 = 1;
        self.zobrist_history
            .iter()
            .rev()
            .take(self.half_move_counter() as usize)
            .for_each(|&hash| {
                if hash == self.zobrist_hash {
                    repetitions += 1;
                }
                if repetitions >= max {
                    return;
                }
            });
        repetitions
    }

    fn is_draw_by_threefold_repetition(&self) -> bool {
        // Check for three repetitions from last capture or pawn move
        if self.count_repetitions(3) >= 3 {
            return true;
        }
        false
    }
}

// // // // // // // // // // // // //
//                                  //
//  Board state manipulation logic  //
//                                  //
// // // // // // // // // // // // //
impl Board {
    #[inline(always)]
    fn put_piece_at(&mut self, square: Square, piece: Piece, color: Color) {
        self.pieces_list[square as usize] = Some(piece);
        self.pieces_bb[piece] |= BB_SQUARES[square as usize];
        self.occupancy_bb[color] |= BB_SQUARES[square as usize];
        self.zobrist_hash ^= self.lookups.zobrist_piece(piece, square);
    }

    #[inline(always)]
    fn clear_piece_at(&mut self, square: Square, color: Color) -> Piece {
        let piece = self.pieces_list[square as usize].unwrap();
        self.pieces_list[square as usize] = None;
        self.pieces_bb[piece] &= !BB_SQUARES[square as usize];
        self.occupancy_bb[color] &= !BB_SQUARES[square as usize];
        self.zobrist_hash ^= self.lookups.zobrist_piece(piece, square);
        piece
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square, color: Color) {
        let piece = self.clear_piece_at(from, color);
        self.put_piece_at(to, piece, color);
    }

    #[inline(always)]
    fn move_piece_and_capture(&mut self, from: Square, to: Square, color: Color) {
        let moved_piece = self.clear_piece_at(from, color);
        self.clear_piece_at(to, !color);
        self.put_piece_at(to, moved_piece, color);
    }

    #[inline(always)]
    fn pawn_pseudo_legal(&self, square: Square, color: Color) -> Bitboard {
        self.lookups.pawn_pseudo_legal(square, color)
    }

    #[inline(always)]
    fn piece_pseudo_legal(&self, square: Square, piece: Piece) -> Bitboard {
        self.lookups.piece_pseudo_legal(square, piece)
    }

    #[inline(always)]
    fn pawn_attacks(&self, pawns_mask: Bitboard, color: Color) -> Bitboard {
        Bitboard::shift(pawns_mask, NORTH_WEST & color)
            | Bitboard::shift(pawns_mask, NORTH_EAST & color)
    }

    #[inline(always)]
    fn knight_attacks(&self, knights: Bitboard) -> Bitboard {
        let mut b = Bitboard::default();
        for sq in knights {
            b |= self.lookups.piece_pseudo_legal(sq, Piece::Knight);
        }
        b
    }

    #[inline(always)]
    fn single_knight_attacks(&self, knight: Square) -> Bitboard {
        self.lookups.piece_pseudo_legal(knight, Piece::Knight)
    }

    #[inline(always)]
    fn bishop_attacks(&self, bishops: Bitboard, occ: Bitboard) -> Bitboard {
        let mut b = Bitboard::default();
        for sq in bishops {
            b |= self.lookups.bishop_attacks(sq, occ);
        }
        b
    }

    #[inline(always)]
    fn single_bishop_attacks(&self, bishop: Square, occ: Bitboard) -> Bitboard {
        self.lookups.bishop_attacks(bishop, occ)
    }

    #[inline(always)]
    fn rook_attacks(&self, rooks: Bitboard, occ: Bitboard) -> Bitboard {
        let mut b = Bitboard::default();
        for sq in rooks {
            b |= self.lookups.rook_attacks(sq, occ);
        }
        b
    }

    #[inline(always)]
    fn single_rook_attacks(&self, rook: Square, occ: Bitboard) -> Bitboard {
        self.lookups.rook_attacks(rook, occ)
    }

    #[inline(always)]
    fn queen_attacks(&self, queens: Bitboard, occ: Bitboard) -> Bitboard {
        let mut b = Bitboard::default();
        for sq in queens {
            b |= self.lookups.rook_attacks(sq, occ) | self.lookups.bishop_attacks(sq, occ);
        }
        b
    }

    #[inline(always)]
    fn single_queen_attacks(&self, queen: Square, occ: Bitboard) -> Bitboard {
        self.lookups.rook_attacks(queen, occ) | self.lookups.bishop_attacks(queen, occ)
    }

    #[inline(always)]
    fn squares_between(&self, sq1: Square, sq2: Square) -> Bitboard {
        self.lookups.squares_between(sq1, sq2)
    }

    #[inline(always)]
    fn line_through(&self, sq1: Square, sq2: Square) -> Bitboard {
        self.lookups.line_through(sq1, sq2)
    }

    // Following the steps of: https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
    fn generate_legal_moves(&self) -> (Vec<Move>, bool) {
        let mut moves: Vec<Move> = Vec::new();
        let mut is_check: bool = false;
        let player: Color = self.turn;
        let opponent: Color = !self.turn;
        let player_pieces: Bitboard = self.occupancy_bb(player);
        let opponent_pieces: Bitboard = self.occupancy_bb(opponent);
        let occupancy: Bitboard = player_pieces | opponent_pieces;
        let player_king_square = self.pieces_bb(Piece::King, player).bsf().unwrap();
        let opponent_king_square = self.pieces_bb(Piece::King, opponent).bsf().unwrap();
        let occupancy_minus_king: Bitboard = occupancy ^ BB_SQUARES[player_king_square];
        let player_orth_sliders: Bitboard =
            self.pieces_bb(Piece::Rook, player) | self.pieces_bb(Piece::Queen, player);
        let player_diag_sliders: Bitboard =
            self.pieces_bb(Piece::Bishop, player) | self.pieces_bb(Piece::Queen, player);
        let opponent_orth_sliders: Bitboard =
            self.pieces_bb(Piece::Rook, opponent) | self.pieces_bb(Piece::Queen, opponent);
        let opponent_diag_sliders: Bitboard =
            self.pieces_bb(Piece::Bishop, opponent) | self.pieces_bb(Piece::Queen, opponent);

        // All squares attacked by opponent (squares to which a king can not move)
        let danger_squares: Bitboard = self
            .pawn_attacks(self.pieces_bb(Piece::Pawn, opponent), opponent)
            | self.knight_attacks(self.pieces_bb(Piece::Knight, opponent))
            | self.bishop_attacks(
                self.pieces_bb(Piece::Bishop, opponent),
                occupancy_minus_king,
            )
            | self.rook_attacks(self.pieces_bb(Piece::Rook, opponent), occupancy_minus_king)
            | self.queen_attacks(self.pieces_bb(Piece::Queen, opponent), occupancy_minus_king)
            | self.piece_pseudo_legal(opponent_king_square, Piece::King);

        // All king generate_moves
        let king_moves = self.piece_pseudo_legal(player_king_square, Piece::King) & !danger_squares;
        for sq in king_moves & opponent_pieces {
            moves.push(Move::new(player_king_square, sq, Flags::Capture));
        }
        for sq in king_moves & !occupancy {
            moves.push(Move::new(player_king_square, sq, Flags::QuietMove));
        }

        // All pawns and knights giving check to players king
        let mut checkers = self.piece_pseudo_legal(player_king_square, Piece::Knight)
            & self.pieces_bb(Piece::Knight, opponent)
            | self.pawn_pseudo_legal(player_king_square, player)
                & self.pieces_bb(Piece::Pawn, opponent);

        // Sliding pieces_bb giving check to players king
        // Calculate for opponent occupancy_bb only
        let checks_and_pins = (self.single_rook_attacks(player_king_square, opponent_pieces)
            & opponent_orth_sliders)
            | (self.single_bishop_attacks(player_king_square, opponent_pieces)
                & opponent_diag_sliders);

        // Add checks (pins with direct line of sight to the king) to checkers
        // Make a mask of players pinned pieces_bb
        let mut pinned_pieces = Bitboard::default();
        for sq in checks_and_pins {
            let tmp = self.squares_between(sq, player_king_square) & player_pieces;
            if tmp == BB_EMPTY {
                checkers ^= BB_SQUARES[sq];
            } else if (tmp & (tmp - 1)) == BB_EMPTY {
                pinned_pieces ^= tmp;
            }
        }
        let not_pinned_pieces: Bitboard = !pinned_pieces;

        let capture_mask: Bitboard;
        let quiet_mask: Bitboard;
        let num_checkers = checkers.pop_count();
        if num_checkers > 1 {
            // Only way to get out of a double check is to move the king out of the way or capture one of the checking pieces_bb.
            return (moves, true);
        } else if let Some(checker_square) = checkers.bsf() {
            is_check = true;
            // With single checks and additional option is to block the check
            // Knight and pawn checks can not be blocked so generate all captures
            if checkers & self.pieces_bb(Piece::Pawn, opponent) != BB_EMPTY
                || checkers & self.pieces_bb(Piece::Knight, opponent) != BB_EMPTY
            {
                // En-passant capture
                if let Some(ep_sq) = self.ep_sq {
                    if Bitboard::shift(BB_SQUARES[ep_sq], SOUTH & player) == checkers {
                        let pawn_attacks = self.pawn_pseudo_legal(ep_sq, opponent)
                            & self.pieces_bb(Piece::Pawn, player)
                            & not_pinned_pieces;
                        for sq in pawn_attacks {
                            moves.push(Move::new(sq, ep_sq, Flags::EpCapture));
                        }
                    };
                };

                // Our pieces_bb that can capture the checking piece (Except the king which is calculated above)
                let mut attackers = self.pawn_pseudo_legal(checker_square, opponent)
                    & self.pieces_bb(Piece::Pawn, player)
                    | self.single_knight_attacks(checker_square)
                        & self.pieces_bb(Piece::Knight, player)
                    | self.single_bishop_attacks(checker_square, occupancy)
                        & self.pieces_bb(Piece::Bishop, player)
                    | self.single_rook_attacks(checker_square, occupancy)
                        & self.pieces_bb(Piece::Rook, player)
                    | self.single_queen_attacks(checker_square, occupancy)
                        & self.pieces_bb(Piece::Queen, player);
                attackers &= not_pinned_pieces;
                for sq in attackers {
                    moves.push(Move::new(sq, checker_square, Flags::Capture));
                }
                return (moves, is_check);
            }
            // Other piece types can be either captured or blocked
            capture_mask = checkers;
            quiet_mask = self.squares_between(player_king_square, checker_square);
        } else {
            // No checks, generate all generate_moves
            capture_mask = opponent_pieces;
            // Move pieces_bb to un-occupied squares
            quiet_mask = !occupancy;

            if let Some(ep_sq) = self.ep_sq {
                let pawns_mask =
                    self.pawn_pseudo_legal(ep_sq, opponent) & self.pieces_bb(Piece::Pawn, player); // Pawns that can do en-passant
                let tmp = pawns_mask & not_pinned_pieces;
                for sq in tmp {
                    let attacks = Bitboard::sliding_attacks(
                        player_king_square,
                        occupancy
                            ^ BB_SQUARES[sq]
                            ^ Bitboard::shift(BB_SQUARES[ep_sq], SOUTH & player),
                        BB_RANKS[player_king_square.rank_idx()],
                    ) & opponent_orth_sliders;
                    if attacks == BB_EMPTY {
                        moves.push(Move::new(sq, ep_sq, Flags::EpCapture));
                    }
                }
                // Pinned pawns can still move in the direction of the pin
                let tmp = pawns_mask & pinned_pieces & self.line_through(player_king_square, ep_sq);
                for from in tmp {
                    moves.push(Move::new(from, ep_sq, Flags::EpCapture));
                }
            }

            // Castling generate_moves
            if self.castling_rights.has_kingside_rights(player)
                && (((occupancy | danger_squares) & BB_0_0_OCC[player]) == BB_EMPTY)
            {
                moves.push(Move::new(
                    Square::E1 & player,
                    Square::G1 & player,
                    Flags::KingSideCastle,
                ));
            }

            if self.castling_rights.has_queenside_rights(player)
                && (((occupancy | (danger_squares & !BB_0_0_IGNORE_DANGER[player]))
                    & BB_0_0_O_OCC[player])
                    == BB_EMPTY)
            {
                moves.push(Move::new(
                    Square::E1 & player,
                    Square::C1 & player,
                    Flags::QueenSideCastle,
                ));
            }

            // Moves for pinned pieces_bb (except Knights & Pawns), Pinned Knights can not move and pawns are handled below
            let tmp = pinned_pieces & player_diag_sliders;
            for from in tmp {
                let diag_attacks = self.single_bishop_attacks(from, occupancy)
                    & self.line_through(player_king_square, from);
                for to in diag_attacks & quiet_mask {
                    moves.push(Move::new(from, to, Flags::QuietMove));
                }
                for to in diag_attacks & capture_mask {
                    moves.push(Move::new(from, to, Flags::Capture));
                }
            }
            let tmp = pinned_pieces & player_orth_sliders;
            for from in tmp {
                let orth_attacks = self.single_rook_attacks(from, occupancy)
                    & self.line_through(player_king_square, from);
                for to in orth_attacks & quiet_mask {
                    moves.push(Move::new(from, to, Flags::QuietMove));
                }
                for to in orth_attacks & capture_mask {
                    moves.push(Move::new(from, to, Flags::Capture));
                }
            }

            // Moves for each pinned pawn
            let tmp = pinned_pieces & self.pieces_bb(Piece::Pawn, player);
            for sq in tmp {
                let pin_mask = self.line_through(player_king_square, sq);
                // Pinned pawns on second to last rank
                if sq.rank_idx() == (RANK7 & player) as usize {
                    let attacks = self.pawn_pseudo_legal(sq, player) & capture_mask & pin_mask;
                    for to in attacks {
                        moves.push(Move::new(sq, to, Flags::KnightPromotionCapture));
                        moves.push(Move::new(sq, to, Flags::BishopPromotionCapture));
                        moves.push(Move::new(sq, to, Flags::RookPromotionCapture));
                        moves.push(Move::new(sq, to, Flags::QueenPromotionCapture));
                    }
                } else {
                    // Pinned pawns capturing other pieces_bb
                    let attacks = self.pawn_pseudo_legal(sq, player) & opponent_pieces & pin_mask;
                    for to in attacks {
                        moves.push(Move::new(sq, to, Flags::Capture));
                    }
                    // Pinned pawns moving forward one square
                    let single_pushes =
                        Bitboard::shift(BB_SQUARES[sq], NORTH & player) & !occupancy & pin_mask;
                    let double_pushes =
                        Bitboard::shift(single_pushes & BB_RANKS[RANK3 & player], NORTH & player)
                            & !occupancy; // & pin_mask;
                    for to in single_pushes {
                        moves.push(Move::new(sq, to, Flags::QuietMove));
                    }
                    for to in double_pushes {
                        moves.push(Move::new(sq, to, Flags::DoublePawnPush));
                    }
                }
            }
        }

        // Moves for non-pinned Knights
        let tmp = self.pieces_bb(Piece::Knight, player) & not_pinned_pieces;
        for from in tmp {
            let attacks = self.single_knight_attacks(from) & !player_pieces;
            for to in attacks & capture_mask {
                moves.push(Move::new(from, to, Flags::Capture));
            }
            for to in attacks & quiet_mask {
                moves.push(Move::new(from, to, Flags::QuietMove));
            }
        }

        // Moves for non-pinned diagonal sliders (Queens, Bishops)
        let tmp = player_diag_sliders & not_pinned_pieces;
        for from in tmp {
            let attacks = self.single_bishop_attacks(from, occupancy) & !player_pieces;
            for to in attacks & capture_mask {
                moves.push(Move::new(from, to, Flags::Capture));
            }
            for to in attacks & quiet_mask {
                moves.push(Move::new(from, to, Flags::QuietMove));
            }
        }

        // Moves for non-pinned orthogonal sliders (Queens, Rooks)
        let tmp = player_orth_sliders & not_pinned_pieces;
        for from in tmp {
            let attacks = self.single_rook_attacks(from, occupancy) & !player_pieces;
            for to in attacks & capture_mask {
                moves.push(Move::new(from, to, Flags::Capture));
            }
            for to in attacks & quiet_mask {
                moves.push(Move::new(from, to, Flags::QuietMove));
            }
        }

        // All pawns excluding last two ranks (Rank 7 to 8 are promotions & pawns on Rank 8 are not possible)
        let tmp =
            self.pieces_bb(Piece::Pawn, player) & not_pinned_pieces & !BB_RANKS[RANK7 & player];
        let mut single_pushes = Bitboard::shift(tmp, NORTH & player) & !occupancy;
        // Treat double pushes as a 2-step move, first we push Rank 2 to 3 and then once more
        // Only pawns on Rank 2 are eligible
        let double_pushes =
            Bitboard::shift(single_pushes & BB_RANKS[RANK3 & player], NORTH & player) & quiet_mask;
        single_pushes &= quiet_mask;

        for sq in single_pushes {
            moves.push(Move::new(sq - NORTH * player, sq, Flags::QuietMove));
        }
        for sq in double_pushes {
            moves.push(Move::new(sq - NORTH2X * player, sq, Flags::DoublePawnPush));
        }

        // Pawn captures
        // tmp still contains positions of non-pinned pawns
        for sq in Bitboard::shift(tmp, NORTH_WEST & player) & capture_mask {
            moves.push(Move::new(sq - NORTH_WEST * player, sq, Flags::Capture));
        }
        for sq in Bitboard::shift(tmp, NORTH_EAST & player) & capture_mask {
            moves.push(Move::new(sq - NORTH_EAST * player, sq, Flags::Capture));
        }

        // Pawn promotions
        let tmp =
            self.pieces_bb(Piece::Pawn, player) & not_pinned_pieces & BB_RANKS[RANK7 & player];
        if tmp != BB_EMPTY {
            // Quiet promotions
            for sq in Bitboard::shift(tmp, NORTH & player) & quiet_mask {
                moves.push(Move::new(sq - NORTH * player, sq, Flags::KnightPromotion));
                moves.push(Move::new(sq - NORTH * player, sq, Flags::BishopPromotion));
                moves.push(Move::new(sq - NORTH * player, sq, Flags::RookPromotion));
                moves.push(Move::new(sq - NORTH * player, sq, Flags::QueenPromotion));
            }
            // Capturing promotions
            for sq in Bitboard::shift(tmp, NORTH_WEST & player) & capture_mask {
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    Flags::KnightPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    Flags::BishopPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    Flags::RookPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    Flags::QueenPromotionCapture,
                ));
            }
            for sq in Bitboard::shift(tmp, NORTH_EAST & player) & capture_mask {
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    Flags::KnightPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    Flags::BishopPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    Flags::RookPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    Flags::QueenPromotionCapture,
                ));
            }
        }
        (moves, is_check)
    }

    fn make_move(&mut self, m: &Move) {
        // Save current state to history
        self.add_history_plane(self.to_history_plane());
        self.zobrist_history.push(self.zobrist_hash);
        let mut ep_sq = None;
        match m.flags() {
            Flags::QuietMove => {
                self.move_piece(m.sq_from(), m.sq_to(), self.turn);
                if self.piece_at(m.sq_to()).unwrap() == Piece::Pawn {
                    self.half_move_counter = 0;
                } else {
                    self.half_move_counter += 1;
                }
            }
            Flags::DoublePawnPush => {
                self.move_piece(m.sq_from(), m.sq_to(), self.turn);
                // Set en-passant square behind the pawn
                let sq = m.sq_to() + (SOUTH * self.turn);
                self.zobrist_hash ^= self.lookups.zobrist_ep(sq);
                ep_sq = Some(sq);
                self.half_move_counter = 0;
            }
            Flags::Capture => {
                self.move_piece_and_capture(m.sq_from(), m.sq_to(), self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            Flags::EpCapture => {
                self.move_piece(m.sq_from(), m.sq_to(), self.turn);
                self.clear_piece_at(m.sq_to() + (SOUTH * self.turn), !self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            Flags::KingSideCastle => {
                self.move_piece(Square::E1 & self.turn, Square::G1 & self.turn, self.turn);
                self.move_piece(Square::H1 & self.turn, Square::F1 & self.turn, self.turn);
                self.castling_rights.remove_both_rights(self.turn);
                self.half_move_counter += 1;
            }
            Flags::QueenSideCastle => {
                self.move_piece(Square::E1 & self.turn, Square::C1 & self.turn, self.turn);
                self.move_piece(Square::A1 & self.turn, Square::D1 & self.turn, self.turn);
                self.castling_rights.remove_both_rights(self.turn);
                self.half_move_counter += 1;
            }
            Flags::KnightPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.put_piece_at(m.sq_to(), Piece::Knight, self.turn);
                self.half_move_counter = 0;
            }
            Flags::BishopPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.put_piece_at(m.sq_to(), Piece::Bishop, self.turn);
                self.half_move_counter = 0;
            }
            Flags::RookPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.put_piece_at(m.sq_to(), Piece::Rook, self.turn);
                self.half_move_counter = 0;
            }
            Flags::QueenPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.put_piece_at(m.sq_to(), Piece::Queen, self.turn);
                self.half_move_counter = 0;
            }
            Flags::KnightPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.clear_piece_at(m.sq_to(), !self.turn);
                self.put_piece_at(m.sq_to(), Piece::Knight, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            Flags::BishopPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.clear_piece_at(m.sq_to(), !self.turn);
                self.put_piece_at(m.sq_to(), Piece::Bishop, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            Flags::RookPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.clear_piece_at(m.sq_to(), !self.turn);
                self.put_piece_at(m.sq_to(), Piece::Rook, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            Flags::QueenPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn);
                self.clear_piece_at(m.sq_to(), !self.turn);
                self.put_piece_at(m.sq_to(), Piece::Queen, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
        };
        self.maybe_update_castling_rights(m);
        self.zobrist_hash ^= self.lookups.zobrist_castling(self.castling_rights);
        self.ep_sq = ep_sq.and_then(|sq| Some(sq)); // Resets en-passant square if the move was not a double pawn push
        if self.turn == BLACK {
            self.zobrist_hash ^= self.lookups.zobrist_black();
        }
        self.turn = !self.turn;
        self.ply += 1;
        self.clear_cache();
    }

    fn clear_cache(&mut self) {
        self.cached_legal_moves = Vec::new();
        self.cached_is_check = None;
    }

    fn maybe_update_castling_rights(&mut self, m: &Move) {
        if self.castling_rights.is_empty() {
            // No one can castle anymore so no sense in checking
            return;
        }
        match m.flags() {
            Flags::KingSideCastle
            | Flags::QueenSideCastle
            | Flags::DoublePawnPush
            | Flags::EpCapture => return, // Skip for flags that dont need checking for update
            _ => {}
        }
        let sq_from: Square = m.sq_from();
        match sq_from {
            Square::E1 | Square::E8 => {
                self.castling_rights.remove_both_rights(self.turn);
            }
            Square::H1 | Square::H8 => {
                self.castling_rights.remove_kingside_rights(self.turn);
            }
            Square::A1 | Square::A8 => {
                self.castling_rights.remove_queenside_rights(self.turn);
            }
            _ => {}
        }
        match m.flags() {
            Flags::Capture
            | Flags::KnightPromotionCapture
            | Flags::BishopPromotionCapture
            | Flags::RookPromotionCapture
            | Flags::QueenPromotionCapture => {}
            _ => return,
        }
        match m.sq_to() {
            Square::H1 => {
                self.castling_rights.remove_kingside_rights(WHITE);
            }
            Square::H8 => {
                self.castling_rights.remove_kingside_rights(BLACK);
            }
            Square::A1 => {
                self.castling_rights.remove_queenside_rights(WHITE);
            }
            Square::A8 => {
                self.castling_rights.remove_queenside_rights(BLACK);
            }
            _ => return,
        }
    }

    #[inline(always)]
    fn add_history_plane(&mut self, history_plane: HistoryPlane) {
        if self.history_planes.len() == self.history_planes.capacity() {
            self.history_planes.pop_back();
        }
        self.history_planes.push_front(history_plane);
    }

    #[inline(always)]
    fn to_history_plane(&self) -> HistoryPlane {
        HistoryPlane {
            turn: self.turn,
            pieces: self.pieces_bb,
            occupancy: self.occupancy_bb,
            repetition_count: self.count_repetitions(3),
        }
    }

    fn get_flag_from_uci(&self, from: Square, to: Square, promo: Option<char>) -> Flags {
        let moving_piece = match self.piece_at(from) {
            Some(piece) => piece,
            None => {
                panic!("No piece at square {}", from);
            }
        };
        let captured_piece: Option<Piece> = self.piece_at(to);
        if moving_piece == Piece::King {
            if (from == Square::E1 && to == Square::G1) || (from == Square::E8 && to == Square::G8)
            {
                return Flags::KingSideCastle;
            } else if (from == Square::E1 && to == Square::C1)
                || (from == Square::E8 && to == Square::C8)
            {
                return Flags::QueenSideCastle;
            }
        }
        if moving_piece == Piece::Pawn {
            if let Some(sq) = self.ep_sq {
                if sq == to {
                    return Flags::EpCapture;
                }
            } /*else if from + NORTH2X * self.turn == to {
                return Flags::DoublePawnPush;
            }*/
            if self.turn == WHITE && from.rank_idx() == 1 {
                if from + NORTH2X * self.turn == to {
                    return Flags::DoublePawnPush;
                }
            } else if self.turn == BLACK && from.rank_idx() == 6 {
                if from + NORTH2X * self.turn == to {
                    return Flags::DoublePawnPush;
                }
            }
        }
        if captured_piece.is_some() {
            if promo.is_some() {
                match promo.unwrap() {
                    'q' => return Flags::QueenPromotionCapture,
                    'r' => return Flags::RookPromotionCapture,
                    'b' => return Flags::BishopPromotionCapture,
                    'n' => return Flags::KnightPromotionCapture,
                    _ => panic!(),
                }
            } else {
                return Flags::Capture;
            }
        } else if (to.rank_idx() == 0 || to.rank_idx() == 7) && promo.is_some() {
            match promo.unwrap() {
                'q' => return Flags::QueenPromotion,
                'r' => return Flags::RookPromotion,
                'b' => return Flags::BishopPromotion,
                'n' => return Flags::KnightPromotion,
                _ => panic!(),
            }
        }
        Flags::QuietMove
    }
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        const HORIZONTAL_LINE: &str = "  +---+---+---+---+---+---+---+---+\n";
        const FILES_LABEL: &str = "    a   b   c   d   e   f   g   h";
        const RANKS_LABELS: [&str; 8] = [
            "| 1\n", "| 2\n", "| 3\n", "| 4\n", "| 5\n", "| 6\n", "| 7\n", "| 8\n",
        ];
        let mut board_str = String::new();
        board_str.push_str("\n");
        board_str.push_str(HORIZONTAL_LINE);
        for rank in (0..8).rev() {
            board_str.push_str("  ");
            for file in 0..8 {
                let square = SQUARES[rank * 8 + file];
                let piece_char = match self.piece_at(square) {
                    Some(piece) => {
                        let mut piece_char = match piece {
                            Piece::Pawn => 'p',
                            Piece::Knight => 'n',
                            Piece::Bishop => 'b',
                            Piece::Rook => 'r',
                            Piece::Queen => 'q',
                            Piece::King => 'k',
                        };
                        if self.occupancy_bb[WHITE] & BB_SQUARES[square] != BB_EMPTY {
                            piece_char = piece_char.to_ascii_uppercase()
                        }
                        piece_char
                    }
                    None => ' ',
                };
                board_str.push_str(format!("| {piece_char} ").as_str());
            }
            board_str.push_str(RANKS_LABELS[rank]);
            board_str.push_str(HORIZONTAL_LINE);
        }
        board_str.push_str(FILES_LABEL);
        board_str.push_str("\n\n");
        board_str.push_str(format!("Fen: {}\n", self.fen()).as_str());
        board_str.push_str(format!("Key: {:02x}", self.zobrist_hash()).as_str());
        write!(f, "{}", board_str)
    }
}
