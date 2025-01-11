#![allow(unused_variables)]
use crate::bitboard::masks::*;
use crate::bitboard::{Bitboard, NORTH, NORTH2X, NORTH_EAST, NORTH_WEST, SOUTH};
use crate::errors::BoardError;
use crate::statics::Lookups;
use crate::types::castling_rights::CastlingRights;
use crate::types::color::{Color, BLACK, WHITE};
use crate::types::piece::{Piece, PIECES};
use crate::types::r#move::{Move, MoveFlags};
use crate::types::ranks_and_files::{RANK3, RANK7};
use crate::types::square::{Square, SQUARES, SQUARES_REV};
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    FiftyMoveRule,
    ThreeFoldRepetition,
}

const STARTING_POSITION: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";

#[derive(Clone)]
pub struct HistoryPlane {
    turn: Color,
    pieces: [Bitboard; 6],
    occupancy: [Bitboard; 2],
    repetition_count: u8,
    zobrist_hash: u64,
}

impl HistoryPlane {
    pub fn turn(&self) -> Color {
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

    pub fn zobrist_hash(&self) -> u64 {
        self.zobrist_hash
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

    moves_history: Vec<Move>,

    cached_legal_moves: Vec<Move>, // Legal moves for the current position
    cached_is_check: Option<bool>, // Is the current position a check

    outcome: Option<Outcome>, // Outcome of the game

    pub lookups: Lookups,
}

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
            moves_history: Vec::new(),
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
            moves_history: self.moves_history.clone(),
            cached_legal_moves: self.cached_legal_moves.clone(),
            cached_is_check: self.cached_is_check,
            outcome: None,
            lookups: self.lookups,
        }
    }
}

impl Board {
    pub fn new(fen: Option<&str>) -> Board {
        let fen = fen.unwrap_or(STARTING_POSITION);
        match Self::try_from_fen(fen) {
            Ok(board) => board,
            Err(err) => panic!("{}", err),
        }
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

    pub fn history_hash(&self) -> u64 {
        let mut combined_hash: u64 = self.zobrist_hash;
        for plane in self.history_planes.iter() {
            combined_hash ^= plane.zobrist_hash;
        }
        combined_hash
    }

    pub fn history(&self, flip_uneven: bool) -> (Vec<Bitboard>, u64) {
        let mut image: Vec<Bitboard> = Vec::new();
        let mut combined_hash: u64 = 0;

        // Add current position to history planes
        let mut time_steps: VecDeque<HistoryPlane> = VecDeque::with_capacity(8);
        time_steps.push_front(self.to_history_plane());
        time_steps.extend(self.history_planes.iter().cloned());
        // Add bitboards to image
        for (t, time_step) in time_steps.iter().enumerate() {
            let us = if flip_uneven {
                time_step.turn()
            } else {
                self.turn()
            };
            for c in [us, !us] {
                for p in PIECES.iter() {
                    let mut bb = time_step.pieces_bb(*p) & time_step.occupancy_bb(c);
                    if us == WHITE {
                        bb = Bitboard::flip_vertical(bb);
                    }
                    image.push(bb);
                }
            }
            if time_step.repetition_count == 2 {
                image.push(BB_FULL);
            } else {
                image.push(BB_EMPTY);
            }
            combined_hash ^= time_step.zobrist_hash;
        }
        if time_steps.len() < 8 {
            image.resize(image.len() + 13 * (8 - time_steps.len()), BB_EMPTY);
        }
        let us = self.turn();
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
        (image, combined_hash)
    }

    pub fn moves_history(&self) -> &Vec<Move> {
        self.moves_history.as_ref()
    }

    pub fn legal_moves(&mut self) -> &Vec<Move> {
        // Always call this method to get legal moves, never legal_moves directly
        if self.cached_legal_moves.is_empty() {
            // A way to cache legal moves for a position
            let (moves, is_check) = self.generate_legal_moves();
            self.cached_legal_moves = moves;
            self.cached_is_check = Some(is_check);
        }
        self.cached_legal_moves.as_ref()
    }

    pub fn terminal(&mut self) -> (bool, Option<Color>) {
        if self.checkmate() {
            self.outcome = Some(Outcome::Checkmate);
            return (true, Some(!self.turn));
        }
        if self.stalemate() {
            self.outcome = Some(Outcome::Stalemate);
            return (true, None);
        }
        if self.draw_by_insufficient_material() {
            self.outcome = Some(Outcome::InsufficientMaterial);
            return (true, None);
        }
        if self.draw_by_50move_rule() {
            self.outcome = Some(Outcome::FiftyMoveRule);
            return (true, None);
        }
        if self.draw_by_threefold_repetition() {
            self.outcome = Some(Outcome::ThreeFoldRepetition);
            return (true, None);
        }
        (false, None)
    }

    // Mid search terminal is used during mcts
    // If a position is replayed after the root of the search tree
    // we assume it is a draw by threefold repetition
    pub fn mid_search_terminal(&mut self, depth_to_root: usize) -> (bool, bool) {
        if self.checkmate() {
            return (true, false);
        }
        if self.stalemate() {
            return (true, true);
        }
        if self.draw_by_insufficient_material() {
            return (true, true);
        }
        if self.draw_by_50move_rule() {
            return (true, true);
        }
        let mut repetitions: u8 = 1;
        for (i, &zh) in self
            .zobrist_history
            .iter()
            .rev()
            .take(self.half_move_counter() as usize)
            .enumerate()
        {
            if zh == self.zobrist_hash {
                repetitions += 1;
                if i < depth_to_root {
                    return (true, true);
                }
            }
            if repetitions >= 3 {
                return (true, true);
            }
        }
        (false, false)
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    pub fn push(&mut self, r#move: &Move) -> Result<(), BoardError> {
        self.make_move(r#move)?;
        Ok(())
    }

    pub fn push_uci(&mut self, uci: &str) -> Result<(), BoardError> {
        let r#move = self.try_parse_uci(uci)?;
        self.push(&r#move)
    }
}

// // // // // // // // // // // // //
//                                  //
//       Board private logic        //
//                                  //
// // // // // // // // // // // // //
impl Board {
    fn try_from_fen(fen: &str) -> Result<Board, BoardError> {
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
                match Piece::try_from(c.to_string().as_str()) {
                    Ok(piece) => {
                        board.put_piece_at(SQUARES_REV[square_idx], piece, color);
                        square_idx += 1;
                        board.piece_count += 1;
                    }
                    Err(err) => {
                        return Err(BoardError::InvalidFen(
                            fen.to_string(),
                            format!("{}: {}", err, c),
                        ));
                    }
                }
            }
        }
        board.turn = match fen_parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            _ => {
                return Err(BoardError::InvalidFen(
                    fen.to_string(),
                    format!("Invalid turn: {}", fen_parts[1]),
                ))
            }
        };
        board.castling_rights = CastlingRights::from(fen_parts[2]);
        board.ep_sq = match fen_parts[3] {
            "-" => None,
            _ => {
                let sq = Square::try_from(fen_parts[3]);
                match sq {
                    Ok(sq) => Some(sq),
                    Err(err) => {
                        return Err(BoardError::InvalidFen(
                            fen.to_string(),
                            format!("Invalid ep square: {}", err),
                        ))
                    }
                }
            }
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
        Ok(board)
    }

    fn try_parse_uci(&self, uci: &str) -> Result<Move, BoardError> {
        if ![4, 5].contains(&uci.len()) {
            return Err(BoardError::InvalidUci(format!(
                "Invalid length: {} (expected 4 or 5, got {})",
                uci,
                uci.len()
            )));
        }
        let from_str = uci.get(0..2).unwrap();
        let from = match Square::try_from(from_str) {
            Ok(square) => square,
            Err(err) => {
                return Err(BoardError::InvalidUci(format!(
                    "Invalid from: {} ({})",
                    from_str, err
                )))
            }
        };
        let to_str = uci.get(2..4).unwrap();
        let to = match Square::try_from(to_str) {
            Ok(square) => square,
            Err(err) => {
                return Err(BoardError::InvalidUci(format!(
                    "Invalid to: {} ({})",
                    to_str, err
                )))
            }
        };
        let promo: Option<char> = if uci.len() == 5 {
            Some(uci.chars().nth(4).unwrap())
        } else {
            None
        };
        let flag = self.get_flag_from_uci(from, to, promo)?;
        Ok(Move::new(from, to, flag))
    }

    fn get_flag_from_uci(
        &self,
        from: Square,
        to: Square,
        promo: Option<char>,
    ) -> Result<MoveFlags, BoardError> {
        let moving_piece = match self.piece_at(from) {
            Some(piece) => piece,
            None => {
                return Err(BoardError::InvalidMove(format!(
                    "No piece to move at {}",
                    from
                )));
            }
        };
        let captured_piece: Option<Piece> = self.piece_at(to);
        if moving_piece == Piece::King {
            if (from == Square::E1 && to == Square::G1) || (from == Square::E8 && to == Square::G8)
            {
                return Ok(MoveFlags::KingSideCastle);
            } else if (from == Square::E1 && to == Square::C1)
                || (from == Square::E8 && to == Square::C8)
            {
                return Ok(MoveFlags::QueenSideCastle);
            }
        }
        if moving_piece == Piece::Pawn {
            if let Some(sq) = self.ep_sq {
                if sq == to {
                    return Ok(MoveFlags::EpCapture);
                }
            }
            if self.turn == WHITE && from.rank_idx() == 1 {
                if from + NORTH2X * self.turn == to {
                    return Ok(MoveFlags::DoublePawnPush);
                }
            } else if self.turn == BLACK && from.rank_idx() == 6 {
                if from + NORTH2X * self.turn == to {
                    return Ok(MoveFlags::DoublePawnPush);
                }
            }
        }
        if captured_piece.is_some() {
            if let Some(p) = promo {
                match p {
                    'q' => return Ok(MoveFlags::QueenPromotionCapture),
                    'r' => return Ok(MoveFlags::RookPromotionCapture),
                    'b' => return Ok(MoveFlags::BishopPromotionCapture),
                    'n' => return Ok(MoveFlags::KnightPromotionCapture),
                    _ => {
                        return Err(BoardError::InvalidUci(format!(
                            "Invalid promotion piece: {}",
                            p
                        )))
                    }
                }
            } else {
                return Ok(MoveFlags::Capture);
            }
        } else if to.rank_idx() == 0 || to.rank_idx() == 7 {
            if let Some(p) = promo {
                match p {
                    'q' => return Ok(MoveFlags::QueenPromotion),
                    'r' => return Ok(MoveFlags::RookPromotion),
                    'b' => return Ok(MoveFlags::BishopPromotion),
                    'n' => return Ok(MoveFlags::KnightPromotion),
                    _ => {
                        return Err(BoardError::InvalidUci(format!(
                            "Invalid promotion piece: {}",
                            p
                        )))
                    }
                }
            }
        }
        Ok(MoveFlags::QuietMove)
    }

    fn checkmate(&mut self) -> bool {
        self.legal_moves().is_empty() && self.cached_is_check.unwrap_or(false)
    }

    fn stalemate(&mut self) -> bool {
        self.legal_moves().is_empty() && !self.cached_is_check.unwrap_or(false)
    }

    fn draw_by_insufficient_material(&self) -> bool {
        if self.piece_count > 4 {
            return false;
        }
        if self.piece_count == 2 {
            return true;
        }
        if (self.pieces_bb[Piece::Pawn] | self.pieces_bb[Piece::Queen] | self.pieces_bb[Piece::Rook]) != BB_EMPTY {
            return false;
        }
        if self.piece_count == 3 {
            return true;
        }
        let white_knights_count = (self.pieces_bb[Piece::Knight] & self.occupancy_bb[WHITE]).pop_count();
        let black_knights_count = (self.pieces_bb[Piece::Knight] & self.occupancy_bb[BLACK]).pop_count();
        let white_light_bishops_count = ((self.pieces_bb[Piece::Bishop] & self.occupancy_bb[WHITE]) & BB_LIGHT_SQUARES).pop_count();
        let white_dark_bishops_count = ((self.pieces_bb[Piece::Bishop] & self.occupancy_bb[WHITE]) & BB_DARK_SQUARES).pop_count();
        let black_light_bishops_count = ((self.pieces_bb[Piece::Bishop] & self.occupancy_bb[BLACK]) & BB_LIGHT_SQUARES).pop_count();
        let black_dark_bishops_count = ((self.pieces_bb[Piece::Bishop] & self.occupancy_bb[BLACK]) & BB_DARK_SQUARES).pop_count();
        let white_minors_count = white_knights_count + white_light_bishops_count + white_dark_bishops_count;
        let black_minors_count = black_knights_count + black_light_bishops_count + black_dark_bishops_count;
        if white_minors_count == 1 && black_minors_count == 1 {
            return true;
        }
        // Only two possible combinations for checkmating are: (one side must have only a king)
        // King vs King + Bishop + knight
        // King vs King + Light squared bishop + dark squared bishop
        // Checking above combinations for white
        if white_knights_count == 1 && (white_light_bishops_count + white_dark_bishops_count) == 1 {
            // knight without bishop can not mate
            return false
        } else if white_light_bishops_count == 1 && white_dark_bishops_count == 1 {
            return false
        }
        // Checking above combinations for black
        if black_knights_count == 1 && (black_light_bishops_count + black_dark_bishops_count) == 1 {
            // knight without bishop can not mate
            return false
        } else if black_light_bishops_count == 1 &&  black_dark_bishops_count == 1 {
            return false
        }
        true
    }

    fn draw_by_50move_rule(&self) -> bool {
        self.half_move_counter >= 100
    }

    fn count_repetitions(&self, max: u8) -> u8 {
        let mut repetitions: u8 = 1;
        for &hash in self
            .zobrist_history
            .iter()
            .rev()
            .take(self.half_move_counter() as usize)
        {
            if hash == self.zobrist_hash {
                repetitions += 1;
            }
            if repetitions >= max {
                return repetitions;
            }
        }
        repetitions
    }

    fn draw_by_threefold_repetition(&self) -> bool {
        // Check for three repetitions from last capture or pawn move
        if self.count_repetitions(3) >= 3 {
            return true;
        }
        false
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square, color: Color) -> Result<(), BoardError> {
        let piece = self.clear_piece_at(from, color)?;
        self.put_piece_at(to, piece, color);
        Ok(())
    }

    #[inline(always)]
    fn move_piece_and_capture(
        &mut self,
        from: Square,
        to: Square,
        color: Color,
    ) -> Result<(), BoardError> {
        // Check if there is a piece to capture before clearing the capturing piece
        if self.piece_at(to).is_none() {
            return Err(BoardError::InvalidMove(format!(
                "No piece to capture at {}",
                to
            )));
        };
        let moved_piece = self.clear_piece_at(from, color)?;
        self.clear_piece_at(to, !color)?; // Should never return error because of the if statement above
        self.put_piece_at(to, moved_piece, color);
        Ok(())
    }

    #[inline(always)]
    fn put_piece_at(&mut self, square: Square, piece: Piece, color: Color) {
        self.pieces_list[square as usize] = Some(piece);
        self.pieces_bb[piece] |= BB_SQUARES[square as usize];
        self.occupancy_bb[color] |= BB_SQUARES[square as usize];
        self.zobrist_hash ^= self.lookups.zobrist_piece(piece, square);
    }

    #[inline(always)]
    fn clear_piece_at(&mut self, square: Square, color: Color) -> Result<Piece, BoardError> {
        // Return error if there is no piece to clear
        let piece = match self.pieces_list[square as usize] {
            Some(piece) => piece,
            None => {
                return Err(BoardError::InvalidMove(format!(
                    "No piece to clear from {}",
                    square
                )))
            }
        };
        self.pieces_list[square as usize] = None;
        self.pieces_bb[piece] &= !BB_SQUARES[square as usize];
        self.occupancy_bb[color] &= !BB_SQUARES[square as usize];
        self.zobrist_hash ^= self.lookups.zobrist_piece(piece, square);
        Ok(piece)
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
    pub fn generate_legal_moves(&self) -> (Vec<Move>, bool) {
        let mut moves: Vec<Move> = Vec::new();
        let mut is_check: bool = false;

        let player: Color = self.turn;
        let opponent: Color = !self.turn;
        let player_pieces: Bitboard = self.occupancy_bb(player);
        let opponent_pieces: Bitboard = self.occupancy_bb(opponent);
        let occupancy: Bitboard = player_pieces | opponent_pieces;
        let player_king_square = self.pieces_bb(Piece::King, player).bsf().unwrap(); // Just panic if any of kings is not on board
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
            moves.push(Move::new(player_king_square, sq, MoveFlags::Capture));
        }
        for sq in king_moves & !occupancy {
            moves.push(Move::new(player_king_square, sq, MoveFlags::QuietMove));
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
                            moves.push(Move::new(sq, ep_sq, MoveFlags::EpCapture));
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
                    moves.push(Move::new(sq, checker_square, MoveFlags::Capture));
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
                        moves.push(Move::new(sq, ep_sq, MoveFlags::EpCapture));
                    }
                }
                // Pinned pawns can still move in the direction of the pin
                let tmp = pawns_mask & pinned_pieces & self.line_through(player_king_square, ep_sq);
                for from in tmp {
                    moves.push(Move::new(from, ep_sq, MoveFlags::EpCapture));
                }
            }

            // Castling generate_moves
            if self.castling_rights.has_kingside_rights(player)
                && (((occupancy | danger_squares) & BB_0_0_OCC[player]) == BB_EMPTY)
            {
                moves.push(Move::new(
                    Square::E1 & player,
                    Square::G1 & player,
                    MoveFlags::KingSideCastle,
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
                    MoveFlags::QueenSideCastle,
                ));
            }

            // Moves for pinned pieces_bb (except Knights & Pawns), Pinned Knights can not move and pawns are handled below
            let tmp = pinned_pieces & player_diag_sliders;
            for from in tmp {
                let diag_attacks = self.single_bishop_attacks(from, occupancy)
                    & self.line_through(player_king_square, from);
                for to in diag_attacks & quiet_mask {
                    moves.push(Move::new(from, to, MoveFlags::QuietMove));
                }
                for to in diag_attacks & capture_mask {
                    moves.push(Move::new(from, to, MoveFlags::Capture));
                }
            }
            let tmp = pinned_pieces & player_orth_sliders;
            for from in tmp {
                let orth_attacks = self.single_rook_attacks(from, occupancy)
                    & self.line_through(player_king_square, from);
                for to in orth_attacks & quiet_mask {
                    moves.push(Move::new(from, to, MoveFlags::QuietMove));
                }
                for to in orth_attacks & capture_mask {
                    moves.push(Move::new(from, to, MoveFlags::Capture));
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
                        moves.push(Move::new(sq, to, MoveFlags::KnightPromotionCapture));
                        moves.push(Move::new(sq, to, MoveFlags::BishopPromotionCapture));
                        moves.push(Move::new(sq, to, MoveFlags::RookPromotionCapture));
                        moves.push(Move::new(sq, to, MoveFlags::QueenPromotionCapture));
                    }
                } else {
                    // Pinned pawns capturing other pieces_bb
                    let attacks = self.pawn_pseudo_legal(sq, player) & opponent_pieces & pin_mask;
                    for to in attacks {
                        moves.push(Move::new(sq, to, MoveFlags::Capture));
                    }
                    // Pinned pawns moving forward one square
                    let single_pushes =
                        Bitboard::shift(BB_SQUARES[sq], NORTH & player) & !occupancy & pin_mask;
                    let double_pushes =
                        Bitboard::shift(single_pushes & BB_RANKS[RANK3 & player], NORTH & player)
                            & !occupancy; // & pin_mask;
                    for to in single_pushes {
                        moves.push(Move::new(sq, to, MoveFlags::QuietMove));
                    }
                    for to in double_pushes {
                        moves.push(Move::new(sq, to, MoveFlags::DoublePawnPush));
                    }
                }
            }
        }

        // Moves for non-pinned Knights
        let tmp = self.pieces_bb(Piece::Knight, player) & not_pinned_pieces;
        for from in tmp {
            let attacks = self.single_knight_attacks(from) & !player_pieces;
            for to in attacks & capture_mask {
                moves.push(Move::new(from, to, MoveFlags::Capture));
            }
            for to in attacks & quiet_mask {
                moves.push(Move::new(from, to, MoveFlags::QuietMove));
            }
        }

        // Moves for non-pinned diagonal sliders (Queens, Bishops)
        let tmp = player_diag_sliders & not_pinned_pieces;
        for from in tmp {
            let attacks = self.single_bishop_attacks(from, occupancy) & !player_pieces;
            for to in attacks & capture_mask {
                moves.push(Move::new(from, to, MoveFlags::Capture));
            }
            for to in attacks & quiet_mask {
                moves.push(Move::new(from, to, MoveFlags::QuietMove));
            }
        }

        // Moves for non-pinned orthogonal sliders (Queens, Rooks)
        let tmp = player_orth_sliders & not_pinned_pieces;
        for from in tmp {
            let attacks = self.single_rook_attacks(from, occupancy) & !player_pieces;
            for to in attacks & capture_mask {
                moves.push(Move::new(from, to, MoveFlags::Capture));
            }
            for to in attacks & quiet_mask {
                moves.push(Move::new(from, to, MoveFlags::QuietMove));
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
            moves.push(Move::new(sq - NORTH * player, sq, MoveFlags::QuietMove));
        }
        for sq in double_pushes {
            moves.push(Move::new(
                sq - NORTH2X * player,
                sq,
                MoveFlags::DoublePawnPush,
            ));
        }

        // Pawn captures
        // tmp still contains positions of non-pinned pawns
        for sq in Bitboard::shift(tmp, NORTH_WEST & player) & capture_mask {
            moves.push(Move::new(sq - NORTH_WEST * player, sq, MoveFlags::Capture));
        }
        for sq in Bitboard::shift(tmp, NORTH_EAST & player) & capture_mask {
            moves.push(Move::new(sq - NORTH_EAST * player, sq, MoveFlags::Capture));
        }

        // Pawn promotions
        let tmp =
            self.pieces_bb(Piece::Pawn, player) & not_pinned_pieces & BB_RANKS[RANK7 & player];
        if tmp != BB_EMPTY {
            // Quiet promotions
            for sq in Bitboard::shift(tmp, NORTH & player) & quiet_mask {
                moves.push(Move::new(
                    sq - NORTH * player,
                    sq,
                    MoveFlags::KnightPromotion,
                ));
                moves.push(Move::new(
                    sq - NORTH * player,
                    sq,
                    MoveFlags::BishopPromotion,
                ));
                moves.push(Move::new(sq - NORTH * player, sq, MoveFlags::RookPromotion));
                moves.push(Move::new(
                    sq - NORTH * player,
                    sq,
                    MoveFlags::QueenPromotion,
                ));
            }
            // Capturing promotions
            for sq in Bitboard::shift(tmp, NORTH_WEST & player) & capture_mask {
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    MoveFlags::KnightPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    MoveFlags::BishopPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    MoveFlags::RookPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_WEST * player,
                    sq,
                    MoveFlags::QueenPromotionCapture,
                ));
            }
            for sq in Bitboard::shift(tmp, NORTH_EAST & player) & capture_mask {
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    MoveFlags::KnightPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    MoveFlags::BishopPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    MoveFlags::RookPromotionCapture,
                ));
                moves.push(Move::new(
                    sq - NORTH_EAST * player,
                    sq,
                    MoveFlags::QueenPromotionCapture,
                ));
            }
        }
        (moves, is_check)
    }

    fn make_move(&mut self, m: &Move) -> Result<(), BoardError> {
        let history_plane = self.to_history_plane();
        let zobrist_hash = self.zobrist_hash;
        let mut ep_sq = None;
        match m.flags() {
            MoveFlags::QuietMove => {
                self.move_piece(m.sq_from(), m.sq_to(), self.turn)?;
                if self.piece_at(m.sq_to()).unwrap() == Piece::Pawn {
                    // Unwrap is safe because we know there is a piece at the square
                    self.half_move_counter = 0;
                } else {
                    self.half_move_counter += 1;
                }
            }
            MoveFlags::DoublePawnPush => {
                self.move_piece(m.sq_from(), m.sq_to(), self.turn)?;
                // Set en-passant square behind the pawn
                let sq = m.sq_to() + (SOUTH * self.turn);
                self.zobrist_hash ^= self.lookups.zobrist_ep(sq);
                ep_sq = Some(sq);
                self.half_move_counter = 0;
            }
            MoveFlags::Capture => {
                self.move_piece_and_capture(m.sq_from(), m.sq_to(), self.turn)?;
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            MoveFlags::EpCapture => {
                self.move_piece(m.sq_from(), m.sq_to(), self.turn)?;
                self.clear_piece_at(m.sq_to() + (SOUTH * self.turn), !self.turn)?;
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            MoveFlags::KingSideCastle => {
                self.move_piece(Square::E1 & self.turn, Square::G1 & self.turn, self.turn)?;
                self.move_piece(Square::H1 & self.turn, Square::F1 & self.turn, self.turn)?;
                self.castling_rights.remove_both_rights(self.turn);
                self.half_move_counter += 1;
            }
            MoveFlags::QueenSideCastle => {
                self.move_piece(Square::E1 & self.turn, Square::C1 & self.turn, self.turn)?;
                self.move_piece(Square::A1 & self.turn, Square::D1 & self.turn, self.turn)?;
                self.castling_rights.remove_both_rights(self.turn);
                self.half_move_counter += 1;
            }
            MoveFlags::KnightPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Knight, self.turn);
                self.half_move_counter = 0;
            }
            MoveFlags::BishopPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Bishop, self.turn);
                self.half_move_counter = 0;
            }
            MoveFlags::RookPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Rook, self.turn);
                self.half_move_counter = 0;
            }
            MoveFlags::QueenPromotion => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Queen, self.turn);
                self.half_move_counter = 0;
            }
            MoveFlags::KnightPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.clear_piece_at(m.sq_to(), !self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Knight, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            MoveFlags::BishopPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.clear_piece_at(m.sq_to(), !self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Bishop, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            MoveFlags::RookPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.clear_piece_at(m.sq_to(), !self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Rook, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
            MoveFlags::QueenPromotionCapture => {
                self.clear_piece_at(m.sq_from(), self.turn)?;
                self.clear_piece_at(m.sq_to(), !self.turn)?;
                self.put_piece_at(m.sq_to(), Piece::Queen, self.turn);
                self.piece_count -= 1;
                self.half_move_counter = 0;
            }
        };
        // Save previous zobrist hash and history plane
        self.add_history_plane(history_plane);
        self.zobrist_history.push(zobrist_hash);
        // Update board info
        self.maybe_update_castling_rights(m);
        self.zobrist_hash ^= self.lookups.zobrist_castling(self.castling_rights);
        self.ep_sq = ep_sq.and_then(|sq| Some(sq)); // Resets en-passant square if the move was not a double pawn push
        if self.turn == BLACK {
            self.zobrist_hash ^= self.lookups.zobrist_black();
        }
        self.turn = !self.turn;
        self.ply += 1;
        self.moves_history.push(*m);
        self.clear_cache();
        Ok(())
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
            MoveFlags::KingSideCastle
            | MoveFlags::QueenSideCastle
            | MoveFlags::DoublePawnPush
            | MoveFlags::EpCapture => return, // Skip for flags that dont need checking for update
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
            MoveFlags::Capture
            | MoveFlags::KnightPromotionCapture
            | MoveFlags::BishopPromotionCapture
            | MoveFlags::RookPromotionCapture
            | MoveFlags::QueenPromotionCapture => {}
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
            repetition_count: self.count_repetitions(2),
            zobrist_hash: self.zobrist_hash,
        }
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

#[cfg(test)]
mod board_tests {
    use super::*;

    #[test]
    fn test_board_new() {
        // Covers testing for starting position fen and empty fen
        // rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -
        let board = Board::new(None);
        assert_eq!(board.turn(), WHITE);
        assert_eq!(board.castling_rights(), "KQkq");
        assert_eq!(board.half_move_counter(), 0);
        assert_eq!(board.ply(), 0);
        assert_eq!(board.full_move_counter(), 1);
        assert_eq!(board.piece_count(), 32);
    }

    #[test]
    fn test_board_from_valid_fen() {
        let board = Board::new(Some(
            "r4rk1/pppnqppp/5n2/2bpp3/3PP1b1/2PBBN2/PPQN1PPP/1R2K2R b K - 10 9",
        ));
        assert_eq!(board.turn(), BLACK);
        assert_eq!(board.castling_rights(), "K");
        assert_eq!(board.half_move_counter(), 10);
        assert_eq!(board.ply(), 17);
        assert_eq!(board.full_move_counter(), 9);
        assert_eq!(board.piece_count(), 32);
    }
    #[test]
    #[should_panic(
        expected = "invalid FEN string: r3k2r/ppFnqppp/5n2/2bpp3/3PP1b1/2PBBN2/PPQN1PPP/R3K2R b KQkq - 8 8 (Invalid piece string: F)"
    )]
    fn test_board_from_invalid_fen() {
        let _board = Board::new(Some(
            "r3k2r/ppFnqppp/5n2/2bpp3/3PP1b1/2PBBN2/PPQN1PPP/R3K2R b KQkq - 8 8",
        ));
    }

    #[test]
    fn test_board_fullmoves_counter() {
        let mut board = Board::new(None);
        assert_eq!(board.full_move_counter(), 1);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        assert_eq!(board.full_move_counter(), 2);
        board.push_uci("e4e5").unwrap();
        assert_eq!(board.full_move_counter(), 2);
        board.push_uci("e1e2").unwrap();
        assert_eq!(board.full_move_counter(), 3);
    }

    #[test]
    fn test_board_halfmoves_incr() {
        // Non capturing piece moves should increment half move counter
        // Castling and moves that lose castling right also increment half move counter
        let mut board = Board::new(Some(
            "r3k2r/pppnqppp/5n2/2bpp3/3PP1b1/2PBBN2/PPQN1PPP/R3K2R b KQkq - 8 8",
        ));
        assert_eq!(board.half_move_counter(), 8);
        assert_eq!(board.piece_count(), 32);
        board.push_uci("f6h5").unwrap(); // Non capturing knight move
        assert_eq!(board.half_move_counter(), 9);
        board.push_uci("e3f5").unwrap(); // Non capturing bishop move
        assert_eq!(board.half_move_counter(), 10);
        board.push_uci("e8g8").unwrap(); // Castling
        assert_eq!(board.half_move_counter(), 11);
        board.push_uci("a1b1").unwrap(); // Non capturing rook move, also loses queenside castling right
        assert_eq!(board.half_move_counter(), 12);
        assert_eq!(board.piece_count(), 32);
    }

    #[test]
    fn test_board_halfmoves_reset() {
        // Capturing piece moves should reset half move counter to 0
        // Pawn moves including promotions should reset half move counter to 0
        let board = Board::new(Some(
            "r3k2r/pppnqppp/5n2/2bpp3/3PP1b1/2PBBN2/PPQN1PPP/R3K2R b KQkq - 8 8",
        ));
        assert_eq!(board.half_move_counter(), 8);
        assert_eq!(board.piece_count(), 32);
        let mut tmp_board = board.clone();
        tmp_board.push_uci("h7h6").unwrap(); // Pawn move
        assert_eq!(tmp_board.half_move_counter(), 0);
        tmp_board = board.clone();
        tmp_board.push_uci("f6e4").unwrap();
        assert_eq!(tmp_board.half_move_counter(), 0);
        assert_eq!(tmp_board.piece_count(), 31);
        assert_eq!(board.piece_count(), 32);
    }

    #[test]
    fn test_board_threefold_repetition() {
        let mut board = Board::new(None);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        board.push_uci("g1f3").unwrap();
        board.push_uci("g8f6").unwrap();
        board.push_uci("f3g1").unwrap();
        board.push_uci("f6g8").unwrap();
        board.push_uci("g1f3").unwrap();
        board.push_uci("g8f6").unwrap();
        board.push_uci("f3g1").unwrap();
        let (terminal, winner) = board.terminal();
        assert_eq!(terminal, false);
        assert_eq!(board.count_repetitions(3), 2);
        board.push_uci("f6g8").unwrap();
        let (terminal, winner) = board.terminal();
        assert_eq!(terminal, true);
        assert_eq!(board.count_repetitions(3), 3);
        assert_eq!(board.outcome(), Some(Outcome::ThreeFoldRepetition));
    }

    #[test]
    fn test_board_threefold_repetition2() {
        let mut board = Board::new(None);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        board.push_uci("g1f3").unwrap();
        board.push_uci("g8f6").unwrap();
        board.push_uci("f3g1").unwrap();
        board.push_uci("f6g8").unwrap();
        board.push_uci("g1f3").unwrap();
        board.push_uci("g8f6").unwrap();
        board.push_uci("d2d4").unwrap();
        board.push_uci("d7d5").unwrap();
        board.push_uci("f3g1").unwrap();
        let (terminal, winner) = board.terminal();
        assert_eq!(terminal, false);
        assert_eq!(board.count_repetitions(3), 1);
        board.push_uci("f6g8").unwrap();
        let (terminal, winner) = board.terminal();
        assert_eq!(terminal, false);
        assert_eq!(board.count_repetitions(3), 1);
        assert_eq!(board.outcome(), None);
    }

    #[test]
    fn test_board_insufficient_material() {
        // King vs king
        let board = Board::new(Some("8/8/4k3/8/4K3/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "K vs K should be draw by insufficient material"
        );
        // King vs king + knight
        let board = Board::new(Some("8/8/4k3/8/2N1K3/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "K vs K+N should be draw by insufficient material"
        );
        // King vs king + bishop
        let board = Board::new(Some("8/8/4k3/8/2B1K3/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "K vs K+B should be draw by insufficient material"
        );
        // King vs king + two knights
        let board = Board::new(Some("8/8/4k3/7N/2N1K3/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "K vs KNN should be draw by insufficient material"
        );
        // King + knight vs King + bishop
        let board = Board::new(Some("8/6b1/4k3/8/2N1K3/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "K+N vs K+B should be draw by insufficient material"
        );
        // King + knight vs King + knight
        let board = Board::new(Some("8/6n1/4k3/8/2N1K3/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "K+N vs K+N should be draw by insufficient material"
        );
        // King + knight & bishop vs King
        let board = Board::new(Some("8/8/4k3/8/2B1K1N/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            false,
            "KBN vs K should not be draw by insufficient material"
        );
        // King + two bishops of different colors vs King
        let board = Board::new(Some("8/4k3/8/8/4BB2/5K2/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            false,
            "KBB (different colors) vs K should not be draw by insufficient material"
        );
        // King + two bishops of same color vs King
        let board = Board::new(Some("8/4k3/8/5B2/4B3/5K2/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "KBB (same colors) vs K should be draw by insufficient material"
        );
        // King + two knights vs King + knight
        /*let board = Board::new(Some("8/6n1/4k3/8/2N1K1N1/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "KNN vs KN should be draw by insufficient material"
        );
        // King + two knights vs King + bishop
        let board = Board::new(Some("8/6b1/4k3/8/2N1K1N1/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "KNN vs KB should be draw by insufficient material"
        );
        // King + two bishop vs King + bishop
        let board = Board::new(Some("8/6b1/4k3/8/2B1K1B1/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "KBB vs KB should be draw by insufficient material"
        );
        // King + two bishop vs King + knight (not draw by insufficient material)
        let board = Board::new(Some("8/6n1/4k3/8/2B1K1B1/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "KBB vs KN should be draw by insufficient material (according to chess.com)"
        );
        // King + knight & bishop vs King (not draw by insufficient material)
        let board = Board::new(Some("8/8/4k3/8/2B1K1N1/8/8/8 w - - 0 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            false,
            "KBN vs K should not be draw by insufficient material"
        );
        // King + queen vs King + queen
        let board = Board::new(Some("8/4k1q1/8/3Q4/4K3/8/8/8 b - - 1 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            true,
            "KQ vs KQ should be draww by insufficient material"
        );
        // King + two rooks vs King + bishop & knight
        let board = Board::new(Some("6nb/4k3/8/8/4K3/3RR3/8/8 b - - 1 1"));
        assert_eq!(
            board.draw_by_insufficient_material(),
            false,
            "KRR vs KBN should not be draw by insufficient material (rooks win)"
        );*/
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn test_history_image_initial_position() {
        let board = Board::new(None);
        let (image, combined_hash) = board.history(false);
        assert_eq!(image.len(), 13 * 8 + 5);
        assert_eq!(combined_hash, board.zobrist_hash());
        for i in 13..(13 * 8) {
            assert_eq!(image[i], BB_EMPTY);
        }
        for i in (13 * 8)..(13 * 8 + 4) {
            assert_eq!(image[i], BB_FULL);
        }
        assert_eq!(image[image.len() - 1], BB_EMPTY);
    }

    #[test]
    fn test_history_image_with_lt8_moves() {
        let mut board = Board::new(None);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        let (image, combined_hash) = board.history(false);
        assert_eq!(image.len(), 13 * 8 + 5);
        for i in 0..3 {
            let from = 13 * i;
            let to = (13 * (i + 1)) - 2;
            for j in from..to {
                assert_ne!(image[j], BB_EMPTY);
            }
        }
        for i in (13 * 3)..(13 * 8) {
            assert_eq!(image[i], BB_EMPTY);
        }
        for i in (13 * 8)..(13 * 8 + 4) {
            assert_eq!(image[i], BB_FULL);
        }
        assert_eq!(image[image.len() - 1], BB_EMPTY);
    }

    #[test]
    fn test_history_image_with_lt8_moves_flipped() {
        let mut board = Board::new(None);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        let (image, combined_hash) = board.history(true);
        assert_eq!(image.len(), 13 * 8 + 5);
        for i in 0..3 {
            let from = 13 * i;
            let to = (13 * (i + 1)) - 2;
            for j in from..to {
                assert_ne!(image[j], BB_EMPTY);
            }
        }
        for i in (13 * 3)..(13 * 8) {
            assert_eq!(image[i], BB_EMPTY);
        }
        for i in (13 * 8)..(13 * 8 + 4) {
            assert_eq!(image[i], BB_FULL);
        }
        assert_eq!(image[image.len() - 1], BB_EMPTY);
    }

    #[test]
    fn test_history_image_with_gt8_moves() {
        let mut board = Board::new(None);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        board.push_uci("d2d4").unwrap();
        board.push_uci("d7d6").unwrap();
        board.push_uci("g1f3").unwrap();
        board.push_uci("g8f6").unwrap();
        board.push_uci("g2g3").unwrap();
        board.push_uci("b8c6").unwrap();
        board.push_uci("f1g2").unwrap();
        board.push_uci("f8e7").unwrap();
        board.push_uci("e1g1").unwrap();
        let (image, combined_hash) = board.history(false);
        assert_eq!(image.len(), 13 * 8 + 5);
        for i in 0..8 {
            let from = 13 * i;
            let to = (13 * (i + 1)) - 2;
            for j in from..to {
                assert_ne!(image[j], BB_EMPTY);
            }
        }
        // Blacks turn to move, so first two castling planes should be BB_FULL
        assert_eq!(image[104], BB_FULL);
        assert_eq!(image[105], BB_FULL);
        // White castled on last turn so both castling planes should be BB_EMPTY
        assert_eq!(image[106], BB_EMPTY);
        assert_eq!(image[107], BB_EMPTY);
        // No captures accured so last plane should be equal to moves played (ply)
        assert_eq!(image[108], Bitboard(4));
        assert_eq!(image[108], Bitboard(board.half_move_counter() as u64))
    }

    #[test]
    fn test_history_image_with_gt8_moves_flipped() {
        let mut board = Board::new(None);
        board.push_uci("e2e4").unwrap();
        board.push_uci("e7e5").unwrap();
        board.push_uci("d2d4").unwrap();
        board.push_uci("d7d6").unwrap();
        board.push_uci("g1f3").unwrap();
        board.push_uci("g8f6").unwrap();
        board.push_uci("g2g3").unwrap();
        board.push_uci("b8c6").unwrap();
        board.push_uci("f1g2").unwrap();
        board.push_uci("f8e7").unwrap();
        board.push_uci("e1g1").unwrap();
        let (image, combined_hash) = board.history(true);
        assert_eq!(image.len(), 13 * 8 + 5);
        for i in 0..8 {
            let from = 13 * i;
            let to = (13 * (i + 1)) - 2;
            for j in from..to {
                assert_ne!(image[j], BB_EMPTY);
            }
        }
        // Blacks turn to move, so first two castling planes should be BB_FULL
        assert_eq!(image[104], BB_FULL);
        assert_eq!(image[105], BB_FULL);
        // White castled on last turn so both castling planes should be BB_EMPTY
        assert_eq!(image[106], BB_EMPTY);
        assert_eq!(image[107], BB_EMPTY);
        // No captures accured so last plane should be equal to moves played (ply)
        assert_eq!(image[108], Bitboard(4));
        assert_eq!(image[108], Bitboard(board.half_move_counter() as u64))
    }
}
