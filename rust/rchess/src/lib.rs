mod bitboard;
pub mod board;
mod errors;
pub mod perft;
mod statics;
mod types;

use pyo3::prelude::*;

#[pymodule]
mod _lib {
    use std::collections::hash_map;
    use std::sync::RwLock;

    use crate::board;
    use crate::types::color;
    use crate::types::r#move;
    use pyo3::prelude::*;

    #[pyclass]
    struct Board {
        board: board::Board,
    }

    #[pymethods]
    impl Board {
        #[new]
        #[pyo3(signature = (fen=None))]
        fn new(fen: Option<&str>) -> Self {
            Self {
                board: board::Board::new(fen),
            }
        }

        fn __repr__(&self) -> String {
            format!("{}", self.board)
        }

        fn fen(&self) -> String {
            self.board.fen()
        }

        fn push_uci(&mut self, uci: &str) -> PyResult<()> {
            if let Err(err) = self.board.push_uci(uci) {
                panic!("{}", err)
            };
            Ok(())
        }

        fn push_num(&mut self, num: u16) -> PyResult<()> {
            let m = r#move::Move(num);
            if let Err(err) = self.board.push(&m) {
                panic!("{}", err)
            };
            Ok(())
        }

        fn push(&mut self, m: Move) -> PyResult<()> {
            if let Err(err) = self.board.push(&m.r#move) {
                panic!("{}", err)
            };
            Ok(())
        }

        #[pyo3(signature = (take=None))]
        fn moves_history(&self, take: Option<usize>) -> Vec<u16> {
            let take = take.unwrap_or(usize::MAX);
            self.board
                .moves_history()
                .iter()
                .rev()
                .take(take)
                .map(|m| m.0)
                .rev()
                .collect()
        }

        fn legal_moves_uci(&mut self) -> Vec<String> {
            self.board.legal_moves().iter().map(|m| m.uci()).collect()
        }

        fn legal_moves_num(&mut self) -> Vec<u16> {
            self.board.legal_moves().iter().map(|m| m.0).collect()
        }

        fn legal_moves_tuple(&mut self) -> Vec<(u16, String)> {
            self.board
                .legal_moves()
                .iter()
                .map(|m| (m.0, m.uci()))
                .collect()
        }

        fn legal_moves(&mut self) -> Vec<Move> {
            self.board
                .legal_moves()
                .iter()
                .map(|m| Move { r#move: *m })
                .collect()
        }

        fn history_hash(&self) -> u64 {
            self.board.history_hash()
        }

        fn history(&self, flip_uneven: bool) -> (Vec<i64>, u64) {
            let (image, hash) = self.board.history(flip_uneven);
            (image.iter().map(|&x| x.0 as i64).collect(), hash)
        }

        fn pieces_on_board(&self) -> u8 {
            self.board.piece_count()
        }

        fn clone(&self) -> Self {
            Self {
                board: self.board.clone(),
            }
        }

        fn outcome_str(&self) -> String {
            match self.board.outcome() {
                Some(outcome) => match outcome {
                    board::Outcome::Checkmate => "Checkmate".to_string(),
                    board::Outcome::Stalemate => "Stalemate".to_string(),
                    board::Outcome::FiftyMoveRule => "FiftyMoveRule".to_string(),
                    board::Outcome::ThreeFoldRepetition => "ThreefoldRepetition".to_string(),
                    board::Outcome::InsufficientMaterial => "InsufficientMaterial".to_string(),
                },
                None => "InProgress".to_string(),
            }
        }

        fn terminal(&mut self) -> (bool, Option<bool>) {
            let (is_terminal, winner) = self.board.terminal();
            (is_terminal, winner.map(|c| c == color::Color::White))
        }

        fn mid_search_terminal(&mut self, depth_to_root: usize) -> (bool, bool) {
            self.board.mid_search_terminal(depth_to_root)
        }

        fn to_play(&self) -> bool {
            self.board.turn() == color::Color::White
        }

        fn ply(&self) -> u16 {
            self.board.ply()
        }

        fn hash(&self) -> u64 {
            self.board.zobrist_hash()
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Move {
        r#move: r#move::Move,
    }

    #[pymethods]
    impl Move {
        #[new]
        fn new(num: u16) -> Self {
            Self {
                r#move: r#move::Move(num),
            }
        }

        fn __repr__(&self) -> String {
            format!("{}", self.r#move)
        }

        fn __hash__(&self) -> u64 {
            self.r#move.0 as u64
        }

        fn uci(&self) -> String {
            self.r#move.uci()
        }
    }

    #[pyclass]
    struct Cache {
        //store: hash_map::HashMap<u64, Vec<u16>>
        //store safe for threading
        store: hash_map::HashMap<u64, u64>,
        lock: RwLock<()>,
    }

    #[pymethods]
    impl Cache {
        #[new]
        fn new() -> Self {
            Self {
                store: hash_map::HashMap::new(),
                lock: RwLock::new(()),
            }
        }

        fn get(&self, key: u64) -> Option<u64> {
            let _lock = self.lock.read().unwrap();
            self.store.get(&key).copied()
        }

        fn set(&mut self, key: u64, value: u64) {
            let _lock = self.lock.write().unwrap();
            self.store.insert(key, value);
        }
    }
}
