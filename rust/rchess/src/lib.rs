mod bitboard;
pub mod board;
mod errors;
pub mod perft;
mod statics;
mod types;

use pyo3::prelude::*;

#[pymodule]
mod _lib {
    use crate::board;
    use crate::types::color;
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

        fn to_string(&self) -> String {
            format!("{}", self.board)
        }

        fn push_uci(&mut self, uci: &str) -> PyResult<()> {
            if let Err(err) = self.board.push_uci(uci) {
                panic!("{}", err)
            };
            Ok(())
        }

        fn legal_moves(&mut self) -> Vec<String> {
            self.board.legal_moves().iter().map(|m| m.uci()).collect()
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
                    board::Outcome::FiftyMoveRule => "FiftyMoveRules".to_string(),
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

    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
