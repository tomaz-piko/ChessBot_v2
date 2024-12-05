use thiserror::Error;

#[derive(Error, Debug)]
pub enum BoardError {
    #[error("invalid move: {0}")]
    InvalidMove(String),
    #[error("invalid UCI string: {0}")]
    InvalidUci(String),

    #[error("invalid FEN string: {0} ({1})")]
    InvalidFen(String, String),
}