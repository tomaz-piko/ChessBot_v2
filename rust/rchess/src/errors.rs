use thiserror::Error;

#[derive(Error, Debug)]
pub enum MakeMoveError {
    #[error("invalid move: {0}")]
    InvalidMove(String),
    #[error("invalid UCI string: {0}")]
    InvalidUci(String),
}