use crate::types::square::Square;
use std::fmt::{Debug, Display};

#[derive(Debug)]
pub enum MoveFlags {
    QuietMove = 0b0000,
    DoublePawnPush = 0b0001,
    KingSideCastle = 0b0010,
    QueenSideCastle = 0b0011,
    Capture = 0b0100,
    EpCapture = 0b0101,
    KnightPromotion = 0b1000,
    BishopPromotion = 0b1001,
    RookPromotion = 0b1010,
    QueenPromotion = 0b1011,
    KnightPromotionCapture = 0b1100,
    BishopPromotionCapture = 0b1101,
    RookPromotionCapture = 0b1110,
    QueenPromotionCapture = 0b1111,
}

impl From<u16> for MoveFlags {
    fn from(value: u16) -> Self {
        match value {
            0b0000 => MoveFlags::QuietMove,
            0b0001 => MoveFlags::DoublePawnPush,
            0b0010 => MoveFlags::KingSideCastle,
            0b0011 => MoveFlags::QueenSideCastle,
            0b0100 => MoveFlags::Capture,
            0b0101 => MoveFlags::EpCapture,
            0b1000 => MoveFlags::KnightPromotion,
            0b1001 => MoveFlags::BishopPromotion,
            0b1010 => MoveFlags::RookPromotion,
            0b1011 => MoveFlags::QueenPromotion,
            0b1100 => MoveFlags::KnightPromotionCapture,
            0b1101 => MoveFlags::BishopPromotionCapture,
            0b1110 => MoveFlags::RookPromotionCapture,
            0b1111 => MoveFlags::QueenPromotionCapture,
            _ => panic!("Invalid flag value"),
        }
    }
}

impl Display for MoveFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoveFlags::QuietMove => write!(f, "QuietMove"),
            MoveFlags::DoublePawnPush => write!(f, "DoublePawnPush"),
            MoveFlags::KingSideCastle => write!(f, "KingSideCastle"),
            MoveFlags::QueenSideCastle => write!(f, "QueenSideCastle"),
            MoveFlags::Capture => write!(f, "Capture"),
            MoveFlags::EpCapture => write!(f, "EpCapture"),
            MoveFlags::KnightPromotion => write!(f, "KnightPromotion"),
            MoveFlags::BishopPromotion => write!(f, "BishopPromotion"),
            MoveFlags::RookPromotion => write!(f, "RookPromotion"),
            MoveFlags::QueenPromotion => write!(f, "QueenPromotion"),
            MoveFlags::KnightPromotionCapture => write!(f, "KnightPromotionCapture"),
            MoveFlags::BishopPromotionCapture => write!(f, "BishopPromotionCapture"),
            MoveFlags::RookPromotionCapture => write!(f, "RookPromotionCapture"),
            MoveFlags::QueenPromotionCapture => write!(f, "QueenPromotionCapture"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Move(pub u16);

impl Move {
    #[inline(always)]
    pub fn new(from: Square, to: Square, flags: MoveFlags) -> Self {
        let from: u16 = from as u16;
        let to: u16 = to as u16;
        let flags: u16 = flags as u16;
        Move((flags << 12) | (from << 6) | to)
    }

    #[inline(always)]
    pub fn sq_from(&self) -> Square {
        Square::from((self.0 >> 6) & 0x3f)
    }

    #[inline(always)]
    pub fn sq_to(&self) -> Square {
        Square::from(self.0 & 0x3f)
    }

    #[inline(always)]
    pub fn flags(&self) -> MoveFlags {
        MoveFlags::from((self.0 >> 12) & 0b1111)
    }

    #[inline(always)]
    pub fn uci(&self) -> String {
        let promotion = match self.flags() {
            MoveFlags::KnightPromotion | MoveFlags::KnightPromotionCapture => "n",
            MoveFlags::BishopPromotion | MoveFlags::BishopPromotionCapture => "b",
            MoveFlags::RookPromotion | MoveFlags::RookPromotionCapture => "r",
            MoveFlags::QueenPromotion | MoveFlags::QueenPromotionCapture => "q",
            _ => "",
        };
        format!("{:?}{:?}{}", self.sq_from(), self.sq_to(), promotion).to_lowercase()
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.uci())
    }
}

impl Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} -> {:?} | {}",
            self.sq_from(),
            self.sq_to(),
            self.flags()
        )
    }
}
