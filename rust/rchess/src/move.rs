use crate::square::Square;
use std::fmt::{Debug, Display};

#[derive(Debug)]
pub enum Flags {
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

impl From<u16> for Flags {
    fn from(value: u16) -> Self {
        match value {
            0b0000 => Flags::QuietMove,
            0b0001 => Flags::DoublePawnPush,
            0b0010 => Flags::KingSideCastle,
            0b0011 => Flags::QueenSideCastle,
            0b0100 => Flags::Capture,
            0b0101 => Flags::EpCapture,
            0b1000 => Flags::KnightPromotion,
            0b1001 => Flags::BishopPromotion,
            0b1010 => Flags::RookPromotion,
            0b1011 => Flags::QueenPromotion,
            0b1100 => Flags::KnightPromotionCapture,
            0b1101 => Flags::BishopPromotionCapture,
            0b1110 => Flags::RookPromotionCapture,
            0b1111 => Flags::QueenPromotionCapture,
            _ => panic!("Invalid flag value"),
        }
    }
}

impl Display for Flags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Flags::QuietMove => write!(f, "QuietMove"),
            Flags::DoublePawnPush => write!(f, "DoublePawnPush"),
            Flags::KingSideCastle => write!(f, "KingSideCastle"),
            Flags::QueenSideCastle => write!(f, "QueenSideCastle"),
            Flags::Capture => write!(f, "Capture"),
            Flags::EpCapture => write!(f, "EpCapture"),
            Flags::KnightPromotion => write!(f, "KnightPromotion"),
            Flags::BishopPromotion => write!(f, "BishopPromotion"),
            Flags::RookPromotion => write!(f, "RookPromotion"),
            Flags::QueenPromotion => write!(f, "QueenPromotion"),
            Flags::KnightPromotionCapture => write!(f, "KnightPromotionCapture"),
            Flags::BishopPromotionCapture => write!(f, "BishopPromotionCapture"),
            Flags::RookPromotionCapture => write!(f, "RookPromotionCapture"),
            Flags::QueenPromotionCapture => write!(f, "QueenPromotionCapture"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Move(u16);

impl Move {
    #[inline(always)]
    pub fn new(from: Square, to: Square, flags: Flags) -> Self {
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
    pub fn flags(&self) -> Flags {
        Flags::from((self.0 >> 12) & 0b1111)
    }

    #[inline(always)]
    pub fn uci(&self) -> String {
        let promotion = match self.flags() {
            Flags::KnightPromotion | Flags::KnightPromotionCapture => "n",
            Flags::BishopPromotion | Flags::BishopPromotionCapture => "b",
            Flags::RookPromotion | Flags::RookPromotionCapture => "r",
            Flags::QueenPromotion | Flags::QueenPromotionCapture => "q",
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
