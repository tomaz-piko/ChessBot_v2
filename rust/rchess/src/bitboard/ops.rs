use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign};
use crate::bitboard::Bitboard;


// // // // // // // // // // // // //
//                                  //
// Shifting and Bitwise Operations  //
//                                  //
// // // // // // // // // // // // //

impl Shl<usize> for Bitboard {
    type Output = Bitboard;

    fn shl(self, rhs: usize) -> Self::Output {
        Bitboard(self.0 << rhs)
    }
}

impl Shl<usize> for &Bitboard {
    type Output = Bitboard;

    fn shl(self, rhs: usize) -> Self::Output {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<usize> for Bitboard {
    type Output = Bitboard;

    fn shr(self, rhs: usize) -> Self::Output {
        Bitboard(self.0 >> rhs)
    }
}

impl Shr<usize> for &Bitboard {
    type Output = Bitboard;

    fn shr(self, rhs: usize) -> Self::Output {
        Bitboard(self.0 >> rhs)
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Bitboard(!self.0)
    }
}

impl Neg for Bitboard {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Bitboard(self.0.wrapping_neg())
    }
}

impl BitAnd for Bitboard {
    type Output = Bitboard;

    fn bitand(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAnd<u64> for Bitboard {
    type Output = Bitboard;

    fn bitand(self, rhs: u64) -> Self::Output {
        Bitboard(self.0 & rhs)
    }
}

impl BitAnd for &Bitboard {
    type Output = Bitboard;

    fn bitand(self, rhs: &Bitboard) -> Self::Output {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitAndAssign for &mut Bitboard {
    fn bitand_assign(&mut self, rhs: &mut Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;

    fn bitor(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOr for &Bitboard {
    type Output = Bitboard;

    fn bitor(self, rhs: &Bitboard) -> Self::Output {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign for &mut Bitboard {
    fn bitor_assign(&mut self, rhs: &mut Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;

    fn bitxor(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXor for &Bitboard {
    type Output = Bitboard;

    fn bitxor(self, rhs: &Bitboard) -> Self::Output {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl BitXorAssign for &mut Bitboard {
    fn bitxor_assign(&mut self, rhs: &mut Bitboard) {
        self.0 ^= rhs.0;
    }
}


// // // // // // // // // // // // //
//                                  //
//       Numerical operations       //
//                                  //
// // // // // // // // // // // // //

impl Add<Bitboard> for Bitboard {
    type Output = Bitboard;

    fn add(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0.wrapping_add(rhs.0))
    }
}

impl Add<Bitboard> for &Bitboard {
    type Output = Bitboard;

    fn add(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0.wrapping_add(rhs.0))
    }
}

impl AddAssign<Bitboard> for Bitboard {
    fn add_assign(&mut self, rhs: Bitboard) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl Sub<Bitboard> for Bitboard {
    type Output = Bitboard;

    fn sub(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0.wrapping_sub(rhs.0))
    }
}

impl Sub<Bitboard> for &Bitboard {
    type Output = Bitboard;

    fn sub(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0.wrapping_sub(rhs.0))
    }
}

impl Sub<u64> for Bitboard {
    type Output = Bitboard;

    fn sub(self, rhs: u64) -> Self::Output {
        Bitboard(self.0.wrapping_sub(rhs))
    }
}

impl SubAssign<Bitboard> for Bitboard {
    fn sub_assign(&mut self, rhs: Bitboard) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

impl Mul for Bitboard {
    type Output = Bitboard;

    fn mul(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.0.wrapping_mul(rhs.0))
    }
}

impl Mul<u64> for Bitboard {
    type Output = Bitboard;

    fn mul(self, rhs: u64) -> Self::Output {
        Bitboard(self.0.wrapping_mul(rhs))
    }
}

impl Mul<Bitboard> for u64 {
    type Output = Bitboard;

    fn mul(self, rhs: Bitboard) -> Self::Output {
        Bitboard(self.wrapping_mul(rhs.0))
    }
}

impl MulAssign for Bitboard {
    fn mul_assign(&mut self, rhs: Bitboard) {
        self.0 = self.0.wrapping_mul(rhs.0);
    }
}