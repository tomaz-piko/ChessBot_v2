use crate::bitboard::Bitboard;
use crate::square::Square;

pub const BB_EMPTY: Bitboard = Bitboard(0);
pub const BB_FULL: Bitboard = Bitboard(0xFFFFFFFFFFFFFFFF);

pub const BB_LIGHT_SQUARES: Bitboard = Bitboard(0x55AA_55AA_55AA_55AA);
pub const BB_DARK_SQUARES: Bitboard = Bitboard(0xAA55_AA55_AA55_AA55);

pub const BB_RANKS: [Bitboard; 8] = [
    Bitboard(0x0000_0000_0000_00FF),
    Bitboard(0x0000_0000_0000_FF00),
    Bitboard(0x0000_0000_00FF_0000),
    Bitboard(0x0000_0000_FF00_0000),
    Bitboard(0x0000_00FF_0000_0000),
    Bitboard(0x0000_FF00_0000_0000),
    Bitboard(0x00FF_0000_0000_0000),
    Bitboard(0xFF00_0000_0000_0000),
];

pub const BB_FILES: [Bitboard; 8] = [
    Bitboard(0x0101_0101_0101_0101),
    Bitboard(0x0202_0202_0202_0202),
    Bitboard(0x0404_0404_0404_0404),
    Bitboard(0x0808_0808_0808_0808),
    Bitboard(0x1010_1010_1010_1010),
    Bitboard(0x2020_2020_2020_2020),
    Bitboard(0x4040_4040_4040_4040),
    Bitboard(0x8080_8080_8080_8080),
];

pub const BB_DIAGONAL_R: [Bitboard; 15] = [
    Bitboard(0x0000_0000_0000_0080),
    Bitboard(0x0000_0000_0000_8040),
    Bitboard(0x0000_0000_0080_4020),
    Bitboard(0x0000_0000_8040_2010),
    Bitboard(0x0000_0080_4020_1008),
    Bitboard(0x0000_8040_2010_0804),
    Bitboard(0x0080_4020_1008_0402),
    Bitboard(0x8040_2010_0804_0201),
    Bitboard(0x4020_1008_0402_0100),
    Bitboard(0x2010_0804_0201_0000),
    Bitboard(0x1008_0402_0100_0000),
    Bitboard(0x0804_0201_0000_0000),
    Bitboard(0x0402_0100_0000_0000),
    Bitboard(0x0201_0000_0000_0000),
    Bitboard(0x0100_0000_0000_0000),
];

pub const BB_DIAGONAL_L: [Bitboard; 15] = [
    Bitboard(0x0000_0000_0000_0001),
    Bitboard(0x0000_0000_0000_0102),
    Bitboard(0x0000_0000_0001_0204),
    Bitboard(0x0000_0000_0102_0408),
    Bitboard(0x0000_0001_0204_0810),
    Bitboard(0x0000_0102_0408_1020),
    Bitboard(0x0001_0204_0810_2040),
    Bitboard(0x0102_0408_1020_4080),
    Bitboard(0x0204_0810_2040_8000),
    Bitboard(0x0408_1020_4080_0000),
    Bitboard(0x0810_2040_8000_0000),
    Bitboard(0x1020_4080_0000_0000),
    Bitboard(0x2040_8000_0000_0000),
    Bitboard(0x4080_0000_0000_0000),
    Bitboard(0x8000_0000_0000_0000),
];

pub const BB_SQUARES: [Bitboard; 64] = [
    Bitboard(1 << Square::A1 as u64),
    Bitboard(1 << Square::B1 as u64),
    Bitboard(1 << Square::C1 as u64),
    Bitboard(1 << Square::D1 as u64),
    Bitboard(1 << Square::E1 as u64),
    Bitboard(1 << Square::F1 as u64),
    Bitboard(1 << Square::G1 as u64),
    Bitboard(1 << Square::H1 as u64),
    Bitboard(1 << Square::A2 as u64),
    Bitboard(1 << Square::B2 as u64),
    Bitboard(1 << Square::C2 as u64),
    Bitboard(1 << Square::D2 as u64),
    Bitboard(1 << Square::E2 as u64),
    Bitboard(1 << Square::F2 as u64),
    Bitboard(1 << Square::G2 as u64),
    Bitboard(1 << Square::H2 as u64),
    Bitboard(1 << Square::A3 as u64),
    Bitboard(1 << Square::B3 as u64),
    Bitboard(1 << Square::C3 as u64),
    Bitboard(1 << Square::D3 as u64),
    Bitboard(1 << Square::E3 as u64),
    Bitboard(1 << Square::F3 as u64),
    Bitboard(1 << Square::G3 as u64),
    Bitboard(1 << Square::H3 as u64),
    Bitboard(1 << Square::A4 as u64),
    Bitboard(1 << Square::B4 as u64),
    Bitboard(1 << Square::C4 as u64),
    Bitboard(1 << Square::D4 as u64),
    Bitboard(1 << Square::E4 as u64),
    Bitboard(1 << Square::F4 as u64),
    Bitboard(1 << Square::G4 as u64),
    Bitboard(1 << Square::H4 as u64),
    Bitboard(1 << Square::A5 as u64),
    Bitboard(1 << Square::B5 as u64),
    Bitboard(1 << Square::C5 as u64),
    Bitboard(1 << Square::D5 as u64),
    Bitboard(1 << Square::E5 as u64),
    Bitboard(1 << Square::F5 as u64),
    Bitboard(1 << Square::G5 as u64),
    Bitboard(1 << Square::H5 as u64),
    Bitboard(1 << Square::A6 as u64),
    Bitboard(1 << Square::B6 as u64),
    Bitboard(1 << Square::C6 as u64),
    Bitboard(1 << Square::D6 as u64),
    Bitboard(1 << Square::E6 as u64),
    Bitboard(1 << Square::F6 as u64),
    Bitboard(1 << Square::G6 as u64),
    Bitboard(1 << Square::H6 as u64),
    Bitboard(1 << Square::A7 as u64),
    Bitboard(1 << Square::B7 as u64),
    Bitboard(1 << Square::C7 as u64),
    Bitboard(1 << Square::D7 as u64),
    Bitboard(1 << Square::E7 as u64),
    Bitboard(1 << Square::F7 as u64),
    Bitboard(1 << Square::G7 as u64),
    Bitboard(1 << Square::H7 as u64),
    Bitboard(1 << Square::A8 as u64),
    Bitboard(1 << Square::B8 as u64),
    Bitboard(1 << Square::C8 as u64),
    Bitboard(1 << Square::D8 as u64),
    Bitboard(1 << Square::E8 as u64),
    Bitboard(1 << Square::F8 as u64),
    Bitboard(1 << Square::G8 as u64),
    Bitboard(1 << Square::H8 as u64),
];

pub const BB_KNIGHT_ATTACKS: [Bitboard; 64] = [
    // Rank 1
    Bitboard(0x0000_0000_0002_0400),
    Bitboard(0x0000_0000_0005_0800),
    Bitboard(0x0000_0000_000A_1100),
    Bitboard(0x0000_0000_0014_2200),
    Bitboard(0x0000_0000_0028_4400),
    Bitboard(0x0000_0000_0050_8800),
    Bitboard(0x0000_0000_00A0_1000),
    Bitboard(0x0000_0000_0040_2000),
    // Rank 2
    Bitboard(0x0000_0000_0204_0004),
    Bitboard(0x0000_0000_0508_0008),
    Bitboard(0x0000_0000_0A11_0011),
    Bitboard(0x0000_0000_1422_0022),
    Bitboard(0x0000_0000_2844_0044),
    Bitboard(0x0000_0000_5088_0088),
    Bitboard(0x0000_0000_A010_0010),
    Bitboard(0x0000_0000_4020_0020),
    // Rank 3
    Bitboard(0x0000_0002_0400_0402),
    Bitboard(0x0000_0005_0800_0805),
    Bitboard(0x0000_000A_1100_110A),
    Bitboard(0x0000_0014_2200_2214),
    Bitboard(0x0000_0028_4400_4428),
    Bitboard(0x0000_0050_8800_8850),
    Bitboard(0x0000_00A0_1000_10A0),
    Bitboard(0x0000_0040_2000_2040),
    // Rank 4
    Bitboard(0x0000_0204_0004_0200),
    Bitboard(0x0000_0508_0008_0500),
    Bitboard(0x0000_0A11_0011_0A00),
    Bitboard(0x0000_1422_0022_1400),
    Bitboard(0x0000_2844_0044_2800),
    Bitboard(0x0000_5088_0088_5000),
    Bitboard(0x0000_A010_0010_A000),
    Bitboard(0x0000_4020_0020_4000),
    // Rank 5
    Bitboard(0x0002_0400_0402_0000),
    Bitboard(0x0005_0800_0805_0000),
    Bitboard(0x000A_1100_110A_0000),
    Bitboard(0x0014_2200_2214_0000),
    Bitboard(0x0028_4400_4428_0000),
    Bitboard(0x0050_8800_8850_0000),
    Bitboard(0x00A0_1000_10A0_0000),
    Bitboard(0x0040_2000_2040_0000),
    // Rank 6
    Bitboard(0x0204_0004_0200_0000),
    Bitboard(0x0508_0008_0500_0000),
    Bitboard(0x0A11_0011_0A00_0000),
    Bitboard(0x1422_0022_1400_0000),
    Bitboard(0x2844_0044_2800_0000),
    Bitboard(0x5088_0088_5000_0000),
    Bitboard(0xA010_0010_A000_0000),
    Bitboard(0x4020_0020_4000_0000),
    // Rank 7
    Bitboard(0x0400_0402_0000_0000),
    Bitboard(0x0800_0805_0000_0000),
    Bitboard(0x1100_110A_0000_0000),
    Bitboard(0x2200_2214_0000_0000),
    Bitboard(0x4400_4428_0000_0000),
    Bitboard(0x8800_8850_0000_0000),
    Bitboard(0x1000_10A0_0000_0000),
    Bitboard(0x2000_2040_0000_0000),
    // Rank 8
    Bitboard(0x0004_0200_0000_0000),
    Bitboard(0x0008_0500_0000_0000),
    Bitboard(0x0011_0A00_0000_0000),
    Bitboard(0x0022_1400_0000_0000),
    Bitboard(0x0044_2800_0000_0000),
    Bitboard(0x0088_5000_0000_0000),
    Bitboard(0x0010_A000_0000_0000),
    Bitboard(0x0020_4000_0000_0000),
];

pub const BB_KING_ATTACKS: [Bitboard; 64] = [
    // Rank 1
    Bitboard(0x0000_0000_0000_0302),
    Bitboard(0x0000_0000_0000_0705),
    Bitboard(0x0000_0000_0000_0E0A),
    Bitboard(0x0000_0000_0000_1C14),
    Bitboard(0x0000_0000_0000_3828),
    Bitboard(0x0000_0000_0000_7050),
    Bitboard(0x0000_0000_0000_E0A0),
    Bitboard(0x0000_0000_0000_C040),
    // Rank 2
    Bitboard(0x0000_0000_0003_0203),
    Bitboard(0x0000_0000_0007_0507),
    Bitboard(0x0000_0000_000E_0A0E),
    Bitboard(0x0000_0000_001C_141C),
    Bitboard(0x0000_0000_0038_2838),
    Bitboard(0x0000_0000_0070_5070),
    Bitboard(0x0000_0000_00E0_A0E0),
    Bitboard(0x0000_0000_00C0_40C0),
    // Rank 3
    Bitboard(0x0000_0000_0302_0300),
    Bitboard(0x0000_0000_0705_0700),
    Bitboard(0x0000_0000_0E0A_0E00),
    Bitboard(0x0000_0000_1C14_1C00),
    Bitboard(0x0000_0000_3828_3800),
    Bitboard(0x0000_0000_7050_7000),
    Bitboard(0x0000_0000_E0A0_E000),
    Bitboard(0x0000_0000_C040_C000),
    // Rank 4
    Bitboard(0x0000_0003_0203_0000),
    Bitboard(0x0000_0007_0507_0000),
    Bitboard(0x0000_000E_0A0E_0000),
    Bitboard(0x0000_001C_141C_0000),
    Bitboard(0x0000_0038_2838_0000),
    Bitboard(0x0000_0070_5070_0000),
    Bitboard(0x0000_00E0_A0E0_0000),
    Bitboard(0x0000_00C0_40C0_0000),
    // Rank 5
    Bitboard(0x0000_0302_0300_0000),
    Bitboard(0x0000_0705_0700_0000),
    Bitboard(0x0000_0E0A_0E00_0000),
    Bitboard(0x0000_1C14_1C00_0000),
    Bitboard(0x0000_3828_3800_0000),
    Bitboard(0x0000_7050_7000_0000),
    Bitboard(0x0000_E0A0_E000_0000),
    Bitboard(0x0000_C040_C000_0000),
    // Rank 6
    Bitboard(0x0003_0203_0000_0000),
    Bitboard(0x0007_0507_0000_0000),
    Bitboard(0x000E_0A0E_0000_0000),
    Bitboard(0x001C_141C_0000_0000),
    Bitboard(0x0038_2838_0000_0000),
    Bitboard(0x0070_5070_0000_0000),
    Bitboard(0x00E0_A0E0_0000_0000),
    Bitboard(0x00C0_40C0_0000_0000),
    // Rank 7
    Bitboard(0x0302_0300_0000_0000),
    Bitboard(0x0705_0700_0000_0000),
    Bitboard(0x0E0A_0E00_0000_0000),
    Bitboard(0x1C14_1C00_0000_0000),
    Bitboard(0x3828_3800_0000_0000),
    Bitboard(0x7050_7000_0000_0000),
    Bitboard(0xE0A0_E000_0000_0000),
    Bitboard(0xC040_C000_0000_0000),
    // Rank 8
    Bitboard(0x0203_0000_0000_0000),
    Bitboard(0x0507_0000_0000_0000),
    Bitboard(0x0A0E_0000_0000_0000),
    Bitboard(0x141C_0000_0000_0000),
    Bitboard(0x2838_0000_0000_0000),
    Bitboard(0x5070_0000_0000_0000),
    Bitboard(0xA0E0_0000_0000_0000),
    Bitboard(0x40C0_0000_0000_0000),
];

pub const BB_PAWN_ATTACKS_W: [Bitboard; 64] = [
    // Rank 1
    Bitboard(0x0000_0000_0000_0200),
    Bitboard(0x0000_0000_0000_0500),
    Bitboard(0x0000_0000_0000_0A00),
    Bitboard(0x0000_0000_0000_1400),
    Bitboard(0x0000_0000_0000_2800),
    Bitboard(0x0000_0000_0000_5000),
    Bitboard(0x0000_0000_0000_A000),
    Bitboard(0x0000_0000_0000_4000),
    // Rank 2
    Bitboard(0x0000_0000_0002_0000),
    Bitboard(0x0000_0000_0005_0000),
    Bitboard(0x0000_0000_000A_0000),
    Bitboard(0x0000_0000_0014_0000),
    Bitboard(0x0000_0000_0028_0000),
    Bitboard(0x0000_0000_0050_0000),
    Bitboard(0x0000_0000_00A0_0000),
    Bitboard(0x0000_0000_0040_0000),
    // Rank 3
    Bitboard(0x0000_0000_0200_0000),
    Bitboard(0x0000_0000_0500_0000),
    Bitboard(0x0000_0000_0A00_0000),
    Bitboard(0x0000_0000_1400_0000),
    Bitboard(0x0000_0000_2800_0000),
    Bitboard(0x0000_0000_5000_0000),
    Bitboard(0x0000_0000_A000_0000),
    Bitboard(0x0000_0000_4000_0000),
    // Rank 4
    Bitboard(0x0000_0002_0000_0000),
    Bitboard(0x0000_0005_0000_0000),
    Bitboard(0x0000_000A_0000_0000),
    Bitboard(0x0000_0014_0000_0000),
    Bitboard(0x0000_0028_0000_0000),
    Bitboard(0x0000_0050_0000_0000),
    Bitboard(0x0000_00A0_0000_0000),
    Bitboard(0x0000_0040_0000_0000),
    // Rank 5
    Bitboard(0x0000_0200_0000_0000),
    Bitboard(0x0000_0500_0000_0000),
    Bitboard(0x0000_0A00_0000_0000),
    Bitboard(0x0000_1400_0000_0000),
    Bitboard(0x0000_2800_0000_0000),
    Bitboard(0x0000_5000_0000_0000),
    Bitboard(0x0000_A000_0000_0000),
    Bitboard(0x0000_4000_0000_0000),
    // Rank 6
    Bitboard(0x0002_0000_0000_0000),
    Bitboard(0x0005_0000_0000_0000),
    Bitboard(0x000A_0000_0000_0000),
    Bitboard(0x0014_0000_0000_0000),
    Bitboard(0x0028_0000_0000_0000),
    Bitboard(0x0050_0000_0000_0000),
    Bitboard(0x00A0_0000_0000_0000),
    Bitboard(0x0040_0000_0000_0000),
    // Rank 7
    Bitboard(0x0200_0000_0000_0000),
    Bitboard(0x0500_0000_0000_0000),
    Bitboard(0x0A00_0000_0000_0000),
    Bitboard(0x1400_0000_0000_0000),
    Bitboard(0x2800_0000_0000_0000),
    Bitboard(0x5000_0000_0000_0000),
    Bitboard(0xA000_0000_0000_0000),
    Bitboard(0x4000_0000_0000_0000),
    // Rank 8
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
];

pub const BB_PAWN_ATTACKS_B: [Bitboard; 64] = [
    // Rank 1
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    Bitboard(0x0000_0000_0000_0000),
    // Rank 2
    Bitboard(0x0000_0000_0000_0002),
    Bitboard(0x0000_0000_0000_0005),
    Bitboard(0x0000_0000_0000_000A),
    Bitboard(0x0000_0000_0000_0014),
    Bitboard(0x0000_0000_0000_0028),
    Bitboard(0x0000_0000_0000_0050),
    Bitboard(0x0000_0000_0000_00A0),
    Bitboard(0x0000_0000_0000_0040),
    // Rank 3
    Bitboard(0x0000_0000_0000_0200),
    Bitboard(0x0000_0000_0000_0500),
    Bitboard(0x0000_0000_0000_0A00),
    Bitboard(0x0000_0000_0000_1400),
    Bitboard(0x0000_0000_0000_2800),
    Bitboard(0x0000_0000_0000_5000),
    Bitboard(0x0000_0000_0000_A000),
    Bitboard(0x0000_0000_0000_4000),
    // Rank 4
    Bitboard(0x0000_0000_0002_0000),
    Bitboard(0x0000_0000_0005_0000),
    Bitboard(0x0000_0000_000A_0000),
    Bitboard(0x0000_0000_0014_0000),
    Bitboard(0x0000_0000_0028_0000),
    Bitboard(0x0000_0000_0050_0000),
    Bitboard(0x0000_0000_00A0_0000),
    Bitboard(0x0000_0000_0040_0000),
    // Rank 5
    Bitboard(0x0000_0000_0200_0000),
    Bitboard(0x0000_0000_0500_0000),
    Bitboard(0x0000_0000_0A00_0000),
    Bitboard(0x0000_0000_1400_0000),
    Bitboard(0x0000_0000_2800_0000),
    Bitboard(0x0000_0000_5000_0000),
    Bitboard(0x0000_0000_A000_0000),
    Bitboard(0x0000_0000_4000_0000),
    // Rank 6
    Bitboard(0x0000_0002_0000_0000),
    Bitboard(0x0000_0005_0000_0000),
    Bitboard(0x0000_000A_0000_0000),
    Bitboard(0x0000_0014_0000_0000),
    Bitboard(0x0000_0028_0000_0000),
    Bitboard(0x0000_0050_0000_0000),
    Bitboard(0x0000_00A0_0000_0000),
    Bitboard(0x0000_0040_0000_0000),
    // Rank 7
    Bitboard(0x0000_0200_0000_0000),
    Bitboard(0x0000_0500_0000_0000),
    Bitboard(0x0000_0A00_0000_0000),
    Bitboard(0x0000_1400_0000_0000),
    Bitboard(0x0000_2800_0000_0000),
    Bitboard(0x0000_5000_0000_0000),
    Bitboard(0x0000_A000_0000_0000),
    Bitboard(0x0000_4000_0000_0000),
    // Rank 8
    Bitboard(0x0002_0000_0000_0000),
    Bitboard(0x0005_0000_0000_0000),
    Bitboard(0x000A_0000_0000_0000),
    Bitboard(0x0014_0000_0000_0000),
    Bitboard(0x0028_0000_0000_0000),
    Bitboard(0x0050_0000_0000_0000),
    Bitboard(0x00A0_0000_0000_0000),
    Bitboard(0x0040_0000_0000_0000),
];

pub const BB_PAWN_ATTACKS: [[Bitboard; 64]; 2] = [BB_PAWN_ATTACKS_B, BB_PAWN_ATTACKS_W];

pub const DEBRUIJN_64: [u16; 64] = [
    0, 47, 1, 56, 48, 27, 2, 60, 57, 49, 41, 37, 28, 16, 3, 61, 54, 58, 35, 52, 50, 42, 21, 44, 38,
    32, 29, 23, 17, 11, 4, 62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45, 25,
    39, 14, 33, 19, 30, 9, 24, 13, 18, 8, 12, 7, 6, 5, 63,
];

pub const DEBRUIJN_MAGIC: Bitboard = Bitboard(0x03f79d71b4cb0a89);

pub const A2A7: Bitboard = Bitboard(0x0001010101010100);
pub const B2G7: Bitboard = Bitboard(0x0040201008040200);
pub const H1B7: Bitboard = Bitboard(0x0002040810204080);

pub const BB_WHITE_O_O_OCC: Bitboard = Bitboard(0x0000000000000060);
pub const BB_WHITE_O_O_O_OCC: Bitboard = Bitboard(0x000000000000000E);
pub const BB_BLACK_O_O_OCC: Bitboard = Bitboard(0x6000000000000000);
pub const BB_BLACK_O_O_O_OCC: Bitboard = Bitboard(0x0E00000000000000);

pub const BB_WHITE_0_0_IGNORE_DANGER: Bitboard = Bitboard(0x0000000000000002);
pub const BB_BLACK_0_0_IGNORE_DANGER: Bitboard = Bitboard(0x0200000000000000);

pub const BB_0_0_OCC: [Bitboard; 2] = [BB_BLACK_O_O_OCC, BB_WHITE_O_O_OCC];

pub const BB_0_0_O_OCC: [Bitboard; 2] = [BB_BLACK_O_O_O_OCC, BB_WHITE_O_O_O_OCC];

pub const BB_0_0_IGNORE_DANGER: [Bitboard; 2] =
    [BB_BLACK_0_0_IGNORE_DANGER, BB_WHITE_0_0_IGNORE_DANGER];
