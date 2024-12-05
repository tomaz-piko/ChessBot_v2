#![allow(static_mut_refs)]
use crate::bitboard::masks::{BB_EMPTY, BB_FILES, BB_RANKS};
use crate::bitboard::Bitboard;
use crate::board::{AFILE, HFILE, RANK1, RANK8};
use crate::types::piece::Piece;
use crate::statics::prng::Prng;
use crate::types::square::SQUARES;
use std::ptr;

const SEEDS: [[u64; 8]; 2] = [
    [8977, 44_560, 54_343, 38_998, 5731, 95_205, 104_912, 17_020],
    [728, 10_316, 55_013, 32_803, 12_281, 15_100, 16_645, 255],
];

const BISHOP_ATTACKS_SIZE: usize = 5248;
static mut BISHOP_MAGICS: [Magic; 64] = [Magic::init_const(); 64];
static mut BISHOP_ATTACKS: [Bitboard; BISHOP_ATTACKS_SIZE] = [BB_EMPTY; BISHOP_ATTACKS_SIZE];

const ROOK_ATTACKS_SIZE: usize = 102400;
static mut ROOK_MAGICS: [Magic; 64] = [Magic::init_const(); 64];
static mut ROOK_ATTACKS: [Bitboard; ROOK_ATTACKS_SIZE] = [BB_EMPTY; ROOK_ATTACKS_SIZE];

#[inline]
pub fn bishop_attacks(sq: usize, blockers: Bitboard) -> Bitboard {
    unsafe {
        let magic = BISHOP_MAGICS.get_unchecked(sq);
        *(magic.ptr as *const Bitboard).add(magic.index(blockers))
    }
}

#[inline]
pub fn rook_attacks(sq: usize, blockers: Bitboard) -> Bitboard {
    unsafe {
        let magic = ROOK_MAGICS.get_unchecked(sq);
        *(magic.ptr as *const Bitboard).add(magic.index(blockers))
    }
}

#[derive(Copy, Clone)]
pub struct TmpMagic {
    pub start: usize,
    pub len: usize,
    pub mask: Bitboard,
    pub magic: u64,
    pub shift: u32,
}

impl Default for TmpMagic {
    fn default() -> TmpMagic {
        TmpMagic {
            start: 0,
            len: 0,
            mask: BB_EMPTY,
            magic: 0,
            shift: 0,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Magic {
    ptr: usize,
    mask: Bitboard,
    magic: u64,
    shift: u32,
}

impl Magic {
    pub const fn init_const() -> Magic {
        Magic {
            ptr: 0,
            mask: BB_EMPTY,
            magic: 0,
            shift: 0,
        }
    }

    #[inline(always)]
    pub fn index(&self, blockers: Bitboard) -> usize {
        let index = (blockers & self.mask)
            .0
            .wrapping_mul(self.magic)
            .wrapping_shr(self.shift);
        index as usize
    }
}

fn get_index(blockers: Bitboard, mask: Bitboard, magic: u64, shift: u32) -> usize {
    let index = (blockers & mask).0.wrapping_mul(magic).wrapping_shr(shift);
    index as usize
}

#[cold]
pub fn init_magics() {
    unsafe {
        generate_magics(
            Piece::Bishop,
            BISHOP_MAGICS.as_mut_ptr(),
            BISHOP_ATTACKS.as_mut_ptr(),
        );
        generate_magics(
            Piece::Rook,
            ROOK_MAGICS.as_mut_ptr(),
            ROOK_ATTACKS.as_mut_ptr(),
        );
    }
}

#[cold]
unsafe fn generate_magics(piece: Piece, magics: *mut Magic, attacks: *mut Bitboard) {
    let mut start = 0;
    let mut tmp_magics: Vec<TmpMagic> = vec![TmpMagic::default(); 64];
    for (sq_idx, sq) in SQUARES.iter().enumerate() {
        let edges = ((BB_FILES[AFILE] | BB_FILES[HFILE]) & !BB_FILES[sq.file_idx()])
            | ((BB_RANKS[RANK1] | BB_RANKS[RANK8]) & !BB_RANKS[sq.rank_idx()]);
        let mask = match piece {
            Piece::Bishop => Bitboard::calc_diag_attacks(*sq, Bitboard::default()) & !edges,
            Piece::Rook => Bitboard::calc_orth_attacks(*sq, Bitboard::default()) & !edges,
            _ => panic!("Invalid piece type"),
        };
        let shift: u32 = 64 - mask.pop_count();
        let mut rng = Prng::init(SEEDS[1][sq.rank_idx()]);

        let mut possible_blockers: Vec<Bitboard> = Vec::new();
        let mut blockers: Bitboard = Bitboard::default();
        'carry_rippler: loop {
            possible_blockers.push(blockers);
            // Switch to the next occupancy_bb variation
            blockers = (blockers - mask) & mask;
            if blockers == BB_EMPTY {
                break 'carry_rippler;
            }
        }
        let mut magic: u64;
        let size = possible_blockers.len();
        tmp_magics[sq_idx].len = size;
        if sq_idx < 63 {
            tmp_magics[sq_idx + 1].start = start + size;
        }
        'table_fill: loop {
            'magic_select: loop {
                magic = rng.sparse_rand();
                if magic.wrapping_mul(mask.0).wrapping_shr(56).count_ones() >= 6 {
                    break 'magic_select;
                }
            }
            let mut tmp_attacks: Vec<Bitboard> = vec![Bitboard::default(); size];
            let mut i = 0;
            while i < size {
                let possible_moves = match piece {
                    Piece::Bishop => Bitboard::calc_diag_attacks(*sq, possible_blockers[i]),
                    Piece::Rook => Bitboard::calc_orth_attacks(*sq, possible_blockers[i]),
                    _ => panic!("Invalid piece type"),
                };
                let index = get_index(possible_blockers[i], mask, magic, shift);
                if tmp_attacks[index] == BB_EMPTY {
                    *attacks.add(start + index) = possible_moves;
                    tmp_attacks[index] = possible_moves;
                } else if tmp_attacks[index] != possible_moves {
                    break;
                }
                i += 1;
            }
            if i >= size {
                break 'table_fill;
            }
        }
        //magics.wrapping_add(*sq as usize).write(Magic::new(start, mask, magic, shift));
        tmp_magics[sq_idx].mask = mask;
        tmp_magics[sq_idx].magic = magic;
        tmp_magics[sq_idx].shift = shift;
        start += size;
    }
    let mut size = 0;
    //for i in 0..64 {
    tmp_magics
        .iter()
        .enumerate()
        .take(64)
        .for_each(|(i, magic)| {
            let beginptr = attacks.add(size);
            let magicptr: *mut Magic = magics.add(i);
            let currmagic: Magic = Magic {
                ptr: beginptr as usize,
                mask: magic.mask,
                magic: magic.magic,
                shift: magic.shift,
            };
            ptr::copy::<Magic>(&currmagic, magicptr, 1);
            size += magic.len;
        })
}
