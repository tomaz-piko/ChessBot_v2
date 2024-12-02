pub struct PRNG {
    seed: u64,
}

impl PRNG {
    #[inline(always)]
    pub fn init(s: u64) -> PRNG {
        PRNG { seed: s }
    }

    pub fn rand(&mut self) -> u64 {
        self.rand_change()
    }

    pub fn sparse_rand(&mut self) -> u64 {
        let mut s = self.rand_change();
        s &= self.rand_change();
        s &= self.rand_change();
        s
    }

    fn rand_change(&mut self) -> u64 {
        self.seed ^= self.seed >> 12;
        self.seed ^= self.seed << 25;
        self.seed ^= self.seed >> 27;
        self.seed.wrapping_mul(2685_8216_5773_6338_717)
    }
}