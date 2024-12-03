pub struct Prng {
    seed: u64,
}

impl Prng {
    #[inline(always)]
    pub fn init(s: u64) -> Prng {
        Prng { seed: s }
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
        self.seed.wrapping_mul(2_685_821_657_736_338_717)
    }
}
