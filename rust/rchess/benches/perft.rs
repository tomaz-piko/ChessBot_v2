use criterion::{black_box, criterion_group, Bencher, Criterion};
use rchess::board::Board;
use rchess::perft::perft;

fn benchmark(c: &mut Criterion) {
    c.bench_function("peft_initial_position", peft_initial_position);
    c.bench_function("peft_kiwipete_position", peft_kiwipete_position);
}

fn peft_initial_position(b: &mut Bencher) {
    let mut board = Board::new(None);
    b.iter(|| {
        black_box(perft(&mut board, 5, None, None));
    });
}

fn peft_kiwipete_position(b: &mut Bencher) {
    let mut board = Board::new(Some(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
    ));
    b.iter(|| {
        black_box(perft(&mut board, 4, None, None));
    });
}

criterion_group!(benches, benchmark);
