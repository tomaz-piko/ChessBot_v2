use criterion::criterion_main;

mod perft;

criterion_main!(perft::benches,);
