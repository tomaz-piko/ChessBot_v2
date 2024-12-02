use rchess::board::Board;
use rchess::perft::perft_bulk;

// Perft Node count testing for up to depth 5
// https://www.chessprogramming.org/Perft_Results
#[cfg(test)]
mod chess_programming_perft_tests {
    use rchess::perft::perft_bulk_uci;
    use super::*;

    #[test]
    fn perft_initial_board() {
        let mut board = Board::new(None);
        assert_eq!(perft_bulk(&mut board, 1), 20);
        assert_eq!(perft_bulk(&mut board, 2), 400);
        assert_eq!(perft_bulk(&mut board, 3), 8902);
        assert_eq!(perft_bulk(&mut board, 4), 197281);
        assert_eq!(perft_bulk(&mut board, 5), 4865609);
    }

    #[test]
    fn perft_kiwipete() {
        let mut board = Board::new(Some(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",
        ));
        assert_eq!(perft_bulk(&mut board, 1), 48);
        assert_eq!(perft_bulk(&mut board, 2), 2039);
        assert_eq!(perft_bulk(&mut board, 3), 97862);
        assert_eq!(perft_bulk(&mut board, 4), 4085603);
        assert_eq!(perft_bulk(&mut board, 5), 193690690);
    }

    #[test]
    fn perft_position_3() {
        let mut board = Board::new(Some("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"));
        assert_eq!(perft_bulk(&mut board, 1), 14);
        assert_eq!(perft_bulk(&mut board, 2), 191);
        assert_eq!(perft_bulk(&mut board, 3), 2812);
        assert_eq!(perft_bulk(&mut board, 4), 43238);
        assert_eq!(perft_bulk(&mut board, 5), 674624);
    }

    #[test]
    fn perft_position_4() {
        let mut board = Board::new(Some(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ));
        assert_eq!(perft_bulk(&mut board, 1), 6);
        assert_eq!(perft_bulk(&mut board, 2), 264);
        assert_eq!(perft_bulk(&mut board, 3), 9467);
        assert_eq!(perft_bulk(&mut board, 4), 422333);
        assert_eq!(perft_bulk(&mut board, 5), 15833292);
    }

    #[test]
    fn perft_position_4_mirrrored() {
        let mut board = Board::new(Some(
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1 ",
        ));
        assert_eq!(perft_bulk(&mut board, 1), 6);
        assert_eq!(perft_bulk(&mut board, 2), 264);
        assert_eq!(perft_bulk(&mut board, 3), 9467);
        assert_eq!(perft_bulk(&mut board, 4), 422333);
        assert_eq!(perft_bulk(&mut board, 5), 15833292);
    }

    #[test]
    fn perft_position_5() {
        let mut board = Board::new(Some(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ",
        ));
        assert_eq!(perft_bulk(&mut board, 1), 44);
        assert_eq!(perft_bulk(&mut board, 2), 1486);
        assert_eq!(perft_bulk(&mut board, 3), 62379);
        assert_eq!(perft_bulk(&mut board, 4), 2103487);
        assert_eq!(perft_bulk(&mut board, 5), 89941194);
    }

    #[test]
    fn perft_position_6() {
        let mut board = Board::new(Some(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",
        ));
        assert_eq!(perft_bulk(&mut board, 1), 46);
        assert_eq!(perft_bulk(&mut board, 2), 2079);
        assert_eq!(perft_bulk(&mut board, 3), 89890);
        assert_eq!(perft_bulk(&mut board, 4), 3894594);
        assert_eq!(perft_bulk(&mut board, 5), 164075551);
    }

    #[test]
    fn perft_kiwipete_uci() {
        let mut board = Board::new(Some(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",
        ));
        assert_eq!(perft_bulk_uci(&mut board, 1), 48);
        assert_eq!(perft_bulk_uci(&mut board, 2), 2039);
        assert_eq!(perft_bulk_uci(&mut board, 3), 97862);
        assert_eq!(perft_bulk_uci(&mut board, 4), 4085603);
        assert_eq!(perft_bulk_uci(&mut board, 5), 193690690);
    }
}
