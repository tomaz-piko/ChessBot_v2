use crate::board::Board;

pub fn perft(
    board: &mut Board,
    depth: u32,
    divide: Option<bool>,
    uci: Option<bool>,
    verbose: bool,
) -> u64 {
    let divide: bool = divide.unwrap_or(false);
    let uci: bool = uci.unwrap_or(true);
    if verbose {
        println!("{}", board);
        println!("Running Perft until depth {}...", depth);
    }

    let nodes_searched: u64 = match divide {
        true => perft_divide(board, depth, uci),
        false => perft_bulk(board, depth),
    };
    if verbose {
        println!("Total nodes searched: {}", nodes_searched);
    }
    nodes_searched
}

pub fn perft_bulk(board: &mut Board, depth: u32) -> u64 {
    let mut nodes: u64 = 0;
    let moves = board.legal_moves();

    if depth == 1 {
        return moves.len() as u64;
    }

    for m in board.legal_moves() {
        let mut board2 = board.clone();
        if let Err(err) = board2.push(&m) {
            panic!("{}", err)
        };
        nodes += perft_bulk(&mut board2, depth - 1);
    }
    nodes
}

pub fn perft_bulk_uci(board: &mut Board, depth: u32) -> u64 {
    let mut nodes: u64 = 0;
    let moves = board.legal_moves();

    if depth == 1 {
        return moves.len() as u64;
    }

    for m in board.legal_moves() {
        let mut board2 = board.clone();
        if let Err(err) = board2.push_uci(&m.uci()) {
            panic!("{}", err)
        };
        nodes += crate::perft::perft_bulk(&mut board2, depth - 1);
    }
    nodes
}

pub fn perft_divide(board: &mut Board, depth: u32, uci: bool) -> u64 {
    let moves = board.legal_moves();
    let mut nodes = 0;
    for mv in moves.iter() {
        let mut board = board.clone();
        if let Err(err) = board.push(mv) {
            panic!("{}", err)
        };
        let count = perft_bulk(&mut board, depth - 1);
        if uci {
            println!("   {}: {},", mv.uci(), count);
        } else {
            println!("   {}: {},", mv, count);
        }
        nodes += count;
    }
    nodes
}
