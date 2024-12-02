from rchess import Board

def main():
    board = Board()
    board.push_uci('e2e4')
    print(board.to_string())
    print(board.legal_moves())

if __name__ == '__main__':
    main()