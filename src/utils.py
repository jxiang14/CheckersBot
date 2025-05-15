import copy

BOARD_SIZE = 8
RED = 1
BLACK = -1

class CheckersState:
    def __init__(self, player, board=None):
        self.red_pieces = []
        self.black_pieces = []
        self.current_player = self.player_from_kivy_player(player)    # 1 for RED, 2 for BLACK
        if board is not None:
            self.board = self.board_from_kivy_board(board)
        else:
            self.board = self.create_board()
        self.must_continue_from = None

    def player_from_kivy_player(self, player):
        if player == "red":
            return RED
        elif player == "black":
            return BLACK
        else:
            raise ValueError("Invalid player color")

    def board_from_kivy_board(self, board):
        new_board = [[0] * 8 for _ in range(8)]
        for (row, col) in board.cells.keys():
            color, king = board.cells[(row, col)]
            if color == "red":
                new_board[row][col] = (RED, king)
                self.red_pieces.append((row, col))
            elif color == "black":
                new_board[row][col] = (BLACK, king)
                self.black_pieces.append((row, col))
        return new_board

    def create_board(self):
        board = [[0] * 8 for _ in range(8)]
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = (RED, False)
                    self.red_pieces.append((row, col))
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = (BLACK, False)
                    self.black_pieces.append((row, col))
        return board
    
    def get_valid_moves(self, row, col):
        valid_moves = []
        color, king = self.board[row][col]
        
        if color == RED:
            directions = [(1, -1), (1, 1)]
        else:
            directions = [(-1, -1), (-1, 1)]

        if king:
            if color == RED:
                directions += [(-1, -1), (-1, 1)]
            elif color == BLACK:
                directions += [(1, -1), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE and self.board[new_row][new_col] == 0:
                valid_moves.append(((row, col),(new_row, new_col)))
            elif 0 <= new_row + dr < BOARD_SIZE and 0 <= new_col + dc < BOARD_SIZE:
                middle_row, middle_col = row + dr, col + dc
                if self.board[middle_row][middle_col] != 0 and self.board[middle_row][middle_col][0] != color:
                    landing_row, landing_col = new_row + dr, new_col + dc
                    if 0 <= landing_row < BOARD_SIZE and 0 <= landing_col < BOARD_SIZE and self.board[landing_row][landing_col] == 0:
                        valid_moves.append(((row, col), (landing_row, landing_col)))
        
        return valid_moves

    # def get_all_valid_moves(self, player):
    #     valid_moves = []
    #     capturing_moves = []

    #     pieces = self.red_pieces if player == RED else self.black_pieces
    #     if self.must_continue_from:
    #         moves = self.get_valid_moves(self.must_continue_from[0], self.must_continue_from[1])
    #         return [m for m in moves if abs(m[1][0] - m[0][0]) > 1]
    #     for piece in pieces:
    #         moves = self.get_valid_moves(*piece)
    #         for move in moves:
    #             if abs(move[1][0] - move[0][0]) > 1:  # capture move
    #                 capturing_moves.append(move)
    #         valid_moves.extend(moves)

    #     return capturing_moves if capturing_moves else valid_moves
    
    def get_all_valid_moves(self, player):
        valid_moves = []
        pieces = self.red_pieces if player == RED else self.black_pieces
        if self.must_continue_from:
            # Only return capturing moves from the same piece
            moves = self.get_valid_moves(*self.must_continue_from)
            return [m for m in moves if abs(m[1][0] - m[0][0]) > 1]
        for piece in pieces:
            row, col = piece
            valid_moves.extend(self.get_valid_moves(row, col))
        return valid_moves

    def make_move(self, move):
        piece, new_position = move
        row, col = piece
        new_row, new_col = new_position
        if self.current_player == RED:
            self.red_pieces.remove(piece)
            self.red_pieces.append(new_position)
        else:
            self.black_pieces.remove(piece)
            self.black_pieces.append(new_position)
        color, king = self.board[row][col]
        self.board[row][col] = 0
        self.board[new_row][new_col] = color, king

        if abs(new_row - row) > 1:
            middle_row = (row + new_row) // 2
            middle_col = (col + new_col) // 2
            if self.current_player == RED:
                self.black_pieces.remove((middle_row, middle_col))
            else:
                self.red_pieces.remove((middle_row, middle_col))
            self.board[middle_row][middle_col] = 0

        if (new_row == 0 and self.current_player == RED) or (new_row == BOARD_SIZE - 1 and self.current_player == BLACK):
            self.board[new_row][new_col] = (self.current_player, True)

        # if abs(new_row - row) > 1:
        #     self.check_further_captures(new_row, new_col)
        if abs(new_row - row) > 1:
            # check if piece must continue capturing
            further_captures = self.get_valid_moves(new_row, new_col)
            self.must_continue_from = (new_row, new_col) if any(
                abs(m[1][0] - m[0][0]) > 1 for m in further_captures
            ) else None
        return self
    
    def check_further_captures(self, row, col):
        moves = self.get_valid_moves(row, col)
        if moves:
            for move in moves:
                if move[0] == (row, col):
                    self.make_move(move)
                    break

    def switch_player(self):
        self.must_continue_from = None
        self.current_player = BLACK if self.current_player == RED else RED

    def get_winner(self):
        has_pieces = False
        has_moves = False
        if self.current_player == RED:
            has_pieces = len(self.red_pieces) > 0
            has_moves = len(self.get_all_valid_moves(RED)) > 0
        else:
            has_pieces = len(self.black_pieces) > 0
            has_moves = len(self.get_all_valid_moves(BLACK)) > 0

        if not has_pieces or not has_moves:
            # print(f"Loser: {self.current_player}")
            return RED if self.current_player == BLACK else BLACK
        return 0
    
    def is_terminal(self):
        return self.get_winner() != 0
    
    def clone(self):
        clone_state = object.__new__(CheckersState)
        clone_state.board = copy.deepcopy(self.board)
        clone_state.red_pieces = self.red_pieces[:]
        clone_state.black_pieces = self.black_pieces[:]
        clone_state.current_player = self.current_player
        clone_state.must_continue_from = self.must_continue_from
        return clone_state