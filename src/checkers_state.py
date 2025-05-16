import copy

BOARD_SIZE = 8
RED = 1
BLACK = -1

class CheckersState:
    def __init__(self, player, board=None):
        self.red_pieces = []
        self.black_pieces = []
        self.must_continue_from = None
        self.current_player = self.player_from_kivy_player(player)
        if board is not None:
            self.board = self.board_from_kivy_board(board)
        else:
            self.board = self.create_board()

    def player_from_kivy_player(self, player):
        """
        Convert the player color from Kivy representation to internal representation.
        """
        if player == "red":
            return RED
        elif player == "black":
            return BLACK
        else:
            raise ValueError("Invalid player color")

    def board_from_kivy_board(self, board):
        """
        Convert the Kivy board representation to the internal representation.
        """
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
        """
        Create the initial board with pieces in their starting positions.
        """
        board = [[0] * 8 for _ in range(8)]
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 0:
                    board[row][col] = (RED, False)
                    self.red_pieces.append((row, col))
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    board[row][col] = (BLACK, False)
                    self.black_pieces.append((row, col))
        return board
    
    def get_directions(self, color, king):
        """
        Get the possible directions for a piece based on its color and whether it's a king.
        """
        if king:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(1, -1), (1, 1)] if color == RED else [(-1, -1), (-1, 1)]
    
    def explore_jumps(self, r, c, visited, color, king):
        """
        Explore all possible jumps from the current position (r, c) for a piece of the given color.
        """
        jumps = []
        for dr, dc in self.get_directions(color, king):
            mid_r, mid_c = r + dr, c + dc
            end_r, end_c = r + 2*dr, c + 2*dc
            if (
                self.in_bounds(end_r, end_c)
                and self.board[mid_r][mid_c] != 0
                and self.board[mid_r][mid_c][0] != color
                and self.board[end_r][end_c] == 0
                and (mid_r, mid_c, end_r, end_c) not in visited
            ):
                next_path = [(r, c), (end_r, end_c)]
                if king:
                    visited.add((mid_r, mid_c, r, c))
                visited.add((mid_r, mid_c, end_r, end_c))
                further_jumps = self.explore_jumps(end_r, end_c, visited, color, king)
                if further_jumps:
                    for fj in further_jumps:
                        jumps.append([(r, c)] + fj)  # chain jumps
                else:
                    jumps.append(next_path)
                visited.remove((mid_r, mid_c, end_r, end_c))
                if king:
                    visited.remove((mid_r, mid_c, r, c))
        return jumps
    
    def in_bounds(self, r, c):
        """
        Check if the given row and column are within the bounds of the board.
        """
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
    
    def get_valid_moves(self, row, col):
        """
        Get all valid moves for a piece at the given row and column.
        """
        valid_moves = []
        color, king = self.board[row][col]

        jumps = self.explore_jumps(row, col, set(), color, king)
        if jumps:
            return jumps
        
        directions = self.get_directions(color, king)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.in_bounds(new_row, new_col) and self.board[new_row][new_col] == 0:
                valid_moves.append([(row, col),(new_row, new_col)])

        return valid_moves

    def get_all_valid_moves(self, player):
        """
        Get all valid moves for the current player.
        If there are capturing moves available, only return those.
        """
        valid_moves = []
        pieces = self.red_pieces if player == RED else self.black_pieces
        for piece in pieces:
            row, col = piece
            valid_moves.extend(self.get_valid_moves(row, col))
        capture_moves = []
        for move in valid_moves:
            r, _, new_r, _ = move[0][0], move[0][1], move[1][0], move[1][1]
            if abs(new_r - r) > 1:
                capture_moves.append(move)
        if len(capture_moves) > 0:
            return capture_moves
        return valid_moves

    def make_move(self, move):
        """
        Make a move on the board and update the state.
        """
        for i in range(len(move) - 1):
            piece, new_position = move[i], move[i + 1]
            row, col = piece
            new_row, new_col = new_position
            color, king = self.board[row][col]
            if self.current_player == RED:
                self.red_pieces.remove(piece)
                self.red_pieces.append(new_position)
            else:
                self.black_pieces.remove(piece)
                self.black_pieces.append(new_position)
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
        final_row, _ = move[-1]
        if (final_row == BOARD_SIZE - 1 and self.current_player == RED) or (final_row == 0 and self.current_player == BLACK):
            self.board[new_row][new_col] = (self.current_player, True)
        self.current_player = BLACK if self.current_player == RED else RED
        return self

    def get_winner(self):
        """
        Check if the game is over and return the winner.
        """
        if self.current_player == RED:
            has_pieces = len(self.red_pieces) > 0
            has_moves = len(self.get_all_valid_moves(RED)) > 0
        else:
            has_pieces = len(self.black_pieces) > 0
            has_moves = len(self.get_all_valid_moves(BLACK)) > 0

        if not has_pieces or not has_moves:
            return RED if self.current_player == BLACK else BLACK
        return 0
    
    def is_terminal(self):
        """
        Check if the game is over.
        """
        return self.get_winner() != 0
    
    def clone(self):
        """
        Create a deep copy of the current state.
        """
        clone_state = object.__new__(CheckersState)
        clone_state.board = copy.deepcopy(self.board)
        clone_state.red_pieces = copy.deepcopy(self.red_pieces)
        clone_state.black_pieces = copy.deepcopy(self.black_pieces)
        clone_state.current_player = copy.deepcopy(self.current_player)
        clone_state.must_continue_from = copy.deepcopy(self.must_continue_from)
        return clone_state
    
    def can_capture_single_piece_if_moved(self, move):
        """
        Check if a single piece can be captured if moved.
        """
        (r1, c1), (r2, c2) = move
        return abs(r2 - r1) == 2
    
    def can_be_captured_if_moved(self, move):
        """
        Check if the piece can be captured if moved.
        """
        (r1, c1), (r2, c2) = move[-2], move[-1]
        if self.current_player == RED:
            opponent_pieces = self.black_pieces
            dirs = [(-1, -1), (-1, 1)]
        else:
            opponent_pieces = self.red_pieces
            dirs = [(1, -1), (1, 1)]
        for dr, dc in dirs:
            r, c = r2 - dr, c2 - dc
            if (r, c) in opponent_pieces:
                if r2 + dr >= 0 and r2 + dr < 8 and c2 + dc >= 0 and c2 + dc < 8:
                    if c2 + dc == c1 or self.board[r2 + dr][c2 + dc] == 0:
                        return True
        return False
    
    def can_become_king(self, move):
        """
        Check if the piece can become a king.
        """
        r2, c2 = move[-1]
        if self.current_player == RED and r2 == 7:
            return True
        elif self.current_player == BLACK and r2 == 0:
            return True
        return False
    
    def can_be_at_edge(self, move):
        """
        Check if the piece can be at the edge of the board.
        """
        r2, c2 = move[-1]
        if r2 == 0 or r2 == 7 or c2 == 0 or c2 == 7:
            return True
        return False
    
