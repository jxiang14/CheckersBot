from kivy.core.image import Image
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color, Ellipse, Line, Triangle
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from best_move import get_best_move
import minimax
import time
import copy

# from qlearning import QLearningAgent, CheckersState, RED, BLACK

# agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.0)
# agent.load('checkers_qtable.pkl')  # path to your trained Q-table
# def get_best_move(board_widget, player_color, moves_made):
#     """
#     Translate the Kivy board (board_widget.cells) into a CheckersState,
#     ask the Q-agent for its action, and return the move tuple.
#     """
#     # Build a CheckersState from the current widget
#     # we rely on the board_from_kivy_board constructor path:
#     state = CheckersState(player_color, board=board_widget)
#     # Ask the agent for its preferred move
#     move = agent.choose_action(state)
#     if move is None:
#         return (None, None)
#     # move is ((r_from, c_from), (r_to, c_to))
#     return move

BOARD_SIZE = 8

class CheckersBoard(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crown_texture = Image("../images/crown.png").texture
        self.cell_size = min(Window.width, Window.height) / BOARD_SIZE
        self.bind(pos=self.update_board_layout, size=self.update_board_layout)

        self.cells = {}
        self.selected_piece = None
        self.selected_position = None
        self.highlight_rect = None
        self.current_turn = "black"
        self.continue_jump = False
        self.further_captures = []
        self.player_color = None
        self.computers_only = False
        self.moves_made = 0
        self.initialize_board()
        # self.create_color_selection_popup()
        Clock.schedule_once(lambda dt: self.create_color_selection_popup(), 0.1)
    
    def initialize_board(self):
        self.canvas.clear()
        with self.canvas:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    x, y = col * self.cell_size + self.pos[0], row * self.cell_size + self.pos[1]
                    if (row + col) % 2 == 0:
                        Color(0.4, 0.2, 0, 1)
                    else:
                        Color(1, 0.83, 0.5, 1)
                    Rectangle(pos=(x, y), size=(self.cell_size, self.cell_size))
                    
                    if (row + col) % 2 == 0:
                        if row < 3:
                            self.add_piece(row, col, "red")
                        elif row > 4:
                            self.add_piece(row, col, "black")

    def choose_color(self, color, popup):
        self.player_color = color
        popup.dismiss()
        if color == "red":
            self.computer_move()

    def create_color_selection_popup(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        label = Label(text="Choose your color:", font_size=20)
        red_button = Button(text="Play as Red", background_color=(1, 0, 0, 1), font_size=18)
        black_button = Button(text="Play as Black", background_color=(0, 0, 0, 1), font_size=18)

        popup = Popup(title="Player Color Selection", content=layout, size_hint=(0.6, 0.4), auto_dismiss=False)

        red_button.bind(on_release=lambda *args: self.choose_color("red", popup))
        black_button.bind(on_release=lambda *args: self.choose_color("black", popup))

        layout.add_widget(label)
        layout.add_widget(red_button)
        layout.add_widget(black_button)
        popup.open()

    def restart_game(self, color, popup):
        self.choose_color(color, popup)
        self.current_turn = "black"
        self.cells = {}
        self.selected_piece = None
        self.selected_position = None
        self.highlight_rect = None
        self.continue_jump = False
        self.further_captures = []
        self.initialize_board()
        self.repaint()
        popup.dismiss()

    def show_win_popup(self, winner):
        print("Moves made: ", self.moves_made)
        winner_count = 0
        loser_count = 0
        for (row, col) in self.cells.keys():
            color = self.cells[(row, col)][0]
            if color == winner:
                winner_count += 1
            else:
                loser_count += 1
        print("Winner # pieces: ", winner_count)
        print("Loser # pieces: ", loser_count)
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        label = Label(text=f"{winner.capitalize()} wins! Play again?", font_size=20)

        red_button = Button(text="Play as Red", background_color=(1, 0, 0, 1), font_size=18)
        black_button = Button(text="Play as Black", background_color=(0, 0, 0, 1), font_size=18)

        popup = Popup(
            title="Game Over",
            content=layout,
            size_hint=(0.6, 0.4),
            auto_dismiss=False,
            title_align='center'
        )

        red_button.bind(on_release=lambda *args: self.restart_game("red", popup))
        black_button.bind(on_release=lambda *args: self.restart_game("black", popup))

        layout.add_widget(label)
        layout.add_widget(red_button)
        layout.add_widget(black_button)
        popup.open()

    def update_board_layout(self, *args):
        board_size = min(self.width, self.height)
        self.cell_size = board_size / BOARD_SIZE

        self.board_offset_x = (self.width - board_size) / 2
        self.board_offset_y = (self.height - board_size) / 2

        self.repaint()
    
    def repaint(self):
        self.canvas.clear()
        with self.canvas:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    x, y = col * self.cell_size + self.pos[0], row * self.cell_size + self.pos[1]
                    if (row + col) % 2 == 0:
                        Color(0.4, 0.2, 0, 1)
                    else:
                        Color(1, 0.83, 0.5, 1)
                    Rectangle(pos=(x, y), size=(self.cell_size, self.cell_size))

            for key in self.cells.keys():
                row, col = key
                piece = self.cells[key]
                color, king = piece
                self.add_piece(row, col, color, king)
    
    def add_piece(self, row, col, color, king=False):
        x, y = col * self.cell_size + self.pos[0], row * self.cell_size + self.pos[1]
        self.cells[(row, col)] = (color, king)
        with self.canvas:
            Color(0.70, 0.18, 0, 1) if color == "red" else Color(0, 0, 0, 1)
            Ellipse(pos=(x + 10, y + 10), size=(self.cell_size - 20, self.cell_size - 20))
            if king:
                crown_size = self.cell_size / 2
                Color(1, 0.84, 0, 1)
                Rectangle(
                    texture=self.crown_texture,
                    pos=(x + self.cell_size / 4, y + self.cell_size / 4),
                    size=(crown_size, crown_size)
                )

    def set_player_color(self, color):
        self.player_color = color
    
    def on_touch_down(self, touch):
        if not self.player_color:
            return
        if self.computers_only:
            self.run_computers_against_each_other()
            return
        col = int((touch.x - self.pos[0]) / self.cell_size)
        row = int((touch.y - self.pos[1]) / self.cell_size)
        if self.continue_jump:
            if (row, col) in self.further_captures:
                self.move_piece(row, col)
            return
        if (row, col) in self.cells:
            piece_color, _ = self.cells[(row, col)]
            if piece_color == self.current_turn and piece_color == self.player_color:
                self.selected_piece = self.cells[(row, col)]
                self.selected_position = (row, col)
                self.remove_highlight()
                self.highlight_selected(row, col)
        elif self.selected_piece:
            valid_moves = self.get_all_valid_moves(self.current_turn)
            if (self.selected_position,(row, col)) in valid_moves:
                self.move_piece(row, col)

    def highlight_selected(self, row, col):
        self.remove_highlight()
        x, y = col * self.cell_size + self.pos[0], row * self.cell_size + self.pos[1]
        with self.canvas:
            Color(1, 1, 0, 1)
            self.highlight_rect = Line(rectangle=(x, y, self.cell_size, self.cell_size), width=2)

    def remove_highlight(self):
        if self.highlight_rect:
            self.canvas.remove(self.highlight_rect)
            self.highlight_rect = None
    
    def move_piece(self, row, col):
        self.moves_made += 1
        old_row, old_col = self.selected_position
        just_made_king = False
        if (row, col) not in self.cells:
            color, is_king = self.selected_piece

            if not is_king and (color == "red" and row == BOARD_SIZE - 1) or (color == "black" and row == 0):
                is_king = True
                just_made_king = True

            self.cells[(row, col)] = (color, is_king)
            self.selected_piece = (color, is_king)
            del self.cells[old_row, old_col]
        
        was_capture = abs(row - old_row) == 2
        if was_capture:
            captured_row = (row + old_row) // 2
            captured_col = (col + old_col) // 2
            if (captured_row, captured_col) in self.cells:
                del self.cells[(captured_row, captured_col)]

        further_moves = self.get_valid_moves(row, col)
        self.further_captures = []
        for (_, (r,c)) in further_moves:
            if abs(r - row) == 2:
                self.further_captures.append((r, c))

        if was_capture and self.further_captures and not just_made_king:
            self.continue_jump = True
            self.remove_highlight()
            self.selected_position = (row, col)
            self.selected_piece = self.cells[(row, col)]
            self.repaint()
            self.highlight_selected(row, col)
            return
        else:
            self.continue_jump = False
        
        self.switch_player()
        
        if self.check_win_condition():
            return

        if self.current_turn != self.player_color and not self.computers_only:
            Clock.schedule_once(lambda dt: self.computer_move(), 0)

    def switch_player(self):
        self.selected_piece = None
        self.selected_position = None
        self.current_turn = "black" if self.current_turn == "red" else "red"
        self.remove_highlight()
        self.repaint()

        if hasattr(self, 'turn_label') and self.turn_label:
            self.turn_label.text = f"{self.current_turn.capitalize()}'s Turn"
            self.turn_label.color = (1, 0, 0, 1) if self.current_turn == "red" else (0, 0, 0, 1)

    def run_computers_against_each_other(self):
        def next_move(dt):
            if self.check_win_condition():
                return  # Game over, stop scheduling moves

            if self.current_turn == "red":
                print("Red's turn")
                self.computer_move(get_best_move)
                Clock.schedule_once(next_move, 1.5)
            else:
                print("Black's turn")
                self.computer_move(minimax.get_best_move)
                Clock.schedule_once(next_move, 1.5)
            print("Moves made: ", self.moves_made)

        Clock.schedule_once(next_move, 0)

    def computer_move(self, best_move_finder=None):
        if best_move_finder is None:
            best_move_finder = get_best_move
        move = best_move_finder(self, self.current_turn, moves_made=self.moves_made)

        if move:
            def do_step(i):
                if i < len(move) - 1:
                    from_pos = move[i]
                    to_pos = move[i + 1]
                    self.selected_position = from_pos
                    self.selected_piece = self.cells[from_pos]
                    self.move_piece(to_pos[0], to_pos[1])
                    Clock.schedule_once(lambda dt: do_step(i + 1), 0.1)
            do_step(0)

    def get_valid_moves(self, row, col):
        valid_moves = []
        color, king = self.cells.get((row, col))
        
        if color == "red":
            directions = [(1, -1), (1, 1)]
        else:
            directions = [(-1, -1), (-1, 1)]

        if king:
            if color == "red":
                directions += [(-1, -1), (-1, 1)]
            elif color == "black":
                directions += [(1, -1), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE and (new_row, new_col) not in self.cells:
                valid_moves.append(((row, col),(new_row, new_col)))
            elif 0 <= new_row + dr < BOARD_SIZE and 0 <= new_col + dc < BOARD_SIZE:
                middle_row, middle_col = row + dr, col + dc
                if (middle_row, middle_col) in self.cells and self.cells[(middle_row, middle_col)][0] != color:
                    landing_row, landing_col = new_row + dr, new_col + dc
                    if 0 <= landing_row < BOARD_SIZE and 0 <= landing_col < BOARD_SIZE and (landing_row, landing_col) not in self.cells:
                        valid_moves.append(((row, col),(landing_row, landing_col)))
        return valid_moves
    
    def get_all_valid_moves(self, player):
        valid_moves = []
        for piece in self.cells.keys():
            row, col = piece
            color, king = self.cells[piece]
            if color == player:
                valid_moves.extend(self.get_valid_moves(row, col))
        capture_moves = []
        for move in valid_moves:
            r, _, new_r, _ = move[0][0], move[0][1], move[1][0], move[1][1]
            if abs(new_r - r) > 1:
                capture_moves.append(move)
        if len(capture_moves) > 0:
            return capture_moves
        return valid_moves
    
    def check_win_condition(self):
        opponent_color = "black" if self.current_turn == "red" else "red"
        has_pieces = False
        has_moves = False
        for (row, col) in self.cells.keys():
            color = self.cells[(row, col)][0]
            if color == opponent_color:
                has_pieces = True
            if color == self.current_turn:
                if len(self.get_valid_moves(row, col)) > 0:
                    has_moves = True

        if not has_pieces:
            self.show_win_popup(winner=self.current_turn)
            return True
        if not has_moves:
            self.show_win_popup(winner=opponent_color)
            return True
        return False
