from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Ellipse
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.clock import Clock
from best_move import get_best_move
from kivy.core.image import Image
import random
from board import CheckersBoard

BOARD_SIZE = 8

class CheckersBoard(CheckersBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cell_size = 0
        self.cells = {}
        self.manual_edit_mode = True
        self.edit_color = "red"
        self.player_color = None
        self.moves_made = 0
        self.crown_texture = Image("../images/crown.png").texture
        Clock.schedule_once(self.create_manual_edit_toolbar, 0.1)
        self.started = False

    def initialize_board(self):
        pass

    def on_size(self, *args):
        self.cell_size = min(self.width, self.height) / BOARD_SIZE
        self.repaint()

    def on_pos(self, *args):
        self.repaint()

    def on_touch_down(self, touch):
        if not self.started:
            col = int((touch.x - self.pos[0]) / self.cell_size)
            row = int((touch.y - self.pos[1]) / self.cell_size)

            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                return

            if self.manual_edit_mode:
                if self.edit_color == "erase":
                    self.cells.pop((row, col), None)
                elif (row, col) in self.cells:
                    self.cells[(row, col)] = (self.edit_color, not self.cells[(row, col)][1])
                else:
                    self.cells[(row, col)] = (self.edit_color, False)
                self.repaint()
            self.create_manual_edit_toolbar(0.1)
        else:
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

    def repaint(self):
        self.canvas.clear()
        with self.canvas:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    light = (row + col) % 2 == 0
                    Color(0.4, 0.2, 0, 1) if light else Color(1, 0.83, 0.5, 1)
                    Rectangle(
                        pos=(self.pos[0] + col * self.cell_size,
                             self.pos[1] + row * self.cell_size),
                        size=(self.cell_size, self.cell_size)
                    )

            for (row, col), (color, is_king) in self.cells.items():
                Color(1, 0, 0) if color == "red" else Color(0, 0, 0)
                Ellipse(
                    pos=(self.pos[0] + col * self.cell_size + 5,
                         self.pos[1] + row * self.cell_size + 5),
                    size=(self.cell_size - 10, self.cell_size - 10)
                )
                if is_king:
                    crown_size = self.cell_size / 2
                    Color(1, 0.84, 0, 1)
                    Rectangle(
                        texture=self.crown_texture,
                        pos=(self.pos[0] + col * self.cell_size + self.cell_size / 4, self.pos[1] + row * self.cell_size  + self.cell_size / 4),
                        size=(crown_size, crown_size)
                    )

    def set_edit_color(self, color):
        self.edit_color = color
        self.edit_popup.dismiss()

    def create_manual_edit_toolbar(self, dt):
        layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50)
        red_btn = Button(text="Place Red", background_color=(1, 0, 0, 1))
        black_btn = Button(text="Place Black", background_color=(0, 0, 0, 1))
        erase_btn = Button(text="Erase", background_color=(1, 1, 1, 1), color=(0, 0, 0, 1))
        done_btn = Button(text="Done", background_color=(0, 1, 0, 1))

        red_btn.bind(on_release=lambda *a: self.set_edit_color("red"))
        black_btn.bind(on_release=lambda *a: self.set_edit_color("black"))
        erase_btn.bind(on_release=lambda *a: self.set_edit_color("erase"))
        done_btn.bind(on_release=lambda *a: self.finish_manual_edit())

        layout.add_widget(red_btn)
        layout.add_widget(black_btn)
        layout.add_widget(erase_btn)
        layout.add_widget(done_btn)

        self.edit_popup = Popup(title="Setup Board", content=layout, size_hint=(0.9, None), height=200)
        self.edit_popup.open()

    def finish_manual_edit(self):
        self.manual_edit_mode = False
        self.edit_popup.dismiss()
        self.repaint()
        self.create_color_selection_popup()

    def create_color_selection_popup(self):
        layout = BoxLayout(orientation='horizontal')
        red_btn = Button(text="Play as Red", background_color=(1, 0, 0, 1))
        black_btn = Button(text="Play as Black", background_color=(0, 0, 0, 1), color=(1, 1, 1, 1))

        red_btn.bind(on_release=lambda *a: self.start_game("red"))
        black_btn.bind(on_release=lambda *a: self.start_game("black"))

        layout.add_widget(red_btn)
        layout.add_widget(black_btn)

        self.color_popup = Popup(title="Choose Your Color", content=layout, size_hint=(0.8, None), height=200)
        self.color_popup.open()

    def start_game(self, color):
        print(f"Starting game as {color}")
        self.player_color = "black" if color == "red" else "red"
        self.current_turn = color
        self.color_popup.dismiss()
        self.started = True
        Clock.schedule_once(lambda dt: self.computer_move(), 0.5)

    def computer_move(self, best_move_finder=None):
        print("Computer's turn")
        if best_move_finder is None:
            best_move_finder = get_best_move
        move = best_move_finder(self, self.current_turn, moves_made=0)

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


class CheckersApp(App):
    def build(self):
        return CheckersBoard()


if __name__ == "__main__":
    CheckersApp().run()
