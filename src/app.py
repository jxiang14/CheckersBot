from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.app import App
from kivy.graphics import Color, Rectangle
from kivy.uix.button import Button
from board import CheckersBoard

class TurnLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self.update_bg, size=self.update_bg)
        with self.canvas.before:
            self.bg_color = Color(1, 1, 1, 1)  # light gray
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

    def update_bg(self, *args):
        bg_height = self.height / 8
        bg_y = self.y + (self.height - bg_height) / 2
        self.bg_rect.pos = (self.x, bg_y)
        self.bg_rect.size = (self.width, bg_height)

class CheckersApp(App):
    def build(self):
        Window.clearcolor = (0.8, 0.8, 0.8, 1)
        layout = BoxLayout(orientation='horizontal', spacing=10, padding=10)

        # Board layout with AnchorLayout to center the board
        board_anchor = AnchorLayout(anchor_x='center', anchor_y='top')
        self.board = CheckersBoard(size_hint=(1, 1))  # Adjust size_hint as needed
        board_anchor.add_widget(self.board)

        self.turn_label = TurnLabel(text="Black's Turn", font_size=36, color=(0, 0, 0, 1))
        self.turn_label.size_hint = (0.2, 1)
        layout.add_widget(board_anchor)
        layout.add_widget(self.turn_label)

        self.board.turn_label = self.turn_label

        return layout
    
if __name__ == "__main__":
    CheckersApp().run()
