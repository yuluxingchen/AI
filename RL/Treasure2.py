import numpy as np
import time
import sys
import Tkinter as tk


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.unit = 40
        self.maze_h = 5
        self.maze_w = 5
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('寻宝')
        self.geometry('{0}x{1}'.format(self.maze_h * self.unit, self.maze_w * self.unit))
        self.build_maze()

    def build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=self.maze_h * self.unit, width=self.maze_w * self.unit)
        for c in range(0, self.maze_w * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.maze_h * self.unit