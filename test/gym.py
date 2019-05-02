import numpy as np
import unittest

from cpprb import gym


class TestGym(unittest.TestCase):

    def test_NoteBookAnimation(self):
        a = gym.NotebookAnimation()
        a.display()

    def test_Animation(self):
        with self.assertRaises(TypeError):
            gym.Animation()

if __name__ == '__main__':
    unittest.main()
