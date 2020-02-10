import platform
import unittest

import numpy as np

from cpprb import gym


class TestGym(unittest.TestCase):
    @unittest.skipIf(platform.system() == 'Windows',
                     'Unsupport Platform')
    def test_NoteBookAnimation(self):
        a = gym.NotebookAnimation()
        a.display()

    @unittest.skipIf(platform.system() == 'Windows',
                     'Unsupport Platform')
    def test_Animation(self):
        with self.assertRaises(TypeError):
            gym.Animation()

if __name__ == '__main__':
    unittest.main()
