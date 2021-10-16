import unittest

from cpprb import LaBERmean, LaBERlazy, LaBERmax

class TestLaBER:
    def test_negative(self):
        with self.assertRaises(ValueError):
            self.cls(-12)

        with self.assertRaises(ValueError):
            self.cls(12, -4)

        with self.assertRaises(ValueError):
            self.cls(12, 4, -2)

class TestLaBERmean(TestLaBER, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = LaBERmean

class TestLaBERlazy(TestLaBER, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = LaBERlazy


class TestLaBERmax(TestLaBER, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = LaBERmax


if __name__ == "__main__":
    unittest.main()
