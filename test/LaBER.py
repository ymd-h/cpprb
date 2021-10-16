import unittest

from cpprb import LaBERmean, LaBERlazy, LaBERmax

class TestLaBER:
    def test_negative(self):
        with self.assertRaises(ValueError):
            self.cls.__class__(-12)

        with self.assertRaises(ValueError):
            self.cls.__class__(12, -4)

        with self.assertRaises(ValueError):
            self.cls.__class__(12, 4, -2)

    def test_call(self):
        batch_size = 32
        m = 4
        m_batch = batch_size * m
        laber = self.cls.__class__(batch_size, m)

        with self.assertRaises(ValueError):
            laber(priorities=[])

        self.assertEqual(laber(priorities=[1.0]*m_batch)["indexes"].shape,
                         (batch_size, ))

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
