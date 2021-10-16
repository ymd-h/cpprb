import unittest
import numpy as np

from cpprb import LaBERmean, LaBERlazy, LaBERmax

class TestLaBER:
    def test_init(self):

        laber = self.cls(12)
        self.assertEqual(laber.batch_size, 12)
        np.testing.assert_array_equal(laber.idx, [i for i in range(12*4)])
        self.assertEqual(laber.eps, 1e-6)

        with self.assertRaises(ValueError):
            self.cls(-12)

        laber = self.cls(12, 5)
        self.assertEqual(laber.batch_size, 12)
        np.testing.assert_array_equal(laber.idx, [i for i in range(12*5)])
        self.assertEqual(laber.eps, 1e-6)

        with self.assertRaises(ValueError):
            self.cls(12, -4)

        laber = self.cls(12, 5, eps=1e-4)
        self.assertEqual(laber.batch_size, 12)
        np.testing.assert_array_equal(laber.idx, [i for i in range(12*5)])
        self.assertEqual(laber.eps, 1e-4)

        with self.assertRaises(ValueError):
            self.cls(12, 4, eps=-2)

    def test_call(self):
        batch_size = 32
        m = 4
        m_batch = batch_size * m

        laber = self.cls(batch_size, m)

        with self.assertRaises(ValueError):
            laber(priorities=[])

        sample = laber(priorities=[1.0]*m_batch)
        self.assertEqual(sample["indexes"].shape, (batch_size, ))
        self.assertEqual(sample["weights"].shape, (batch_size, ))

    def test_uniform(self):
        laber = self.cls(2, 2)

        sample = laber(priorities=[1,1,1,1])
        np.testing.assert_array_equal(sample["weights"], self.uniform)

    def test_onehot(self):
        laber = self.cls(2, 2)
        laber.eps = 0 # Hack for test

        sample = laber(priorities=[1, 0, 0, 0])
        np.testing.assert_array_equal(sample["indexes"], [0, 0])
        np.testing.assert_array_equal(sample["weights"], self.onehot)


class TestLaBERmean(TestLaBER, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = LaBERmean
        cls.uniform = (1, 1)
        cls.onehot = (1, 1)


class TestLaBERlazy(TestLaBER, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = LaBERlazy
        cls.uniform = (4, 4)
        cls.onehot = (1, 1)


class TestLaBERmax(TestLaBER, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = LaBERmax
        cls.uniform = (1, 1)
        cls.onehot = (1, 1)


if __name__ == "__main__":
    unittest.main()
