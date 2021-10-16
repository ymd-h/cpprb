import unittest

from cpprb import LaBERmean, LaBERlazy, LaBERmax

class TestLaBER:
    def test_init(self):

        laber = self.cls(12)
        self.assertEqual(laber.batch_size, 12)
        self.assertEqual(laber.idx, [i for i in range(12*4)])
        self.assertEqual(laber.eps, 1e-6)

        with self.assertRaises(ValueError):
            self.cls(-12)

        laber = self.cls(12, 5)
        self.assertEqual(laber.batch_size, 12)
        self.assertEqual(laber.idx, [i for i in range(12*5)])
        self.assertEqual(laber.eps, 1e-6)

        with self.assertRaises(ValueError):
            self.cls(12, -4)

        laber = self.cls(12, 5, eps=1e-4)
        self.assertEqual(laber.batch_size, 12)
        self.assertEqual(laber.idx, [i for i in range(12*5)])
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
