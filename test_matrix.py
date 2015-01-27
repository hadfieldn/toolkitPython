from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
from matrix import Matrix


class TestMatrix(TestCase):

    infinity = float("infinity")

    def setUp(self):

        # NOTE: for discrete attributes, at least one value must be a float in order for numpy array
        # functions to work properly.
        m = Matrix()
        m.attr_names = ['A', 'B', 'C']
        m.str_to_enum = [{}, {}, {'R': 0, 'G': 1, 'B': 2}]
        m.enum_to_str = [{}, {}, {0: 'R', 1: 'G', 2: 'B'}]
        m.data = [[1.5, -6, 1.0],
                  [2.3, -8, 2],
                  [4.1, self.infinity, 2]]
        self.m = m

        m2 = Matrix()
        m2.attr_names = ['A', 'B', 'C', 'D', 'E']
        m2.str_to_enum = [{}, {}, {}, {}, {'R': 0, 'G': 1, 'B': 2}]
        m2.enum_to_str = [{}, {}, {}, {}, {0: 'R', 1: 'G', 2: 'B'}]
        m2.data = [[0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.1, 1.1, 2.1, 3.1, 1.0],
                   [0.2, 1.2, 2.2, 3.2, 1.0],
                   [0.3, 1.3, 2.3, 3.3, 2.0],
                   [0.4, 1.4, 2.4, 3.4, 2.0]]
        self.m2 = m2

    def test_init_from(self):
        m2 = Matrix(self.m, 1, 1, 2, 2)
        self.assertListEqual(m2.row(0), [-8, 2])
        self.assertListEqual(m2.row(1), [self.infinity, 2])

    def test_add(self):
        self.m.add(self.m2, 0, 2, 3)
        self.m.print()
        self.assertListEqual(self.m.row(3), self.m2.row(0)[2:])
        self.m.add(self.m2, 3, 2, 3)
        self.m.print()
        self.assertListEqual(self.m.row(9), self.m2.row(4)[2:])

    def test_set_size(self):
        m = Matrix()
        m.set_size(3, 4)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 4)

    def test_load_arff(self):
        t = Matrix()
        t.load_arff("test/cm1_req.arff")
        self.assertListEqual(t.row(t.rows-1), [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 3.0, 1.0, 1.0])

    def test_rows(self):
        self.assertEquals(self.m.rows, 3)

    def test_cols(self):
        self.assertEquals(self.m.cols, 3)

    def test_row(self):
        self.assertListEqual(self.m.row(1), [2.3, -8, 2])

    def test_col(self):
        self.assertListEqual(self.m.col(1), [-6, -8, self.infinity])

    def test_get(self):
        self.assertEquals(self.m.get(0, 2), 1)
        self.assertEquals(self.m.get(2, 0), 4.1)

    def test_set(self):
        self.m.set(2, 1, 2.5)
        self.assertEquals(self.m.get(2, 1), 2.5)

    def test_attr_name(self):
        name = self.m.attr_name(2)
        self.assertEquals(name, 'C')

    def test_set_attr_name(self):
        self.m.set_attr_name(2, 'Color')
        self.assertEquals(self.m.attr_name(2), 'Color')

    def test_attr_value(self):
        self.assertEquals(self.m.attr_value(2, 0), 'R')

    def test_value_count(self):
        self.assertEquals(self.m.value_count(1), 0)     # continuous
        self.assertEquals(self.m.value_count(2), 3)     # R, G, B

    def test_shuffle(self):
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        pass

    def test_column_mean(self):
        self.assertAlmostEquals(self.m.column_mean(0), 2.6333, 4)
        self.assertAlmostEquals(self.m.column_mean(1), -7, 4)

    def test_column_min(self):
        self.assertEquals(self.m.column_min(0), 1.5)
        self.assertEquals(self.m.column_min(1), -8)

    def test_column_max(self):
        self.assertEquals(self.m.column_max(0), 4.1)
        self.assertEquals(self.m.column_max(1), -6)

    def test_most_common_value(self):
        self.assertEquals(self.m.most_common_value(0), 1.5)
        self.assertEquals(self.m.most_common_value(2), 2)

# suite = TestLoader().loadTestsFromTestCase(TestMatrix)
# TextTestRunner(verbosity=2).run(suite)