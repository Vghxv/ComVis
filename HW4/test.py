import unittest
import numpy as np

class PixelGroup:
    def __init__(self):
        self.pixels: dict = {}
        self.means: dict = {}
        self.variances: dict = {}

    def add_pixel(self, pixel, label):
        if label not in self.pixels:
            self.pixels[label] = [np.array(pixel)]
        else:
            self.pixels[label].append(np.array(pixel))

    def update_mean(self):
        for label in self.pixels:
            self.means[label] = np.mean(self.pixels[label], axis=0)

    def update_variance(self):
        for label in self.pixels:
            self.variances[label] = np.var(self.pixels[label], axis=0)

    def update_mean(self, label):
        self.means[label] = np.mean(self.pixels[label], axis=0)

    def update_variance(self, label):
        self.variances[label] = np.var(self.pixels[label], axis=0)

    def get_mean(self, label):
        return self.means[label]

    def get_variance(self, label):
        return self.variances[label]


class TestPixelGroup(unittest.TestCase):
    def setUp(self):
        self.pixel_group = PixelGroup()

    def test_add_pixel(self):
        self.pixel_group.add_pixel([1, 2, 3], 'red')
        self.assertTrue('red' in self.pixel_group.pixels)
        np.testing.assert_array_equal(self.pixel_group.pixels['red'][0], np.array([1, 2, 3]))

    def test_update_mean(self):
        self.pixel_group.add_pixel([1, 2, 3], 'red')
        self.pixel_group.add_pixel([4, 5, 6], 'red')
        self.pixel_group.update_mean('red')
        np.testing.assert_array_equal(self.pixel_group.means['red'], np.array([2.5, 3.5, 4.5]))

    def test_update_variance(self):
        self.pixel_group.add_pixel([1, 2, 3], 'red')
        self.pixel_group.add_pixel([4, 5, 6], 'red')
        self.pixel_group.update_variance('red')
        np.testing.assert_array_equal(self.pixel_group.variances['red'], np.array([2.25, 2.25, 2.25]))

    def test_get_mean(self):
        self.pixel_group.add_pixel([1, 2, 3], 'red')
        self.pixel_group.add_pixel([4, 5, 6], 'red')
        self.pixel_group.update_mean('red')
        np.testing.assert_array_equal(self.pixel_group.get_mean('red'), np.array([2.5, 3.5, 4.5]))

    def test_get_variance(self):
        self.pixel_group.add_pixel([1, 2, 3], 'red')
        self.pixel_group.add_pixel([4, 5, 6], 'red')
        self.pixel_group.update_variance('red')
        np.testing.assert_array_equal(self.pixel_group.get_variance('red'), np.array([2.25, 2.25, 2.25]))

if __name__ == '__main__':
    unittest.main()