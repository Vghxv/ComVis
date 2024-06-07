import cv2
import numpy as np
from alive_progress import alive_bar, config_handler
config_handler.set_global(spinner='classic', bar='classic', unknown='stars')

class Processing:
    def __init__(self, image):
        self.image = image
        self.height, self.width = self.image.shape[:2]
        self.padding_size = 0
        self.mask_size = 0
        self.kernel = None

    def rgb_to_gray(self):
        self.image = np.uint8(0.3 * self.image[:, :, 0] + 0.3 * self.image[:, :, 1] + 0.4 * self.image[:, :, 2])

    def set_mask_size(self, mask_size):
        self.mask_size = mask_size
        self.padding_size = mask_size // 2

    def padding(self):
        self.padding_size = self.mask_size - 1
        self.image = np.pad(self.image, ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size)), 'constant', constant_values=0)

    def remove_padding(self):
        self.image = self.image[self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        self.padding_size = 0

    def is_valid(self, x, y):
        return 0 <= x < self.height + self.padding_size * 2 and 0 <= y < self.width + self.padding_size * 2

    def get_n_by_n_region_pixel(self, x, y):
        n = self.mask_size
        directions = [(dx, dy) for dx in range(-n//2+1, n//2+1) for dy in range(-n//2+1, n//2+1) if self.is_valid(x + dx, y + dy)]
        neighbors = [(x + dx, y + dy) for dx, dy in directions]
        pixels = [self.image[x, y] for x, y in neighbors]
        return pixels

    def set_result_image(func, *args, **kwargs):
        def wrapper(self):
            self.result_image = np.zeros((self.height, self.width))
            return func(self, *args, **kwargs)
        return wrapper

    @set_result_image
    def gaussian_filter(self):
        self.kernel = np.zeros((self.mask_size, self.mask_size))
        for i in range(self.mask_size):
            for j in range(self.mask_size):
                self.kernel[i, j] = self.G(i - self.mask_size//2, j - self.mask_size//2, self.sigma)
        print("gaussian filter processing...")
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.padding_size, self.height+self.padding_size):
                for j in range(self.padding_size, self.width+self.padding_size):
                    pixels = self.get_n_by_n_region_pixel(i, j)
                    pixels = np.array(pixels).reshape(self.mask_size, self.mask_size)
                    self.result_image[i-self.padding_size, j-self.padding_size] = np.sum(pixels * self.kernel)
                    bar()

    @set_result_image
    def sobel_filter(self):
        self.kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        print("sobel filter processing...")
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.padding_size, self.height+self.padding_size):
                for j in range(self.padding_size, self.width+self.padding_size):
                    pixels = self.get_n_by_n_region_pixel(i, j)
                    pixels = np.array(pixels).reshape(3, 3)
                    self.result_image[i-self.padding_size, j-self.padding_size] = np.sqrt((np.sum(pixels * self.kernel)**2 + np.sum(pixels * self.kernel.T)**2))
                    bar()

    def sobel_filter_process(self):
        self.padding()
        self.sobel_filter()
        self.remove_padding()

    def G(self, x, y, sigma):
        return np.exp(-((x**2 + y**2)/(2*sigma**2))) / (2*np.pi*sigma**2)

    def gaussian_filter_process(self, sigma):
        self.sigma = sigma
        self.padding()
        self.gaussian_filter()
        self.remove_padding()

    def non_max_suppression(self):
        print("non max suppression processing...")
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.padding_size, self.height+self.padding_size):
                for j in range(self.padding_size, self.width+self.padding_size):
                    pixels = self.get_n_by_n_region_pixel(i, j)
                    pixels.remove(self.image[i, j])
                    if self.result_image[i - self.padding_size, j - self.padding_size] < max(pixels):
                        self.result_image[i - self.padding_size, j - self.padding_size] = 0
                    bar()

    def non_max_suppression_process(self):
        self.padding()
        self.non_max_suppression()
        self.remove_padding()

    def double_threshold(self, low, high):
        print("double threshold processing...")
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.height):
                for j in range(self.width):
                    if self.result_image[i, j] > high:
                        self.result_image[i, j] = 255
                    elif self.result_image[i, j] < low:
                        self.result_image[i, j] = 0
                    else:
                        self.result_image[i, j] = 127
                    bar()

    def edge_tracking_by_hysteresis(self):
        print("edge tracking by hysteresis processing...")
        with alive_bar(self.height * self.width) as bar:
            for i in range(1, self.height-1):
                for j in range(1, self.width-1):
                    if self.result_image[i, j] == 127:
                        if 255 in self.result_image[i-1:i+2, j-1:j+2]:
                            self.result_image[i, j] = 255
                        else:
                            self.result_image[i, j] = 0
                    bar()

if __name__ == '__main__':
    for i in range(1, 4):
        image = cv2.imread(f'./images/img{i}.jpg')
        print(f'Processing image{i}.jpg')
        processing = Processing(image)
        processing.rgb_to_gray()
        processing.set_mask_size(5)
        processing.gaussian_filter_process(0.3)
        processing.set_mask_size(3)
        processing.sobel_filter_process()
        processing.non_max_suppression_process()
        processing.double_threshold(50, 150)
        processing.edge_tracking_by_hysteresis()
        cv2.imwrite(f'./results/img{i}_sobel.jpg', processing.result_image)