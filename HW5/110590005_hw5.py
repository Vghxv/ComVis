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

    def set_image_first_channel(self):
        self.image = self.image[:, :, 0]

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
    def mean_filter(self):
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.padding_size, self.height+self.padding_size):
                for j in range(self.padding_size, self.width+self.padding_size):
                    pixels = self.get_n_by_n_region_pixel(i, j)
                    self.result_image[i-self.padding_size, j-self.padding_size] = np.mean(pixels)
                    bar()

    def mean_filter_process(self, param):
        self.mask_size = param
        self.padding()
        self.mean_filter()
        self.remove_padding()

    @set_result_image    
    def median_filter(self):
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.padding_size, self.height+self.padding_size):
                for j in range(self.padding_size, self.width+self.padding_size):
                    pixels = self.get_n_by_n_region_pixel(i, j)
                    self.result_image[i-self.padding_size, j-self.padding_size] = np.median(pixels)
                    bar()

    def median_filter_process(self, mask_size):
        self.mask_size = mask_size
        self.padding()
        self.median_filter()
        self.remove_padding()

    @set_result_image
    def gaussian_filter(self):
        self.kernel = np.zeros((self.mask_size, self.mask_size))
        for i in range(self.mask_size):
            for j in range(self.mask_size):
                self.kernel[i, j] = self.G(i - self.mask_size//2, j - self.mask_size//2, self.sigma)
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
        with alive_bar(self.height * self.width) as bar:
            for i in range(self.padding_size, self.height+self.padding_size):
                for j in range(self.padding_size, self.width+self.padding_size):
                    pixels = self.get_n_by_n_region_pixel(i, j)
                    pixels = np.array(pixels).reshape(3, 3)
                    self.result_image[i-self.padding_size, j-self.padding_size] = (np.sum(pixels * self.kernel)**2 + np.sum(pixels * self.kernel.T)**2) * 0.1 + self.image[i, j]
                    bar()

    def G(self, x, y, sigma):
        return np.exp(-((x**2 + y**2)/(2*sigma**2))) / (2*np.pi*sigma**2)

    def gaussian_filter_process(self, mask_size, sigma):
        self.mask_size = mask_size
        self.sigma = sigma
        self.padding()
        self.gaussian_filter()
        self.remove_padding()

    def combo_process(self):
        self.mask_size = 3
        self.padding()
        self.median_filter()
        self.image = self.result_image

        self.mask_size = 7
        self.sigma = 0.5
        self.padding()
        self.gaussian_filter()
        self.image = self.result_image

        self.mask_size = 3
        self.padding()
        self.sobel_filter()

        self.remove_padding()

if __name__ == '__main__':
    for i in range(1, 4):
        image = cv2.imread(f'./images/img{i}.jpg')
        processing = Processing(image)
        processing.set_image_first_channel()
        processing.mean_filter_process(3)
        cv2.imwrite(f'./results/img{i}_q1_3.jpg', processing.result_image)
        processing.mean_filter_process(7)
        cv2.imwrite(f'./results/img{i}_q1_7.jpg', processing.result_image)
        processing.median_filter_process(3)
        cv2.imwrite(f'./results/img{i}_q2_3.jpg', processing.result_image)
        processing.median_filter_process(7)
        cv2.imwrite(f'./results/img{i}_q2_7.jpg', processing.result_image)
        processing.gaussian_filter_process(5, 1)
        cv2.imwrite(f'./results/img{i}_q3.jpg', processing.result_image)
        processing.combo_process()
        cv2.imwrite(f'./results/img{i}_q4.jpg', processing.result_image)