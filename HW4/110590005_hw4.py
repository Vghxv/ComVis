import cv2
import numpy as np
import heapq
from alive_progress import alive_bar, config_handler
config_handler.set_global(spinner='dots_waves2', bar='blocks', unknown='stars')


class PriorityQueue:
    def __init__(self):
        self.elements = []
    def debug(self):
        with open('queue.txt', 'w') as f:
            for element in self.elements:
                f.write(f'{element[0]} {element[1]} {element[2]}\n')
    def is_empty(self):
        return len(self.elements) == 0

    def put(self, priority: int, x: int, y: int):
        heapq.heappush(self.elements, (priority, x, y))

    def get(self):
        return heapq.heappop(self.elements)

    def __len__(self):
        return len(self.elements)

class NeighborGroup:
    def __init__(self, image: np.ndarray, cor_list: list):
        self.labeled = []
        self.unlabeled = []
        for x, y in cor_list:
            if image[x, y] == 0:
                self.unlabeled.append((image[x, y], x, y))
            elif image[x, y] > 0:
                self.labeled.append((image[x, y], x, y))

    def get_labeled(self):
        return self.labeled

    def get_unlabeled(self):
        return self.unlabeled


class PixelGroup:
    def __init__(self):
        self.pixels: dict = {}
        self.means: dict = {}
        self.variances: dict = {}

    def debug(self):
        with open('pixels.txt', 'w') as f:
            for label in self.pixels:
                for pixel in self.pixels[label]:
                    f.write(f'{label} {pixel[0]} {pixel[1]} {pixel[2]}\n')
        with open('means.txt', 'w') as f:
            for label in self.means:
                f.write(f'{label} {self.means[label][0]} {self.means[label][1]} {self.means[label][2]}\n')
        with open('variances.txt', 'w') as f:
            for label in self.variances:
                f.write(f'{label} {self.variances[label][0]} {self.variances[label][1]} {self.variances[label][2]}\n')

    def add_pixel(self, pixel, label):
        if label not in self.pixels:
            self.pixels[label] = [np.array(pixel)]
        else:
            self.pixels[label].append(np.array(pixel))
        count = len(self.pixels[label])
        if count == 1:
            self.means[label] = np.array(pixel)
            self.variances[label] = np.zeros(3)
        else:
            self.means[label] = (self.means[label] * (count - 1) + np.array(pixel)) / count
            self.variances[label] = (self.variances[label] * (count - 1) + (np.array(pixel) - self.means[label]) ** 2) / count

    def priority(self, pixel, neighbor_label):
        mean_diff = pixel - self.means[neighbor_label]
        count = len(self.pixels[neighbor_label])
        new_mean = (self.means[neighbor_label] * count + pixel) / (count + 1)
        variance_diff = (self.variances[neighbor_label] * count + (np.array(pixel) - new_mean) ** 2) / (count + 1) - self.variances[neighbor_label]
        return mean_diff @ mean_diff + variance_diff @ variance_diff

    def update_mean_all(self):
        for label in self.pixels:
            self.means[label] = np.mean(self.pixels[label], axis=0)

    def update_variance_all(self):
        for label in self.pixels:
            self.variances[label] = np.var(self.pixels[label], axis=0)

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.variance


class Processing:
    def __init__(self, image: np.ndarray, labeled_image: np.ndarray):
        self.image = np.copy(image)
        self.labeled_image = np.copy(labeled_image)
        self.height, self.width = image.shape[:2]
        self.labels = np.zeros((self.height, self.width))
        self.pqueue = PriorityQueue()
        self.pixel_group = PixelGroup()
        self.global_priority_weight = 0.999

    def is_valid(self, x, y):
        return 0 <= x < self.height and 0 <= y < self.width

    def get_four_neighbors(self, x, y):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = [(x + dx, y + dy) for dx, dy in directions if self.is_valid(x + dx, y + dy)]
        return neighbors
    
    def get_3x3_region(self, x, y):
        directions = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if self.is_valid(x + dx, y + dy)]
        neighbors = [(x + dx, y + dy) for dx, dy in directions]
        return neighbors
    
    def edge_priority(self, x, y):
        sobel_horizontal_mask = np.array([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]])
        sobel_vertical_mask = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]])
        eight_neighbors_pixel = np.zeros(3)
        for nx, ny in self.get_3x3_region(x, y):
            eight_neighbors_pixel += self.image[nx, ny]
        horizontal_gradient = np.sum(np.abs(np.sum(eight_neighbors_pixel * sobel_horizontal_mask, axis=-1)))
        vertical_gradient = np.sum(np.abs(np.sum(eight_neighbors_pixel * sobel_vertical_mask, axis=-1)))
        return abs(horizontal_gradient) + abs(vertical_gradient)

    def match_color(self, seeds):
        for index, color in enumerate(seeds):
            mask = np.all(self.labeled_image == color, axis=-1)
            self.labels[mask] = index + 1
        i_array, j_array = np.where(self.labels > 0)
        for x, y in zip(i_array, j_array):
            self.pixel_group.add_pixel(self.image[x, y], self.labels[x, y])
        self.pixel_group.update_mean_all()
        self.pixel_group.update_variance_all()

    def mark_boundary(self):
        i_array, j_array = np.where(self.labels == 0)
        for i, j in zip(i_array, j_array):
            for nx, ny in self.get_four_neighbors(i, j):
                if self.labels[nx, ny] > 0:
                    self.labels[i, j] = -2
                    global_priority = self.pixel_group.priority(self.image[i, j], self.labels[nx, ny])
                    local_priority = self.edge_priority(i, j)
                    self.pqueue.put(global_priority * self.global_priority_weight + local_priority * (1 - self.global_priority_weight), i, j)
                    break

    def watershed_segmentation(self):
        label_num = len(np.where(self.labels < 1)[0])
        with alive_bar(label_num) as bar:
            while not self.pqueue.is_empty():
                _, x, y = self.pqueue.get()
                neighbor_coordinates = self.get_four_neighbors(x, y)
                
                neighbor_group = NeighborGroup(self.labels, neighbor_coordinates)
                neighbor_label_set = set([i[0] for i in neighbor_group.get_labeled()])

                if len(neighbor_label_set) > 1:
                    self.labels[x, y] = -1
                    continue

                label, _, _ = neighbor_group.get_labeled()[0]
                self.labels[x, y] = label
                self.pixel_group.add_pixel(self.image[x, y], label)
                for _, ux, uy in neighbor_group.get_unlabeled():
                    self.labels[ux, uy] = -2
                    global_priority = self.pixel_group.priority(self.image[ux, uy], label)
                    local_priority = self.edge_priority(ux, uy)
                    self.pqueue.put(global_priority * self.global_priority_weight + local_priority * (1 - self.global_priority_weight), ux, uy)
                bar()

    def color_original_image(self, seeds):
        markers_weight = 0.7
        image_weight = 1 - markers_weight
        output_image = np.zeros((self.height, self.width, 3))
        for i in range(self.height):
            for j in range(self.width):
                if self.labels[i, j] == -2:
                    output_image[i, j] = np.array([50, 50, 50]) * markers_weight + self.image[i, j] * image_weight
                elif self.labels[i, j] == -1:
                    output_image[i, j] = np.array([0, 0, 0]) * markers_weight + self.image[i, j] * image_weight
                elif self.labels[i, j] == 0:
                    output_image[i, j] = np.array([255, 255, 255]) * markers_weight + self.image[i, j] * image_weight
                else:
                    output_image[i, j] = np.array(seeds[int(self.labels[i, j]) - 1]) * markers_weight + self.image[i, j] * image_weight
        return output_image

if __name__ == '__main__':
    seeds = [
        [164, 73, 163], # 0 purple
        [204, 72, 63], # 1 indigo
        [232, 162, 0], # 2 turquoise
        [76, 177, 34], # 3 green
        [0, 242, 255], # 4 yellow
        [39, 127, 255], # 5 orange
        [36, 28, 237], # 6 red
        [21, 0, 136], # 7 dark red
        [231, 191, 200], # 8 lavender
        [190, 146, 112], # 9 blue gray
        [234, 217, 153], # 10 light turquoise
        [29, 230, 181], # 11 lime
    ]
    seed_list_index = [
        [0, 1, 2, 8],
        [i for i in range(12)],
        [5, 6, 7]
    ]
    for i in range(1, 4):
        image = cv2.imread(f'./images/img{i}.png')
        marker = cv2.imread(f'./markers/img{i}.png')
        image_processing = Processing(image, marker)
        question_seeds = [seeds[j] for j in seed_list_index[i - 1]]
        image_processing.match_color(question_seeds)
        image_processing.mark_boundary()
        image_processing.watershed_segmentation()
        output_image = image_processing.color_original_image(question_seeds)
        cv2.imwrite(f'./results/img{i}_q1.png', output_image)
