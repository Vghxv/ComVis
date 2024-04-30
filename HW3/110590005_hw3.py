import cv2
import numpy as np


def rgb_to_binary(rtb_image, red_weight=0.3, green_weight=0.3, blue_weight=0.4, threshold=128):
    r, g, b = rtb_image[:, :, 0], rtb_image[:, :, 1], rtb_image[:, :, 2]
    gray_image = np.uint8(red_weight * r + green_weight * g + blue_weight * b)
    binary_image_result = np.where(gray_image < threshold, 1, 0).astype(np.uint8)
    return binary_image_result


def map_value_to_pixel(value, max_value):
    return int(value * 255 / max_value)


def distance_color_image(image_input, max_value):
    height, width = image_input.shape
    result_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result_image[i, j] = [map_value_to_pixel(image_input[i, j], max_value)] * 3
    return result_image


def color_image(image_input):
    result = image_input.copy()
    result = np.where(result == 1, 255, 0).astype(np.uint8)
    return result


def get_eight_neighbors(image_input, i, j):
    return np.array([image_input[i - 1, j - 1: j + 2], image_input[i, j - 1: j + 2], image_input[i + 1, j - 1: j + 2]])


def get_maximal_distance_in_neighbors(image_input, i, j):
    neighbors = get_eight_neighbors(image_input, i, j)
    flatten_neighbors = neighbors.flatten()
    flatten_neighbors = np.delete(flatten_neighbors, 4)
    return np.max(flatten_neighbors)


def four_distance_transform(image_input):
    b_image = image_input.copy()
    b_image = np.pad(b_image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    height, width = image_input.shape
    depth = 1
    pre_depth = depth
    while True: 
        max_depth = 0
        for j in range(1, width + 1):
            for i in range(1, height + 1):
                if b_image[i, j] == depth:
                    four_neighbors = [
                        b_image[i - 1, j],
                        b_image[i, j - 1],
                        b_image[i, j + 1],
                        b_image[i + 1, j],
                    ]
                    min_distance = min(four_neighbors)
                    max_depth = min_distance
                    b_image[i, j] = min_distance + 1
        if max_depth == pre_depth:
            break
        pre_depth = max_depth
        depth += 1
    return b_image[1:-1, 1:-1]


def match_pattern(neighbors):
    neighbors_copy = neighbors.copy()
    neighbors_copy[1, 1] = 0 
    if np.sum(neighbors_copy[1, :]) == 0 and np.sum(neighbors_copy[2, :]) != 0 and np.sum(neighbors_copy[0, :]) != 0:
        return True
    elif np.sum(neighbors_copy[:, 1]) == 0 and np.sum(neighbors_copy[:, 2]) != 0 and np.sum(neighbors_copy[:, 0]) != 0:
        return True
    elif neighbors_copy[1, 0] != 0 and neighbors_copy[0, 1] != 0 and neighbors_copy[2, 2] != 0 and neighbors_copy[1, 2] == 0 and neighbors_copy[2, 1] == 0:
        return True
    elif neighbors_copy[1, 2] != 0 and neighbors_copy[0, 1] != 0 and neighbors_copy[2, 0] != 0 and neighbors_copy[1, 0] == 0 and neighbors_copy[2, 1] == 0:
        return True
    elif neighbors_copy[1, 2] != 0 and neighbors_copy[2, 1] != 0 and neighbors_copy[0, 0] != 0 and neighbors_copy[0, 1] == 0 and neighbors_copy[1, 0] == 0:
        return True
    elif neighbors_copy[1, 0] != 0 and neighbors_copy[2, 1] != 0 and neighbors_copy[0, 2] != 0 and neighbors_copy[0, 1] == 0 and neighbors_copy[1, 2] == 0:
        return True
    elif ((neighbors_copy[0, 0] != 0 and neighbors_copy[2, 2] != 0) or (neighbors_copy[2, 0] != 0 and neighbors_copy[2, 0] != 0)) and neighbors_copy[1, 2] == 0 and neighbors_copy[2, 1] == 0 and neighbors_copy[1, 0] == 0 and neighbors_copy[0, 1] == 0: 
        return True
    return False


def remain_local_maxima(image_input):
    result_image = image_input.copy()
    distance_image = image_input.copy()
    result_image = np.pad(result_image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    distance_image = np.pad(distance_image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    height, width = image_input.shape
    depth = np.max(image_input)
    for d in range(1, depth+1):
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                if distance_image[i, j] == d:
                    eight_neighbors = get_eight_neighbors(result_image, i, j)
                    max_distance = get_maximal_distance_in_neighbors(distance_image, i, j)
                    if max_distance > distance_image[i, j] and not match_pattern(eight_neighbors):
                        result_image[i, j] = 0
    return result_image[1:-1, 1:-1]

def remove_redundant_pixels(image_input):
    result_image = image_input.copy()
    result_image = np.pad(result_image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    height, width = image_input.shape
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if result_image[i, j] != 0 and result_image[i, j + 1] != 0:
                eight_neighbors = get_eight_neighbors(result_image, i, j)
                for u in range(3):
                    if u == 1:
                        continue
                    for v in range(3):
                        if eight_neighbors[u, v] != 0:
                            target_neighbors = get_eight_neighbors(result_image, i + u - 1, j + v - 1)
                            if not match_pattern(target_neighbors):
                                result_image[i + u - 1, j + v - 1] = 0
            if result_image[i, j] != 0 and result_image[i + 1, j] != 0:
                eight_neighbors = get_eight_neighbors(result_image, i, j)
                for u in range(3):
                    for v in range(3):
                        if v == 1:
                            continue
                        if eight_neighbors[u, v] != 0:
                            target_neighbors = get_eight_neighbors(result_image, i + u - 1, j + v - 1)
                            if not match_pattern(target_neighbors):
                                result_image[i + u - 1, j + v - 1] = 0
            if result_image[i - 1, j] != 0 and result_image[i, j - 1] != 0:
                target_neighbors = get_eight_neighbors(result_image, i - 1, j - 1)
                if not match_pattern(target_neighbors):
                    result_image[i - 1, j - 1] = 0
                target_neighbors = get_eight_neighbors(result_image, i, j)
                if not match_pattern(target_neighbors):
                    result_image[i, j] = 0
            if result_image[i, j] != 0 and result_image[i - 1, j - 1] != 0:
                target_neighbors = get_eight_neighbors(result_image, i, j - 1)
                if not match_pattern(target_neighbors):
                    result_image[i, j - 1] = 0
                target_neighbors = get_eight_neighbors(result_image, i - 1, j)
                if not match_pattern(target_neighbors):
                    result_image[i - 1, j] = 0
    return result_image[1:-1, 1:-1]


def erosion(b_image, kernel):
    height, width = b_image.shape
    result = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if b_image[i, j] == 1:
                if np.array_equal(b_image[i : i + 3, j : j + 3], kernel):
                    result[i, j] = 1
    return result.astype(np.uint8)


def dilation(b_image, kernel):
    height, width = b_image.shape
    result = np.zeros((height + 2, width + 2))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if b_image[i, j] == 1:
                result[i + 1 : i + 4, j + 1 : j + 4] = kernel
    result = result.astype(np.uint8)
    return result[1:-1, 1:-1]


def closing(image_input, max_iter = 1):
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    result = image_input.copy()
    for _ in range(max_iter):
        result = dilation(result, kernel)
    for _ in range(max_iter):
        result = erosion(result, kernel)
    return result


def opening(image_input, max_iter = 1):
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    result = image_input.copy()
    for _ in range(max_iter):
        result = erosion(result, kernel)
    for _ in range(max_iter):
        result = dilation(result, kernel)
    return result


if __name__ == '__main__':
    for i in range(1, 5):
        image = cv2.imread(f'images/img{i}.jpg')
        binary_image = rgb_to_binary(image)
        if i == 1:
            binary_image = closing(binary_image, 2)
            binary_image = opening(binary_image, 2)
        result_1 = four_distance_transform(binary_image)
        result_color = distance_color_image(result_1, np.max(result_1))
        cv2.imwrite(f'results/img{i}_q1_1.jpg', result_color)
        result_2 = remain_local_maxima(result_1)
        result_2 = remove_redundant_pixels(result_2)
        result_2_color = np.where(result_2 > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(f'results/img{i}_q1_2.jpg', result_2_color)
