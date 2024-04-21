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


def eight_distance_transform(image_input):
    b_image = image_input.copy()
    b_image = np.pad(b_image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    height, width = image_input.shape
    depth = 1
    pre_depth = depth
    while True: 
        max_depth = 0
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                if b_image[i, j] == depth:
                    eight_neighbors = [
                        b_image[i - 1, j - 1], b_image[i - 1, j], b_image[i - 1, j + 1],
                        b_image[i, j - 1], b_image[i, j + 1],
                        b_image[i + 1, j - 1], b_image[i + 1, j], b_image[i + 1, j + 1]
                    ]
                    min_distance = min(eight_neighbors)
                    max_depth = min_distance
                    b_image[i, j] = min_distance + 1
        if max_depth == pre_depth:
            break
        pre_depth = max_depth
        depth += 1
    b_image = b_image[1:-1, 1:-1]
    return b_image, depth


def remain_local_maxima(image_input):
    b_image = image_input.copy()
    b_image = np.pad(b_image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    height, width = image_input.shape
    result = np.zeros((height + 2, width + 2))
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if b_image[i, j] > 0:
                eight_neighbors = [
                    # b_image[i-1, j-1], 
                    b_image[i-1, j], 
                    # b_image[i-1, j+1],
                    
                    b_image[i, j-1], 
                    b_image[i, j + 1],
                    
                    # b_image[i+1, j-1], 
                    b_image[i+1, j], 
                    # b_image[i+1, j+1]
                ]
                max_distance = max(eight_neighbors)
                if max_distance == b_image[i, j]:
                    result[i, j] = 1

    result = result[1:-1, 1:-1]
    return result


# def can_be_removed(b_image, tlx, tly, brx, bry):
#     unit = b_image[tlx:brx + 1, tly:bry + 1]
#     unit[1, 1] = 0
#     orphan = 0
#     for y in range(3):
#         for x in range(3):
#             if unit[y, x] > 0:
#                 eight_neighbors = [n for n in 
#                         [
#                         unit[y - 1, x - 1] if y - 1 >= 0 and x - 1 >= 0 else 0, 
#                         unit[y - 1, x] if y - 1 >= 0 else 0,
#                         unit[y - 1, x + 1] if y - 1 >= 0 and x + 1 < 3 else 0,

#                         unit[y, x - 1] if x - 1 >= 0 else 0,
#                         unit[y, x + 1] if x + 1 < 3 else 0,
                        
#                         unit[y + 1, x - 1] if y + 1 < 3 and x - 1 >= 0 else 0,
#                         unit[y + 1, x] if y + 1 < 3 else 0,
#                         unit[y + 1, x + 1] if y + 1 < 3 and x + 1 < 3 else 0
#                     ]
#                     if n != 0
#                 ]
#                 if len(eight_neighbors) == 0:
#                     orphan += 1
#     if orphan > 1:
#         return False
#     return True

def erosion(b_image, kernel):
    height, width = b_image.shape
    result = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if b_image[i, j] == 1:
                if np.array_equal(b_image[i : i + 3, j : j + 3], kernel):
                    result[i, j] = 1
    result = result.astype(np.uint8)
    return result


def dilation(b_image, kernel):
    height, width = b_image.shape
    result = np.zeros((height + 2, width + 2))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if b_image[i, j] == 1:
                result[i + 1 : i + 4, j + 1 : j + 4] = kernel
    result = result.astype(np.uint8)
    result = result[1:-1, 1:-1]
    return result


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


def remove_depth_pixels(image_input, depth):
    result = image_input.copy()
    result = np.where(result < depth, 0, 1).astype(np.uint8)
    return result


if __name__ == '__main__':
    for i in range(4, 6):
        image = cv2.imread(f'images/img{i}.jpg')
        h, w = image.shape[:2]
        binary_image = rgb_to_binary(image)
        if i == 1:
            binary_image = closing(binary_image, 2)
            binary_image = opening(binary_image, 2)

        # q1_1
        result_1, depth = eight_distance_transform(binary_image)
        result_color = distance_color_image(result_1, depth)
        cv2.imwrite(f'results/img{i}_q1_1.jpg', result_color)
        
        result_2 = remain_local_maxima(result_1)
        # kernel = np.array([
        #     [1, 1, 1],
        #     [1, 1, 1],
        #     [1, 1, 1]
        # ])
        # result_2 = dilation(result_2, kernel)
        # result_2 = dilation(result_2, kernel)
        # result_2 = erosion(result_2, kernel)
        result_2 = np.where(result_2 == 1, 255, 0).astype(np.uint8)
        cv2.imwrite(f'dev/img{i}_q1_2.jpg', result_2)
        # with open(f'txts/img{i}_q1_2.txt', 'w') as f:
        #     for row in result_m:
        #         f.write(' '.join([str(x) for x in row]) + '\n')
        
        
        # cv2.imwrite(f'dev/img{i}_q1_2.jpg', result_color)
        # break
        # resultf = medial_axis(result)
        # cv2.imwrite(f'results/img{i}_q1_2.jpg', resultf)
