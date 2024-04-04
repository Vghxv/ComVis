import cv2
import numpy as np
from collections import defaultdict


def rgb_to_binary(rtb_image, red_weight=0.3, green_weight=0.3, blue_weight=0.4, threshold=128):
    """
    Convert the color image to a binary image using a weighted sum of the color channels.

    Args:
        image: NumPy array, the input color image.
        red_weight: Float, the weight of the red channel.
        green_weight: Float, the weight of the green channel.
        blue_weight: Float, the weight of the blue channel.
        threshold: Integer, the threshold value.

    Returns:
        NumPy array, the binary image.
    """
    r, g, b = rtb_image[:, :, 0], rtb_image[:, :, 1], rtb_image[:, :, 2]
    gray_image = np.uint8(red_weight * r + green_weight * g + blue_weight * b)
    binary_image_result = np.where(gray_image < threshold, 0, 255).astype(np.uint8)
    return binary_image_result

def pixel_color(value):
    """
    Generate a color for the pixel.
    
    Args:
        value: Integer, the label of the pixel.

    Returns:
        Tuple, the color of the pixel.
    """
    r = (value * 123) % 255
    g = (value * 233) % 255
    b = (value * 193) % 255
    return (r, g, b)

def four_neighbor(image_param, height, width):
    """
    Traverse the image using a 4-neighbor.
    
    Args:
        image: NumPy array, the input image.
        height: Integer, the height of the image.
        width: Integer, the width of the image.
        
    Returns:
        NumPy array, the result image.
    """
    # Initialize the variables
    eq_list = []
    label = 1

    # Add padding to the image
    result_image = image_param.copy().astype(np.uint16)
    result_image = np.pad(result_image, ((1, 0), (1, 0)), 'constant', constant_values=(255, 255))

    # Traverse the image, find the connected components and assign labels
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            if result_image[y, x] == 0:
                neighbors = [
                    n for n in
                    [
                        result_image[y - 1, x],
                        result_image[y, x - 1]
                    ]
                    if n != 255
                ]
                if len(neighbors) == 0:
                    result_image[y, x] = label
                    label += 1
                    if label == 255:
                        label += 1
                else:
                    result_image[y, x] = neighbors[0]
                    for n in neighbors:
                        if n != result_image[y, x]:
                            eq_list.append((n, result_image[y, x]))
    result_image = result_image[1:, 1:]
    return result_image, eq_list

def eight_neighbor(image_param, height, width):
    """
    Traverse the image using a 8-neighbor.
    
    Args:
        image: NumPy array, the input image.
        height: Integer, the height of the image.
        width: Integer, the width of the image.
        
    Returns:
        NumPy array, the result image.
    """
    eq_list = []
    label = 1

    # Add padding to the image
    result_image = image_param.copy().astype(np.uint16)
    result_image = np.pad(result_image, ((1, 0), (1, 0)), 'constant', constant_values=(255, 255))

    # Traverse the image, find the connected components and assign labels
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            if result_image[y, x] == 0:
                neighbors = [
                    n for n in
                    [
                        result_image[y - 1, x - 1],
                        result_image[y - 1, x],
                        result_image[y - 1, x + 1] if x + 1 <= width else 255,
                        result_image[y, x - 1]
                    ]
                    if n != 255
                ]
                if len(neighbors) == 0:
                    result_image[y, x] = label
                    label += 1
                    if label == 255:
                        label += 1
                else:
                    result_image[y, x] = neighbors[0]
                    for n in neighbors:
                        if n != result_image[y, x]:
                            eq_list.append((n, result_image[y, x]))
    result_image = result_image[1:, 1:]
    return result_image, eq_list

def traverse_image(image_param, fn=True):
    """
    Traverse the image using a image.

    Args:
        image: NumPy array, the input image.

    Returns:
        NumPy array, the result image.
    """
    height, width = image_param.shape
    if fn:
        result_image, eq_list = four_neighbor(image_param, height, width)
    else:
        result_image, eq_list = eight_neighbor(image_param, height, width)
    
    # replace the labels with the smallest label in the connected components
    def dfs(node, visited, cluster):
        visited.add(node)
        cluster.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, cluster)
    graph = defaultdict(list)
    for u, v in eq_list:
        graph[u].append(v)
        graph[v].append(u)
    clusters = []
    visited = set()
    for node in graph:
        if node not in visited:
            cluster = []
            dfs(node, visited, cluster)
            clusters.append(cluster)
    for _, cluster in enumerate(clusters):
        for node in cluster:
            result_image[result_image == node] = min(cluster)
    
    # Remove the small clusters
    cluster_size = {}
    for y in range(0, height):
        for x in range(0, width):
            if result_image[y, x] != 255:
                if result_image[y, x] in cluster_size:
                    cluster_size[result_image[y, x]] += 1
                else:
                    cluster_size[result_image[y, x]] = 1
    for item in cluster_size.items():
        if item[1] < 500:
            result_image[result_image == item[0]] = 255    
    
    # Generate the color result image 
    color_result_image = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            if result_image[y, x] != 255:
                color_result_image[y, x] = pixel_color(result_image[y, x])
    return color_result_image


if __name__ == '__main__':
    for i in range(1, 5):
        image = cv2.imread(f'images/img{i}.png')
        binary_image = rgb_to_binary(image, threshold=240)
        result = traverse_image(binary_image, fn=True)
        cv2.imwrite(f'results/img_{i}_4.png', result)
        result = traverse_image(binary_image, fn=False)
        cv2.imwrite(f'results/img_{i}_8.png', result)
