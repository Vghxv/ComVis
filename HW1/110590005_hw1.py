import cv2
import numpy as np


def rgb_to_grayscale(image):
    """
    Convert the color image to a grayscale image.

    Args:
        image: NumPy array, the input color image.

    Returns:
        NumPy array, the gray110590005_hw1" doesn't conformscale image.
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray_image = np.uint8(0.3 * r + 0.3 * g + 0.4 * b)
    return gray_image

def grayscale_to_binary(gray_image, threshold=128):
    """
    Convert the grayscale image to a binary image using a threshold.
    
    Args:
        gray_image: NumPy array, the input grayscale image.
        threshold: Integer, the threshold value.
        
    Returns:
        NumPy array, the binary image.
    """
    binary_image = np.where(gray_image < threshold, 0, 255).astype(np.uint8)
    return binary_image

def convert_to_indexed(image, num_colors):
    """
    Convert the color image to an indexed-color image.
    
    Args:
        image: NumPy array, the input color image.
        num_colors: Integer, the number of colors in the indexed-color image.
        
    Returns:
        NumPy array, the indexed-color image.
        NumPy array, the palette.
    """
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    indexed_image = centers[labels.flatten()]
    indexed_image = indexed_image.reshape(image.shape)
    palette = np.unique(centers, axis=0)
    return indexed_image, palette


def resize_nearest_neighbor(image, scale_factor):
    """
    Resize the image using nearest neighbor interpolation.

    Args:
        image: NumPy array, the input image.
        scale_factor: Float, the scaling factor.
    
    Returns:
        NumPy array, the resized image.
    """
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    i_indices = (np.arange(new_height) / scale_factor).astype(int)
    j_indices = (np.arange(new_width) / scale_factor).astype(int)
    resized_image = image[i_indices[:, np.newaxis], j_indices]    
    return resized_image


def bilinear_interpolation(image, scale_factor):
    """
    Resize the image using bilinear interpolation.

    Args:
        image: NumPy array, the input image.
        scale_factor: Float, the scaling factor.
    
    Returns:
        NumPy array, the resized image.
    """
    old_height, old_width = image.shape[:2]
    new_height, new_width = int(old_height * scale_factor), int(old_width * scale_factor)
    row_indices = (np.arange(new_height) * (old_width / new_width)).astype(int)
    col_indices = (np.arange(new_width) * (old_height / new_height)).astype(int)
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            row_low, row_high = row_indices[i], min(row_indices[i] + 1, old_height - 1)
            col_low, col_high = col_indices[j], min(col_indices[j] + 1, old_width - 1)
            dy = (row_indices[i] - row_low) / (row_high - row_low + 1e-6)
            dx = (col_indices[j] - col_low) / (col_high - col_low + 1e-6)
            resized_image[i, j] = (
                (1 - dx) * (1 - dy) * image[row_low, col_low] +
                dx * (1 - dy) * image[row_low, col_high] +
                (1 - dx) * dy * image[row_high, col_low] +
                dx * dy * image[row_high, col_high]
            )
    return resized_image

if __name__ == "__main__":
    for i in range(1,4):
        color_image = cv2.imread(f"./images/img{i}.png")
        gray_image = rgb_to_grayscale(color_image)
        cv2.imwrite(f"results/img{i}_q1-1.jpg", gray_image)

        binary_image = grayscale_to_binary(gray_image)
        cv2.imwrite(f"results/img{i}_q1-2.jpg", binary_image)

        index_image, _ = convert_to_indexed(color_image, 16)
        cv2.imwrite(F"results/img{i}_q1-3.jpg", index_image)
        # np.savetxt('palette.txt', palette, fmt='%d')

        nearest_image = resize_nearest_neighbor(color_image, 0.5)
        cv2.imwrite(f"results/img{i}_q2-1-half.jpg", nearest_image)
        nearest_image = resize_nearest_neighbor(color_image, 2)
        cv2.imwrite(f"results/img{i}_q2-1-double.jpg", nearest_image)

        bilinear_image = bilinear_interpolation(color_image, 0.5)
        cv2.imwrite(f"results/img{i}_q2-2-half.jpg", bilinear_image)
        bilinear_image = bilinear_interpolation(color_image, 2)
        cv2.imwrite(f"results/img{i}_q2-2-double.jpg", bilinear_image)
