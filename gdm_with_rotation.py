import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import scipy.ndimage
import numpy as np
from mosaic import rectangular_decomposition
from mosaic.utilities import plot_image_decomposition
from pathlib import Path

def rotate_image(image, angel , interpolation):
    return  scipy.ndimage.rotate(image,angel,reshape=True,order=interpolation)


def gdm_with_rotation(file_name, angle, show_original=False, show_decomposed=False):
    image = Image.open(file_name)
    array_image = np.array(image)
    binary_image = np.array(array_image).astype(bool).astype(int)
    
    if show_original:
        plt.imshow(binary_image, cmap='gray')
        plt.title(f"Original Image: {file_name}")
        plt.axis("off")
        plt.show()
            
    
    binary_image=rotate_image(array_image,angle,0)
    
    imagesize_y, imagesize_x = binary_image.shape
    y_offset = round(imagesize_y / 2, 3)
    x_offset = round(imagesize_x / 2, 3)
    #print(f"offset_y:{y_offset}, offset_x: {x_offset}")
    num_zeros = np.count_nonzero(binary_image == 0)
    rectangles = rectangular_decomposition(binary_image)
    for index, rect in enumerate(rectangles):
        y_size = round(round(rect.y_end - rect.y_start + 1, 2), 3)
        x_size = round(round(rect.x_end - rect.x_start + 1, 2), 3)
        y_corner = round(round(rect.y_start - y_offset, 2), 3)
        x_corner = round(round(rect.x_start - x_offset, 2), 3)
        y_center = round(y_corner + y_size/2, 3)
        x_center = round(x_corner + x_size/2, 3)
        #print(f"{y_corner}, {x_corner}, {y_center}, {x_center}, {y_size}, {x_size}")
        voxelname = "voxel_" + str(index + 1).zfill(4)

    #print("Number of object pixels:", num_zeros)
    print("Number of rectangles:", len(rectangles))

    if show_decomposed:
        plot_image_decomposition(binary_image, rectangles=rectangles)
    return rectangles

def iter_image_files(directory: str):
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"Directory {directory} does not exist or is not a directory.")
    for p in sorted(d.rglob("*")):
        if p.is_file():
            yield p

def batch_process_images(directory: str, theta: float):
    paths = list(iter_image_files(directory))
    success=0
    error=0
    for p in paths:
        try:
            print(f"Processing file: {p}")
            gdm_with_rotation(p, theta, show_decomposed=True)
            success += 1
        except Exception as e:
            error += 1
            print(f"Error processing file {p}: {e}")
    
    print(f"Processing completed. Success: {success}, Errors: {error}")

def find_best_rotation(file_name):
    theta_list = np.linspace(0.0, 180.0, 180, endpoint=False)
    
    best_angle = 0
    number_of_rectangles = float('inf')
    worst_angle = 0
    max_number_of_rectangles = 0
    
    for theta in theta_list:
        print(f"Testing angle: {theta}")
        rectangles = gdm_with_rotation(file_name, theta)
        print("\n")
        if len(rectangles) < number_of_rectangles:
            number_of_rectangles = len(rectangles)
            best_angle = theta
        if len(rectangles) > max_number_of_rectangles:
            max_number_of_rectangles = len(rectangles)
            worst_angle = theta
    
    gdm_with_rotation(file_name, best_angle, show_original=True, show_decomposed=True)
    print(f"Best angle: {best_angle} with {number_of_rectangles} rectangles")
    print(f"Worst angle: {worst_angle} with {max_number_of_rectangles} rectangles")
    
    
if __name__ == "__main__":
    #gdm_with_rotation(r'TestImages\simple.tif', angle=45, show_original=True, show_decomposed=True)
    #batch_process_images(r'TestImages', theta=45)
    find_best_rotation(r'TestImages\simple.tif')
