from dataclasses import dataclass
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import random
from collections import defaultdict
from skimage.measure import label, regionprops

@dataclass(frozen=True)
class Ray:
    start: tuple[float, float]
    end: tuple[float, float]

def find_corners(binary_image):
    obj0 = (binary_image == 0)
    obj = np.pad(obj0, 1, mode="constant", constant_values=False)

    H, W = obj.shape
    corners = []
    corners_NE = []
    corners_SW = []
    corners_triangle = []

    for i in range(H - 1):
        for j in range(W - 1):
            block = obj[i:i+2, j:j+2]
            s = int(block.sum())

            if s == 3:
                bi, bj = np.argwhere(~block)[0]
                r, c = i + bi - 1, j + bj - 1 
            elif s == 1:
                oi, oj = np.argwhere(block)[0]
                bi, bj = 1 - oi, 1 - oj
                r, c = i + bi - 1, j + bj - 1
            else:
                continue

            if 0 <= r < obj0.shape[0] and 0 <= c < obj0.shape[1]:
                if binary_image[r, c] != 0:
                    if binary_image[r-1, c+1]!=0:
                        corners_NE.append((r, c))
                    if binary_image[r+1, c-1]!=0:
                        corners_SW.append((r, c))
                    if (binary_image[r-1, c+1]==0) and (binary_image[r+1, c-1]==0):
                        corners_triangle.append((r, c))
                    corners.append((r, c))

    corners = list(dict.fromkeys(corners))
    return corners, corners_NE, corners_SW, corners_triangle

def show_corners(image, corners):
    plt.imshow(image, cmap='gray')
    y_coords, x_coords = zip(*corners)
    plt.scatter(x_coords, y_coords, c='red', s=10)
    plt.title("Detected Vertices")
    plt.axis("off")
    plt.show()
    
def show_corners_classes(image, diff_corners):
    plt.imshow(image, cmap='gray')
    colors = ['red', 'blue', 'green']
    i=0
    for corner_class in diff_corners:
        if(len(corner_class)==0):
            continue
        y_coords, x_coords = zip(*corner_class)
        plt.scatter(x_coords, y_coords, c=colors[i], s=10)
        i+=1
    plt.title("Detected NE Vertices")
    plt.axis("off")
    plt.show()

def project_ray_diagonal(image, corners, di, dj):
    H, W = image.shape
    corner_set = set(corners)
    rays = []

    for start in corners:
        i, j = start

        while True:
            ni, nj = i + di, j + dj

            if not (0 <= ni < H and 0 <= nj < W):
                rays.append(Ray(start=start, end=(i, j)))
                break

            if (ni, nj) in corner_set and (ni, nj) != start:
                rays.append(Ray(start=start, end=(ni, nj)))
                break

            if image[ni, nj] == 0:
                rays.append(Ray(start=start, end=(i, j)))
                break

            i, j = ni, nj

    return rays


def project_NE_ray(image, corners):
    return project_ray_diagonal(image, corners, di=-1, dj=+1)


def project_SW_ray(image, corners):
    rays = project_ray_diagonal(image, corners, di=+1, dj=-1)
    canonical_rays = [Ray(start=r.end, end=r.start) for r in rays]
    return canonical_rays

def show_rays(image, rayNE_list,raySW_list):
    plt.imshow(image, cmap='gray')
    for ray in rayNE_list:
        y_values = [ray.start[0], ray.end[0]]
        x_values = [ray.start[1], ray.end[1]]
        plt.plot(x_values, y_values, color='red')
        
    for ray in raySW_list:
        y_values = [ray.start[0], ray.end[0]]
        x_values = [ray.start[1], ray.end[1]]
        plt.plot(x_values, y_values, color='blue')
    plt.title("Projected Rays")
    plt.axis("off")
    plt.show()

def unique_rays(ray_NE_list, ray_SW_list):
    rays = ray_NE_list + ray_SW_list
    return list(set(rays))

def merge_rays_end_to_start(rays):
    starts_at: dict[tuple[float, float], list[float]] = defaultdict(list)
    for i, r in enumerate(rays):
        starts_at[r.start].append(i)

    original_set = set(rays)
    new_rays: set[Ray] = set()

    n = len(rays)

    for i in range(n):
        start = rays[i].start
        first_end = rays[i].end

        stack: list[tuple[tuple[float, float], set[float], bool]] = [(first_end, {i}, False)]

        while stack:
            cur_end, used, extended = stack.pop()

            next_idxs = [j for j in starts_at.get(cur_end, []) if j not in used]

            if not next_idxs:
                if extended and start != cur_end:
                    candidate = Ray(start, cur_end)
                    if candidate not in original_set:
                        new_rays.add(candidate)
                continue

            for j in next_idxs:
                used2 = set(used)
                used2.add(j)
                stack.append((rays[j].end, used2, True))

    rays.extend(sorted(new_rays, key=lambda r: (r.start, r.end)))
    return rays

def find_polygons(binary_image,rays_list, corners_triangle):
    polygons_list = []
    rays_ordered = []
    corner_rays=[]
    
    for r in rays_list:
        mid_point = ((r.start[0] + r.end[0]) / 2, (r.start[1] + r.end[1]) / 2)
        col = mid_point[1] + mid_point[0]
        row = mid_point[1] - mid_point[0]
        rays_ordered.append((col, row, r))
        
    rays_ordered.sort(key=lambda x: (x[0],x[1]))
    rays_ordered = [r[2] for r in rays_ordered]
    
    for ray in rays_ordered:
        print(f"Ray from {ray.start} to {ray.end}")
    
    for c in corners_triangle:
        cx, cy = c
        print(f"Finding ray for corner {c}")
        closest_ray = None
        possible_rays = []
        
        for l in rays_ordered:
            lx1, ly1 = l.start
            lx2, ly2 = l.end
            if (ly1 == cy and  lx2 == cx) or (lx1 == cx and ly2 == cy):
                possible_rays.append(l)
                print(f"Possible ray for corner {c}: from {l.start} to {l.end}")
                
        min_dist= 10000000        
        for l in possible_rays:
            lx1, ly1 = l.start
            lx2, ly2 = l.end

            dist = abs(lx1 - cx) + abs(ly2 - cy) + abs(lx2 - cx) + abs(ly1 - cy)
            if dist<min_dist:
                min_dist=dist
                closest_ray=l
        
        if closest_ray is not None:
            corner_rays.append(closest_ray)
            x = [closest_ray.start[0], closest_ray.end[0], cx]
            y = [closest_ray.start[1], closest_ray.end[1], cy]
            print(f"Closest ray for corner {c}: from {closest_ray.start} to {closest_ray.end}\n")
            polygons_list.append(list(zip(y, x)))
    
    for i in range(0,len(rays_ordered)):
        l = rays_ordered[i]
        lx1, ly1 = l.start
        lx2, ly2 = l.end
        for j in range(i+1,len(rays_ordered)):
            k = rays_ordered[j]
            kx1, ky1 = k.start
            kx2, ky2 = k.end
            if ((lx2 == kx2 and ly1 == ky1) or (lx1 == kx1 and ly2 == ky2) or\
                (lx1 == kx1 and lx2 == kx2) or (ly1 == ky1 and ly2 == ky2)):
                x = [lx1, lx2, kx2, kx1]
                y = [ly1, ly2, ky2, ky1]
                polygons_list.append(list(zip(y, x)))
                break              

    return polygons_list

def show_polygons(image, polygons, rays):
    plt.imshow(image, cmap='gray',origin="upper")
    
    ax = plt.gca()

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for poly in polygons:
        color = (random.random(), random.random(), random.random())
        patch = Polygon(poly, closed=True, fill=True, facecolor=color)
        ax.add_patch(patch)
    
    for ray in rays:
        y_values = [ray.start[0], ray.end[0]]
        x_values = [ray.start[1], ray.end[1]]
        plt.plot(x_values, y_values, color='red')
    
    plt.title("Detected Quadliterals")
    plt.axis("off")
    plt.show()

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.axis("off")
    plt.show()
    
def triangle_matrix(M,h=64, w=64, apex_row=8, apex_col=None, height=40, base_width=40):
    if apex_col is None:
        apex_col = w // 2

    for r in range(height):
        row = apex_row + r
        if row < 0 or row >= h:
            continue

        half = (base_width / 2) * (r / (height - 1)) if height > 1 else 0
        left = int(np.ceil(apex_col - half))
        right = int(np.floor(apex_col + half))

        left = max(left, 0)
        right = min(right, w - 1)
        if left <= right:
            M[row, left:right+1] = 1

    return M

def circle_matrix(M,h=64, w=64, center=None, radius=18):
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center

    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    M[mask] = 1
    return M

def GDM45_decomp(file_name):
    image = Image.open(file_name)
    array_image = np.array(image)
    binary_image = np.array(array_image).astype(bool).astype(int)

    #binary_image = np.full((300,300),0, dtype=int)
    #Square
    #binary_image[64:192, 64:192] = 1
    #Rectanle 1
    #binary_image[64:192, 40:192] = 1
    #Random shape
    """binary_image[64:192, 40:192] = 1
    binary_image[40:70, 40:100] = 1
    binary_image[40:80, 150:180] = 1
    binary_image[170:225, 40:100] = 1
    binary_image[170:225, 150:180] = 0
    """
    #Random shape
    """binary_image[53:178, 28:228] = 1
    binary_image[3:53, 78:128] = 1
    binary_image[3:53, 178:228] = 1
    binary_image[178:278, 78:128] = 1
    binary_image[228:278, 28:178] = 1
    binary_image[103:153, 228:253] = 1
    binary_image[103:128, 253:278] = 1
    """
    """
    binary_image = np.full((100,100),0, dtype=int)
    binary_image = triangle_matrix(binary_image, h=100,w=100, apex_row=30, height=64, base_width=64)
    """
    #binary_image = circle_matrix(binary_image, h=100,w=100, center=(50,50), radius=30)

    obj = (binary_image == 1)
    
    L = label(obj, connectivity=2)
    
    masks = [(L == k).astype(np.uint8) for k in range(1, L.max() + 1)]
    polygons_list = []
    for mask in masks:
        corners, corners_NE, corners_SW, corners_triangle = find_corners(mask)
        ray_NE_list = project_NE_ray(mask, corners_NE)
        ray_SW_list = project_SW_ray(mask, corners_SW)
        unique_rays_list = unique_rays(ray_NE_list, ray_SW_list)
        merged_rays_list = merge_rays_end_to_start(unique_rays_list)
        polygons=find_polygons(mask,merged_rays_list, corners_triangle)

        polygons_list += polygons
        """show_corners(mask, corners)
        show_corners_classes(mask, [corners_NE, corners_SW, corners_triangle]) 
        show_rays(mask, ray_NE_list, ray_SW_list)
        show_rays(mask, unique_rays_list, [])
        show_rays(mask, merged_rays_list, [])
        show_polygons(mask, polygons, [])"""
    show_polygons(binary_image, polygons_list, [])
    print(f"Number of polygons: {len(polygons_list)}")

file_name="TestImages/ellipse_256.tif" #189
#file_name="TestImages/6_circle.tif"
#file_name="TestImages/test00_x.tif"
#file_name="TestImages/simple.tif"
#file_name="TestImages/turbine.tif"



GDM45_decomp(file_name)


