import numpy as np

def bilinear_interpolation(x: float, y: float, image: np.ndarray):
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1

    dx = x - x1
    dy = y - y1

    q11 = image[y1, x1]
    q12 = image[y2, x1]
    q21 = image[y1, x2]
    q22 = image[y2, x2]

    val = (1 / (x2 - x1) * (y2 - y1)) * np.array([[x2 - x, x - x1]], dtype=np.float32)
    val = val @ np.array([[q11, q12], [q21, q22]], dtype=np.float32) @ np.array([[y2 - y], [y - y1]], dtype=np.float32)
    return val[0][0]

def get_interpolated_value(x: int, y: int, image: np.ndarray):
    """
    return color + dx color + dy color
    """
    # check if it's valid
    if x < 0 or x >= image.shape[1] - 1 or y < 0 or y >= image.shape[0] - 1:
        return None

    tl = image[y, x]
    tr = image[y, x + 1]
    bl = image[y + 1, x]
    br = image[y + 1, x + 1]

    dx = x - int(x)
    dy = y - int(y)

    top_color = (1 - dx) * tl + dx * tr
    bottom_color = (1 - dx) * bl + dx * br
    left_color = (1 - dy) * tl + dy * bl
    right_color = (1 - dy) * tr + dy * br

    return np.array([dx * right_color + (1 - dx) * left_color, 
                     right_color - left_color,
                     bottom_color - top_color], dtype=np.float32)

def debug_bilinear_interpolation():
    # gen random values for testing
    x = np.random.uniform(0, 10)
    y = np.random.uniform(0, 10)
    image = np.random.uniform(0, 10, size=(10, 10))

    # bilinear interpolation
    val = bilinear_interpolation(x, y, image)

    # check
    print(f'x: {x}, y: {y}')
    print(f'image: {image[int(y), int(x)]}')
    print(f'val: {val}')

if __name__ == '__main__':
    debug_bilinear_interpolation()