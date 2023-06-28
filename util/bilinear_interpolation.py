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