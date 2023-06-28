import numpy as np

def divide_image_into_grid(image, grid_size):
    # 이미지의 크기
    image_height, image_width = image.shape[:2]

    # 그리드 크기 계산 (반올림)
    grid_height = int(round(image_height / grid_size))
    grid_width = int(round(image_width / grid_size))

    n_over_h, n_over_w = 0, 0

    # 그리드로 이미지 분할
    grids = []
    for v in range(grid_height):
        for u in range(grid_width):
            # 그리드의 좌상단 좌표
            start_v = int(round(v * grid_size))
            start_u = int(round(u * grid_size))

            # 그리드의 우하단 좌표
            end_v = int(round((v + 1) * grid_size))
            end_u = int(round((u + 1) * grid_size))

            if end_v > image_height:
                breakpoint()
                end_v = image_height
                n_over_h += 1


            if end_u > image_width:
                end_u = image_width
                n_over_w += 1

            # 그리드 추출
            grid = image[start_v:end_v, start_u:end_u]
            grids.append(grid)

    return grids, n_over_h, n_over_w

# 예시 이미지 생성 (1241x376)
image = np.random.randint(0, 255, (376, 1241))

# 이미지를 32x32 그리드로 나누기
grid_size = 32
grids, n_over_h, n_over_w = divide_image_into_grid(image, grid_size)

# 그리드 확인
print(len(grids))
print(n_over_h)
print(n_over_w)