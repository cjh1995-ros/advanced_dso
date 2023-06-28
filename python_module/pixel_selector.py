from dataclasses import dataclass, field
from pathlib import Path
from image import Image
from copy import deepcopy
from basic_data_structure import Pixel, Point
import numpy as np
import matplotlib.pyplot as plt
import cv2

@dataclass
class PixelSelector:
    w:                          int
    h:                          int
    random_pattern:             np.ndarray = field(init=False)  # post_init 에서 생성됨
    smoothed_threshold:         np.ndarray = field(init=False)  # _create_histogram 에서 생성됨

    # setting values
    setting_grid_size:          tuple = (32, 32)
    setting_potential_value:    int = 3 # <-- 진행하면서 바뀔 수 있음
    setting_grad_threshold:     int = 50
    setting_hist_ratio:         float = 0.5
    setting_grad_down_weight:   float = 0.75
    setting_min_grad_hist_add:  float = 7.

    def __post_init__(self):
        # 0 ~ 255 사이의 값을 가지는 이미지와 동일한 사이즈의 random pattern 생성
        self.random_pattern = np.random.randint(0, 256, size=(self.h, self.w) , dtype=np.uint8)

    def make_maps(self, image: Image, density: float, recursion_left: int, th_factor: float = 2.0):
        n_have, n_want = 0., density
        ideal_pot = self.setting_potential_value # 초기 potential 값.
        # histogram 을 한 번이라도 만들었는지?
        if not image.grad_hist: self._create_histogram(image)
        # select 하기 
        pixel_mask, n_selected = self._select(image)

        n_have = np.sum(n_selected)
        quotia = n_want / n_have

        # n_have 와 n_want 의 비율을 맞춰서 potential 을 조절함. 어떻게 보면 sqrt(quotia) 의 역수를 potential 에 곱하는 것과 유사
        K = n_have * (self.setting_potential_value + 1) ** 2
        ideal_pot = np.sqrt(K / n_want) - 1

        if ideal_pot < 1: ideal_pot = 1
        
        # recursion 해도 되고, 예상보다 더 적게 갖고 있다면, potential 을 낮춰서 다시 샘플링함.
        if recursion_left > 0 and quotia > 1.25 and self.setting_potential_value > 1:
            if ideal_pot > self.setting_potential_value:
                ideal_pot = self.setting_potential_value - 1

            self.setting_potential_value = ideal_pot
            return self.make_maps(image, density, recursion_left - 1, th_factor)

        # recursion 해도 되고, 예상 보다 더 많이 갖고 있다면, potential 을 높여서 다시 샘플링함.
        elif recursion_left > 0 and quotia < 0.25:
            if ideal_pot <= self.setting_potential_value:
                ideal_pot = self.setting_potential_value + 1

            self.setting_potential_value = ideal_pot
            return self.make_maps(image, density, recursion_left - 1, th_factor)

        n_have_sub = n_have
        if quotia < 0.95:
            rn = 0
            TH = 255 * quotia

            for y in range(self.h):
                for x in range(self.w):
                    if pixel_mask[y, x] != 0:
                        if self.random_pattern[y, x] > TH:
                            pixel_mask[y, x] = 0
                            n_have_sub -= 1

        return n_have_sub, pixel_mask


    def _select(self, image: Image, th_factor: float = 2.0):
        angle_list = np.linspace(0, np.pi * 2, 16, endpoint=False)
        random_dir = np.column_stack((np.cos(angle_list), np.sin(angle_list)))

        n4, n3, n2 = 0, 0, 0

        d1 = self.setting_grad_down_weight
        d2 = d1 * d1

        h, w = image.pyramid_shape[0]

        h32, w32 = h // self.setting_grid_size[0], w // self.setting_grid_size[1]

        img_grad0 = image.image_grad[0]# 0번째 level 의 sqrt(grad_x^2 + grad_y^2), grad_x, grad_y
        img_grad1 = image.image_grad[1]# 1번째 level 의 sqrt(grad_x^2 + grad_y^2), grad_x, grad_y
        img_grad2 = image.image_grad[2]# 2번째 level 의 sqrt(grad_x^2 + grad_y^2), grad_x, grad_y

        mask = np.zeros((h, w), dtype=np.uint8)

        self.setting_potential_value = int(self.setting_potential_value)

        for y4 in range(0, h, 4 * self.setting_potential_value):
            for x4 in range(0, w, 4 * self.setting_potential_value):
                # 12x12 patch 를 사용해서 이미지를 나눔
                my3 = min(4 * self.setting_potential_value, h - y4)
                mx3 = min(4 * self.setting_potential_value, w - x4)
                best_idx4, best_val4 = (-1, -1), 0
                dir4 = random_dir[np.random.randint(0, 16)]

                for y3 in range(0, my3, 2 * self.setting_potential_value):
                    for x3 in range(0, mx3, 2 * self.setting_potential_value):
                        # 6x6 patch 를 사용해서 12x12 patch 를 나눔
                        x34 = x3 + x4
                        y34 = y3 + y4
                        my2 = min(2 * self.setting_potential_value, h - y34)
                        mx2 = min(2 * self.setting_potential_value, w - x34)
                        best_idx3, best_val3 = (-1, -1), 0
                        dir3 = random_dir[np.random.randint(0, 16)]

                        for y2 in range(0, my2, self.setting_potential_value):
                            for x2 in range(0, mx2, self.setting_potential_value):
                                # 3x3 patch 를 사용해서 6x6 patch 를 나눔
                                x234 = x2 + x34
                                y234 = y2 + y34
                                
                                my1 = min(self.setting_potential_value, h - y234)
                                mx1 = min(self.setting_potential_value, w - x234)
                                best_idx2, best_val2 = (-1, -1), 0
                                dir2 = random_dir[np.random.randint(0, 16)]

                                for y1 in range(0, my1):
                                    for x1 in range(0, mx1):
                                        # 이미지 상 idx
                                        xf = x1 + x234 
                                        yf = y1 + y234
                                        
                                        # 이미지를 벗어나는지 확인
                                        assert xf < w and yf < h

                                        # 해당 픽셀이 갖는 threhold 값 찾기
                                        pix_threshold0 = self.smoothed_threshold[yf // h32, xf // w32]
                                        pix_threshold1 = pix_threshold0 * d1
                                        pix_threshold2 = pix_threshold1 * d2

                                        ag0 = img_grad0[yf, xf][0]
                                        if ag0 > pix_threshold0 * th_factor:
                                            ag0d = img_grad0[yf, xf][1:] # (grad_x, grad_y)
                                            dir_norm = ag0d @ dir2

                                            if dir_norm > best_val2:
                                                best_val2 = dir_norm
                                                best_idx2 = (yf, xf)
                                                best_idx3 = (-2, -2)
                                                best_idx4 = (-2, -2)
                                        
                                        if best_idx3 == (-2, -2):
                                            continue

                                        ag1 = img_grad1[int(yf // 2 + 0.25), int(xf // 2 + 0.25)][0]
                                        if ag1 > pix_threshold1 * th_factor:
                                            ag1d = img_grad1[int(yf // 2 + 0.25), int(xf // 2 + 0.25)][1:]
                                            dir_norm = ag1d @ dir3

                                            if dir_norm > best_val3:
                                                best_val3 = dir_norm
                                                best_idx3 = (yf, xf)
                                                best_idx4 = (-2, -2)
                                        
                                        if best_idx4 == (-2, -2):
                                            continue

                                        ag2 = img_grad2[int(yf // 4 + 0.125), int(xf // 4 + 0.125)][0]
                                        if ag2 > pix_threshold2 * th_factor:
                                            ag2d = img_grad2[int(yf // 4 + 0.125), int(xf // 4 + 0.125)][1:]
                                            dir_norm = ag2d @ dir4

                                            if dir_norm > best_val4:
                                                best_val4 = dir_norm
                                                best_idx4 = (yf, xf)

                                if best_idx2[0] > 0 and best_idx2[1] > 0:
                                    mask[best_idx2[0], best_idx2[1]] = 1
                                    n2 += 1

                        if best_idx3[0] > 0 and best_idx3[1] > 0:
                            mask[best_idx3[0], best_idx3[1]] = 2
                            n3 += 1
                
                if best_idx4[0] > 0 and best_idx4[1] > 0:
                    mask[best_idx4[0], best_idx4[1]] = 4
                    n4 += 1

        return mask, np.array([n2, n3, n4], dtype=np.int32)



    def _create_histogram(self, image: Image) -> None:
        """
            이미지를 32x32(self.setting_grid_size)로 나누고, 각각의 그리드에서
            픽셀의 gradient를 구해서 histogram을 만든다. 이후 만들어진 histogram을
            smoothing 하면서 pixel threshold를 만든다.
        """
        image.grad_hist = True
        h, w = image.pyramid_shape[0] 

        pixel_thresholds = np.full(self.setting_grid_size, 0, dtype=np.float32)
        grid_y, grid_x = self.setting_grid_size
        
        # original gradient 이미지에서 해당 픽셀의 sqrt(idx^2 + idy^2) 를 갖고 옴.
        original_grad_img = image.image_grad[0][..., 0]
        
        # 이미지를 32x32 로 나눈 후 각각의 그리드에서 픽셀의 threshold를 구한다.
        # 이 경우 히스토그램의 idx 값이 threshold 가 된다.
        inc_x, inc_y = int(w / grid_x), int(h / grid_y)

        for grid_j in range(0, h, inc_y):
            for grid_i in range(0, w, inc_x):
                # 왜 마지막 까지 가는거냐....? 쨋든 index 에서 제외해줘야 하니 해당 idx는 continue 로 탈출한다.
                histogram = np.zeros(self.setting_grad_threshold + 1, dtype=np.int32)
                for j in range(inc_y):
                    for i in range(inc_x):                         
                         grad = original_grad_img[int(grid_j + j), int(grid_i + i)]
                         if grad > self.setting_grad_threshold: grad = self.setting_grad_threshold
                         histogram[int(grad)] += 1
                
                # pixel threshold is made from histogram
                pixel_threshold = self._create_histogram_threshold(histogram) + self.setting_min_grad_hist_add
                
                pixel_thresholds[grid_j // inc_y, grid_i // inc_x] = pixel_threshold

        smoothed_thresholds = np.full(self.setting_grid_size, 0, dtype=np.float32)

        for j in range(grid_y):
            for i in range(grid_x):
                sum, num = 0, 0
                if i > 0:
                    if j > 0:
                        num += 1
                        sum += pixel_thresholds[j - 1, i - 1]
                    if j < grid_y - 1:
                        num += 1
                        sum += pixel_thresholds[j + 1, i - 1]

                    num += 1
                    sum += pixel_thresholds[j, i - 1]
                
                if i < grid_x - 1:
                    if j > 0:
                        num += 1
                        sum += pixel_thresholds[j - 1, i + 1]
                    if j < grid_y - 1:
                        num += 1
                        sum += pixel_thresholds[j + 1, i + 1]

                    num += 1
                    sum += pixel_thresholds[j, i + 1]
                
                if j > 0:
                    num += 1
                    sum += pixel_thresholds[j - 1, i]

                if j < grid_y - 1:
                    num += 1
                    sum += pixel_thresholds[j + 1, i]
                
                num += 1
                sum += pixel_thresholds[j, i]

                smoothed_thresholds[j, i] = sum / num

        self.smoothed_threshold = smoothed_thresholds

    def _create_histogram_threshold(self, histogram: list) -> float:
        """
        create threshold for histogram
        """
        sum = 0
        threshold = np.sum(histogram) * self.setting_hist_ratio
        for i in range(len(histogram)):
            sum += histogram[i]
            if sum > threshold: return i
        return 90


def test_create_histogram(img_path: Path):
    img = Image(img_path, 4)

    h, w = img.pyramid_shape[0]

    selector = PixelSelector(h=h, w=w)

    smoothed_thresholds = selector._create_histogram(img)


def test_select(img_path: Path):
    img = Image(img_path, 4)

    h, w = img.pyramid_shape[0]

    selector = PixelSelector(h=h, w=w)

    selector._create_histogram(img)
    mask, selected = selector._select(img)

    print(selected)

    # masking image with mask, mask value is 1, 2, 4. so if 1, it means we use 1st gradient image.
    # if 2, we use 2nd gradient image. if 4, we use 3rd gradient image.
    new_img = deepcopy(img.image_orig)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB) # convert to RGB

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 1:
                new_img[y, x] = [255, 0, 0]
            elif mask[y, x] == 2:
                new_img[y, x] = [0, 255, 0]
            elif mask[y, x] == 4:
                new_img[y, x] = [0, 0, 255]

    plt.imshow(new_img)
    plt.title('selected pixels')
    plt.show()


def test_make_maps(img_path: Path):
    img = Image(img_path, 4)

    h, w = img.pyramid_shape[0]

    selector = PixelSelector(h=h, w=w)

    densities = [0.03, 0.05, 0.15, 0.5, 1.0]
    recursion = 1

    # n_point, masked_map = selector.make_maps(image=img, density=densities[0] * w * h, recursion_left=recursion)
    n_point, masked_map = selector.make_maps(image=img, density=1500, recursion_left=recursion)
    
    print(n_point) # 40544 -> 1515

    new_img = deepcopy(img.image_orig)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB) # convert to RGB

    # show masked threshold image
    plt.imshow(selector.smoothed_threshold, cmap='jet')
    plt.title('masked threshold image')
    plt.colorbar()
    plt.show()

    # show selected pixels
    plt.imshow(new_img)
    for y in range(masked_map.shape[0]):
        for x in range(masked_map.shape[1]):
            if masked_map[y, x] == 1:
                plt.scatter(x, y, c='r', s=3)
                # new_img[y, x] = [255, 0, 0]
            elif masked_map[y, x] == 2:
                plt.scatter(x, y, c='g', s=3)
                # new_img[y, x] = [0, 255, 0]
            elif masked_map[y, x] == 4:
                plt.scatter(x, y, c='b', s=3)
                # new_img[y, x] = [0, 0, 255]

    plt.title('selected pixels')
    plt.show()




if __name__ == '__main__':
    import zipfile

    zip_name = 'my_archive.zip'

    with zipfile.ZipFile(zip_name, 'r') as zipf:
        file_list = zipf.namelist()

        for file_name in file_list:
            if file_name.endswith('.jpg'):
                # 선택된 파일 읽기
                with zipf.open(file_name, 'r') as file:
                    file_contents = file.read()
                    # 읽은 파일 내용을 활용하여 원하는 작업 수행
                    # 예시로 파일 이름과 내용 출력
                    print(f'File: {file_name}')
                    print('Contents:', file_contents)

    print('압축 파일 내의 필요한 파일들을 읽었습니다.')

    img_path = Path.cwd() / 'data' / 'sequence_09' / 'images' / '00000.jpg'
    # test_create_histogram(img_path)
    # test_select(img_path)
    test_make_maps(img_path)