from pathlib import Path
from dataclasses import dataclass, field
from util import get_interpolated_value
from image import Image
from pixel_selector import PixelSelector
import numpy as np

PASCAL_VOC_SEGMENT_LABEL = {
    1: 'Aeroplane',
    2: 'Bicycle',
    3: 'Bird',
    4: 'Boat',
    5: 'Bottle',
    6: 'Bus',
    7: 'Car',
    8: 'Dog',
    9: 'Chair',
    10: 'Cow',
    11: 'Diningtable',
    12: 'Cat',
    13: 'Horse',
    14: 'Motorbike',
    15: 'Person',
    16: 'Pottedplant',
    17: 'Sheep',
    18: 'Sofa',
    19: 'Train',
    20: 'Tvmonitor',
}

DEL_LIST = [9, 11, 16, 18, 20]

PASCAL_VOC_SEGMENT_LABEL = {k: v for k, v in PASCAL_VOC_SEGMENT_LABEL.items() if k not in DEL_LIST}

PATTERN = [[0, -2], [-1, -1], [1, -1], [-2, 0], [0, 0], [2, 0], [-1, 1], [0, 2]]

PATTERN_PADDING = 2


@dataclass
class ImmaturePoint:
    ui: int
    vi: int
    
    host_image: Image
    my_type: int
    idepth_gt: float            = field(init=False)
    idepth_min: float           = field(init=False)
    idepth_max: float           = field(init=False)
    grad_h: float               = field(init=False)
    is_masked: bool             = field(init=False)
    # if it's energy is high, then it's not a good pixel to be a seed.
    energy_th: float            = field(init=False)
    quality: float              = field(init=False)
    colors: np.ndarray          = field(init=False)
    weights: np.ndarray         = field(init=False)

    # setting from dso.
    setting_outlier_th_sum_component: float = 1.0
    setting_outlier_th: float               = 12 * 12
    setting_overall_energy_th_weight: float = 1.0

    # setting from me >.<
    setting_mask_score: float   = 0.95

    def __post_init__(self):
        self.grad_h = 0.0
        self.colors = np.zeros(len(PATTERN), dtype=np.float32)
        self.weights = np.zeros(len(PATTERN), dtype=np.float32)

        for i, dxy in enumerate(PATTERN):
            dx, dy = dxy
            if self.host_image.is_valid(self.ui + dx, self.vi + dy):
                ptc = get_interpolated_value(self.ui + dx, self.vi + dy, self.host_image.image_gray)

                self.colors[i] = ptc[0]
                if self.colors is None:
                    self.energy_th = None
                    return
                
                self.grad_h += ptc[1:] @ ptc[1:]
                self.weights[i] = np.sqrt(self.setting_outlier_th_sum_component / (self.setting_outlier_th_sum_component + np.linalg.norm(ptc[1:])))

        self.energy_th = len(PATTERN) * self.setting_outlier_th * (self.setting_overall_energy_th_weight ** 2)

        self.idepth_gt = 0.0
        self.quality = 10000
        self.idepth_min = 0.0
        self.idepth_max = None

@dataclass
class MaskedRegion:
    mask_idx: int           = field(init=False)
    pixel_list: list        = field(init=False)


def test_init_immature_points(img_path: Path):
    img = Image(img_path, 4)

    h, w = img.pyramid_shape[0]

    selector = PixelSelector(h=h, w=w)

    # densities = [0.03, 0.05, 0.15, 0.5, 1.0]
    recursion = 1

    # n_point, masked_map = selector.make_maps(image=img, density=densities[0] * w * h, recursion_left=recursion)
    n_point, masked_map = selector.make_maps(image=img, density=1500, recursion_left=recursion)

    ips = []

    for v in range(1 + PATTERN_PADDING, masked_map.shape[0] - PATTERN_PADDING - 2):
        for u in range(1 + PATTERN_PADDING, masked_map.shape[1] - PATTERN_PADDING - 2):
            if masked_map[v, u] == 0:
                continue
            
            ip = ImmaturePoint(u, v, img, masked_map[v, u])
            if ip.energy_th is None:
                continue
            ips.append(ip)

    print(len(ips), n_point)

if __name__ == '__main__':
    img_path = Path.cwd() / 'data' / 'data_odometry_color' / 'dataset' / 'sequences' / '00' / 'image_2' / '000000.png'
    test_init_immature_points(img_path)