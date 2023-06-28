from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Image:
    file_path:          Path
    lvl_pyramid:        int
    debug_cfg:          dict = field(default_factory=dict)
    image_orig:         np.ndarray = field(init=False) # color image
    image_gray:         np.ndarray = field(init=False) # gray image
    image_pyramid:      list = field(init=False)
    pyramid_shape:      list = field(init=False)
    image_grad:         list = field(init=False)

    grad_hist:          bool = False

    def __post_init__(self):
        self.image_orig = cv2.imread(str(self.file_path))
        self.image_gray = cv2.cvtColor(self.image_orig, cv2.COLOR_BGR2GRAY)
        
        self.image_pyramid  = self._create_pyramid()
        self.pyramid_shape  = [image.shape for image in self.image_pyramid]
        self.image_grad = self._create_gradient()
        
    def _create_pyramid(self):
        pyramid = [self.image_gray]
        h, w = self.image_gray.shape
        for i in range(self.lvl_pyramid):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid

    def _create_gradient(self):
        """
        create gradient image pyramid from original image pyramid
        """
        # Check if the image is 2D (grayscale)
        if len(self.image_gray.shape) > 2:
            raise ValueError("The input self.orig_file should be grayscale (2D array).")

        # Create arrays to store the gradients
        grad_i_x = np.zeros_like(self.image_gray, dtype=np.float32)
        grad_i_y = np.zeros_like(self.image_gray, dtype=np.float32)

        # Calculate the gradients of original image
        for v in range(1, self.image_gray.shape[0]-1):
            for u in range(1, self.image_gray.shape[1]-1):
                grad_i_x[v, u] = self.image_gray[v, u+1].astype(np.float32) - self.image_gray[v, u-1].astype(np.float32)
                grad_i_y[v, u] = self.image_gray[v+1, u].astype(np.float32) - self.image_gray[v-1, u].astype(np.float32)

        grad_img = np.sqrt(grad_i_x**2 + grad_i_y**2)

        orig_grad = np.stack([grad_img, grad_i_x, grad_i_y], axis=2)

        grad_img_list = [orig_grad]

        for i in range(self.lvl_pyramid):
            if i != 0:
                grad_img = grad_img_list[i-1][..., 0]
                grad_img_x = grad_img_list[i-1][..., 1]
                grad_img_y = grad_img_list[i-1][..., 2]
                
                new_grad_img = np.full_like(grad_img, 0, dtype=np.float32)
                new_grad_img_x = np.full_like(grad_img_x, 0, dtype=np.float32)
                new_grad_img_y = np.full_like(grad_img_y, 0, dtype=np.float32)
                
                v_, u_ = grad_img.shape
                v_ = v_ // 2
                u_ = u_ // 2


                for v in range(v_):
                    for u in range(u_):
                        new_grad_img[v, u] = 0.25 * (grad_img[2*v, 2*u] +\
                                                     grad_img[2*v+1, 2*u] +\
                                                     grad_img[2*v, 2*u+1] +\
                                                     grad_img[2*v+1, 2*u+1])
                        new_grad_img_x[v, u] = 0.25 * (grad_img_x[2*v, 2*u+1] +\
                                                        grad_img_x[2*v+1, 2*u+1] +\
                                                        grad_img_x[2*v, 2*u] +\
                                                        grad_img_x[2*v+1, 2*u])
                        new_grad_img_y[v, u] = 0.25 * (grad_img_y[2*v, 2*u+1] +\
                                                        grad_img_y[2*v+1, 2*u+1] +\
                                                        grad_img_y[2*v, 2*u] +\
                                                        grad_img_y[2*v+1, 2*u])
                
                new_grad_img = np.stack([new_grad_img, new_grad_img_x, new_grad_img_y], axis=2)
                grad_img_list.append(new_grad_img)
        
        return grad_img_list

    def convert(self, color:str = 'rgb'):
        if color == 'rgb':
            return cv2.cvtColor(self.image_orig, cv2.COLOR_BGR2RGB)
        elif color == 'gray':
            return self.image_gray
        elif color == 'bgr':
            return self.image_orig

    def is_valid(self, u, v) -> bool:
        if u < 0 or u >= self.image_gray.shape[1] \
            or v < 0 or v >= self.image_gray.shape[0]:
            return False
        return True

    def _debug_pyramid(self):
        # show all information of image pyramid
        for i, img in enumerate(self.image_pyramid):
            print(f'Level {i}: {img.shape}')
            plt.imshow(img, cmap='gray')
            plt.title(f'Level {i}')
            plt.axis('off')
            plt.show()

    def _debug_gradient(self):
        # show all information of gradient
        for i, img in enumerate(self.image_grad):
            plt.imshow(img[..., 1], cmap='gray')
            plt.title(f'Gradient: Level {i}')
            plt.axis('off')
            plt.show()


    def debug(self):
        if self.debug_cfg['pyramid']: self._debug_pyramid()
        if self.debug_cfg['gradient']: self._debug_gradient()

if __name__ == '__main__':
    img_path = Path.cwd() / 'data' / 'sequence_09' / 'images' / '00000.jpg'
    debug_cfg = {
        'pyramid': False,
        'gradient': True
    }

    img = Image(img_path, 4, debug_cfg=debug_cfg)
    img.debug()