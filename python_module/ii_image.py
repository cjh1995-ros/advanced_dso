import numpy as np
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def illumination_invariant_imaging(image: np.ndarray, alpha: float, format: str = 'cv') -> np.ndarray:
    """
    Performs illumination invariant imaging on the given image.
    
    Args:
        image (np.ndarray): CV Image to be processed.(BGR)
        alpha (float): Parameter for illumination invariant imaging.
    
    Returns:
        np.ndarray: Processed image.
    """
    if format == 'cv':
        new_image = np.log(image[..., 1]) + (1 - alpha) * np.log(image[..., 2]) + alpha * np.log(image[..., 0])
        return (255 * new_image).astype('uint8')
    elif format == 'plt':
        normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        new_image = 0.5 + np.log(normalized_image[..., 1]) + alpha * np.log(normalized_image[..., 2]) + (1 - alpha) * np.log(normalized_image[..., 0])
        return new_image.clip(-10, 10)
        

def peaks2alpha(lambda1: float, lambda2: float, lambda3: float) -> float:
    """
    Calculates alpha value from the given eigenvalues.
    
    Args:
        lambda1 (float): Peak spectral response of R channel.
        lambda2 (float): Peak spectral response of G channel.
        lambda3 (float): Peak spectral response of B channel.
    
    Returns:
        float: alpha value.
    """
    return (lambda1 * (lambda3 - lambda2)) / (lambda2 * (lambda3 - lambda1))


def show(image, title, format='cv'):
    # Check channel of image
    assert image.ndim == 2, "Image should be grayscale"

    if format == 'cv':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif format == 'plt':
        plt.imshow(image)
        plt.title(title)
        plt.show()


def main(image_path: str, format: str = 'cv'):
    image = cv2.imread(image_path)
    if format == 'plt': image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # B: 460
    # G: 540
    # R: 610
    # Calculate alpha value
    lambda1 = 460
    lambda2 = 540
    lambda3 = 610
    alpha = peaks2alpha(lambda1, lambda2, lambda3)
    
    # Perform illumination invariant imaging
    new_image = illumination_invariant_imaging(image, alpha, format)

    show(new_image, "image", format)

if __name__ == "__main__":
    parser = ArgumentParser(description='Illumination Invariant Imaging')
    parser.add_argument('image', type=str, help='Path to the image.')
    parser.add_argument('--format', type=str, default='cv', help='Format of the image. (cv or plt)')

    args = parser.parse_args()
    
    main(args.image, args.format)