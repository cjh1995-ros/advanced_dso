import cv2
import argparse
import numpy as np

def adjust_image_intensity(image, delta):
    return (image / delta).clip(0, 255).astype('uint8')

def main(image1_path, image2_path):
    image1_original = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    delta = 1.0
    while True:
        image1_modified = adjust_image_intensity(image1_original, delta)
        
        # 이미지를 수직으로 쌓기
        stacked_images = np.hstack((image1_original, image1_modified, image2))
        
        cv2.imshow('Images', stacked_images)
        
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
        elif key == ord('w'):  # UP arrow
            delta += 0.3
            print(f"delta: {delta}")
        elif key == ord('x'):  # DOWN arrow
            delta -= 0.3
            print(f"delta: {delta}")


    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust image intensity using arrow keys.')
    parser.add_argument('image1', type=str, help='Path to the first image.')
    parser.add_argument('image2', type=str, help='Path to the second image for comparison.')
    
    args = parser.parse_args()
    
    main(args.image1, args.image2)
