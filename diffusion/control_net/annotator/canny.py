import cv2


class CannyDetector:
    def __call__(self, img):
        low_threshold = 100
        high_threshold = 200

        img = img[:, :, ::-1]  # BGR -> RGB
        detected_map = cv2.Canny(img, low_threshold, high_threshold)

        return detected_map

    def map2img(self, detected_map):
        return 255 - detected_map
