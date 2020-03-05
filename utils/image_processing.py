import cv2


def transform_img(image):
    """
    First image is convert to grey scale -> binary image -> applies opening morphological operations.
    @param image: np.array - Image to be processed. 
    @return np.array
    """
    kernel_size = 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert grayscale
    transformed_img = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # convert to binary using adaptive threshold
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    transformed_img = cv2.morphologyEx(
        transformed_img, cv2.MORPH_OPEN, kernel)  # apply opening morphological operation.
    return transformed_img
