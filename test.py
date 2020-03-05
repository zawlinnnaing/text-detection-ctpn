import cv2

IMAGE = "data/cropped-colored-images/photp-crop-erode/photp-crop-erode_2.jpg"

image = cv2.imread(IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_img.jpg", image)
# edge_image = cv2.Canny(image, 60, 70)
edge_image = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# edge_image = cv2.erode(edge_image, kernel, iterations=2)
# edge_image = cv2.dilate(edge_image, kernel, iterations=2)

edge_image = cv2.morphologyEx(
    edge_image, cv2.MORPH_OPEN, kernel)
# edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel)

# edge_image = cv2.cvtColor(edge_image,cv2.COLOR_B)

cv2.imwrite("edge_img.jpg", edge_image)
