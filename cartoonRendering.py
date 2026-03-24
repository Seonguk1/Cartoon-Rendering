import cv2
import numpy as np

img = cv2.imread('images/image1.jpg')
# img = cv2.imread('images/image2.webp')

# 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Median Blur (노이즈 제거)
gray = cv2.medianBlur(gray, 5) 

# Adaptive Thresholding (스케치 선 마스크 생성)
edges = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=5, 
    C=3
)

# 십자가 커널
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

# 침식 적용
edges = cv2.erode(edges, kernel, iterations=1)

# Bilateral Filter (색상 평탄화)
color = img.copy()
for _ in range(7):
    color = cv2.bilateralFilter(color, 9, 250, 250)

# 합성
cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imshow("Original", img)
cv2.imshow("Cartoon Rendering", cartoon)

cv2.waitKey(0)
cv2.destroyAllWindows()