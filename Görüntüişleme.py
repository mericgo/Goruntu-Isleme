import cv2
import numpy as np
import imutils

def detect_shape(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    if len(approx) == 3:
        shape = "üçgen"
    elif len(approx) == 4:
    
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "kare" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "beşgen"
    else:
        shape = "daire"
    
    return shape


image = cv2.imread('foto.jpg')


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


alt_red_1 = np.array([0, 120, 70])
ust_red_1 = np.array([10, 255, 255])
alt_red_2 = np.array([170, 120, 70])
ust_red_2 = np.array([180, 255, 255])


mask1 = cv2.inRange(hsv_image, alt_red_1, ust_red_1)
mask2 = cv2.inRange(hsv_image, alt_red_2, ust_red_2)

mask = cv2.bitwise_or(mask1, mask2)


kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    if cv2.contourArea(contour) > 100:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        area = cv2.contourArea(contour)
        
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        shape = detect_shape(contour)
        
        print(f"Cisim şekli: {shape}")
        print(f"Koordinatlar: ({cX}, {cY})")
        print(f"Alan: {area}")
        print("-" * 30)

cv2.imshow('Detected Red Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

