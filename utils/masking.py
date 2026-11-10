import cv2

def _ellipse(n):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

def create_bottle_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Masking the bottle region
    mask = (
        cv2.inRange(hsv, (0, 0, 110), (180, 80, 255))           
        | cv2.inRange(hsv, (0, 0, 125), (180, 255, 255))    
        | cv2.inRange(hsv, (45, 150, 100), (66, 255, 255))   
        | cv2.inRange(hsv, (158, 35, 70), (180, 100, 255))   
        | cv2.inRange(hsv, (140, 110, 70), (175, 255, 255))  
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(15), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(51)) 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask[:] = 0
        cv2.drawContours(mask, [largest], -1, 255, -1)

    mask = cv2.dilate(mask, _ellipse(9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(15))
    return mask > 0
