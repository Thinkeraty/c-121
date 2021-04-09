import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'Xvid')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#starting webcam
capture = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

#capturing the background for 60 frames.
for i in range(60):
    ret, bg = capture.read()

#flipping the background
bg = np.flip(bg, axis=1)

#reading the captured frame until the camera is opened
while capture.isOpened():
    ret, img = capture.read()
    if not ret: 
        break
    
    #flipping the image for consistency
    img = np.flip(img, axis=1)

    #converting the color from bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #generating mask to detect the red color
    lower_black = np.array([104, 153, 70])
    upper_black = np.array([30, 30, 0])
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)

    lower_black = np.array([170, 120, 70])
    upper_black = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = mask_1 + mask_2

    #morphologyEX(src, dst, op, kernel)
    #src= input image
    #dst = output image
    #op = type of the morphological operasion
    #kernel = matrix

    #opening and expanding the image where mask_1 color has been detected on the red cloth
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    #selecting only the part that does not have mask_1 and saving it in mask_2
    mask_2 = cv2.bitwise_not(mask_1)
    
    #keeping only the part of the images without the red color
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    #keeping only the part of the images with the red color
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    #generating the final output by mergin res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)
    cv2.imshow('magic', final_output)
    cv2.waitKey(1)

capture.release()
# out.release()
cv2.destroyAllWindows()