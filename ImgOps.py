import cv2
import numpy as np

img = cv2.imread('sample/m3.jpg', cv2.IMREAD_COLOR)

px = img[55,100] # Each pixel is a array of size 3. Indicating the BGR values

px = [5,10,252] # changing the pixel value to specific BGR

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# changing a set of pixels

img[50:500, 50:500] = [255,255,255]
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("ne.jpg",img)


# copy a part of image

subImg = img[1500:2000, 1500:1944]
cv2.imshow("subimage", subImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#pasteing a subimage on a image P.S.: its important to match the dimentions of subImage with origanl image's replaced portion 

img[100:600, 100:544] = subImg
cv2.imshow("paste", img)
cv2.waitKey(0)
cv2.destroyAllWindows()