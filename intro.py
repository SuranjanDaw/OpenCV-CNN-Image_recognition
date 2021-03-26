import cv2
import numpy as np  
import matplotlib.pyplot as plt 

img = cv2.imread("sample/m3.jpg", 0) 
'''
0 = greyscale
1 = color
-1 = unchanged
'''


# Showing image using OpenCV
'''
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
'''
#showing image using Matplotlib
plt.imshow(img, cmap='gray', interpolation='bicubic')
# ploting lines using plt
'''
plt.plot([50,500],[100,1000],'c', linewidth=5),
'''
plt.show()


# write a image 

cv2.imwrite("sample/grayscale.png", img)
