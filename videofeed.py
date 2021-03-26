import cv2
import numpy as np

#Capture Webcam footage. 0 is for first web cam footage, if some other web cam we would have used 1 or 2 or 3...

# to caputure video feed
cap = cv2.VideoCapture(0)
#A video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output file stream for video
out  = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while True:
    ret, frame = cap.read() # if feed available then ret is true else false. frame will hold the img
    #to flip screen
    '''frame = cv2.flip(frame,0)'''

    cv2.imshow("frame", frame)

    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    # writing to video out stream
    gray = np.stack((gray,)*3,axis=-1)
    out.write(gray)

    # 0xFF takes the keyboard key entry value and we compare it with 'q'. If 'q' is pressed we break out of infinitr loop
    if(cv2.waitKey(1)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


