import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    lines = cv2.HoughLinesP(edges,1, np.pi/180,125,minLineLength=0,maxLineGap=0)
    
    # for contour in contours:
    #     peri = cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # if len(approx) == 4:
        #     approxes = approx
        #     points = approxes.reshape(-1, 2)

            # cv2.drawContours(frame, [approxes], -1, (0, 255, 0), 4)

            # cv2.drawContours(frame, approxes, -1, (0,0,255), 2)
            # for i, line in enumerate(approxes):
            #     print(f"{i} {line[0][0]}")
            #     print(f"{i} {line[0][1]}")
            #     # cv2.circle(frame, line[0][0], line[0][1], (255,0,0), 3)
    else:
        print("No items detected.")
    
    # edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 

    # result = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    cv2.imwrite('test.png', frame)
    
cap.release()
cv2.destroyAllWindows()