import time
import cv2
import numpy as np
import math

def intersect(line1, line2):
    """
    Compute intersection point of 2 lines in (rho, theta) form.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [math.cos(theta1), math.sin(theta1)],
        [math.cos(theta2), math.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    solution = np.linalg.solve(A, b)
    x0 = solution[0].item()
    y0 = solution[1].item()
    return [int(x0), int(y0)]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 25, 150)

    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None:
        continue
    # Separate lines into ~horizontal and ~vertical
    horizontals = []
    verticals = []
    
    for l in lines:
        rho, theta = l[0]
        deg = theta * 180 / np.pi
        if abs(deg - 0) < 20 or abs(deg - 180) < 20:
            verticals.append((rho, theta))
        elif abs(deg - 90) < 20:
            horizontals.append((rho, theta))


    if len(horizontals) < 2 or len(verticals) < 2:
        continue

    horizontals = sorted(horizontals, key=lambda x: x[0])
    verticals = sorted(verticals, key=lambda x: x[0])

    top, bottom = horizontals[0], horizontals[-1]
    left, right = verticals[0], verticals[-1]

    pts = [
        intersect(top, left),
        intersect(top, right),
        intersect(bottom, left),
        intersect(bottom, right)
    ]
    pts = np.array(pts, dtype=np.int32)


    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

    # # HOUGH LINESP ATTEMPT
    # lines = cv2.HoughLinesP(edges,5, np.pi/90,120, minLineLength=50,maxLineGap=5)

    # if lines is not None:
    #     for line in lines:
    #         print(line)
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # CONTOURS ATTEMPT
    # for contour in contours:
    #     peri = cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    #     if len(approx) == 4:
    #         approxes = approx
    #         points = approxes.reshape(-1, 2)

    #         cv2.drawContours(frame, [approxes], -1, (0, 255, 0), 4)

    #         cv2.drawContours(frame, approxes, -1, (0,0,255), 2)
    #         for i, line in enumerate(approxes):
    #             print(f"{i} {line[0][0]}")
    #             print(f"{i} {line[0][1]}")
    #             # cv2.circle(frame, line[0][0], line[0][1], (255,0,0), 3)
    # else:
    #     print("No items detected.")
    
    # edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 

    # result = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)

    cv2.imshow("Capture", frame)
    cv2.imshow("Contours", edges)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()