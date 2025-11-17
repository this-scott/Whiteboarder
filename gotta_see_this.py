import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

# -----------------------------
# Intersect two Hough lines
# -----------------------------
def intersect(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ], dtype=float)

    b = np.array([rho1, rho2], dtype=float)

    x0, y0 = np.linalg.solve(A, b)
    return [int(x0.item()), int(y0.item())]


# -----------------------------
# Order rectangle points
# -----------------------------
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.int32)


# -----------------------------
# Main rectangle detection
# -----------------------------
def detect_rectangle(img):
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 70, 150)

    # Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None or len(lines) < 4:
        return None, None

    # Normalize angles to [0,180)
    thetas = np.array([l[0][1] for l in lines])
    thetas_deg = (np.rad2deg(thetas) + 180) % 180

    # Cluster into 2 groups: vertical-like and horizontal-like
    kmeans = KMeans(n_clusters=2, n_init=10).fit(thetas_deg.reshape(-1, 1))

    cluster1 = []
    cluster2 = []

    for idx, line in enumerate(lines):
        if kmeans.labels_[idx] == 0:
            cluster1.append(line[0])
        else:
            cluster2.append(line[0])

    # Determine which cluster is vertical vs horizontal by average angle
    mean1 = np.mean([l[1] for l in cluster1])
    mean2 = np.mean([l[1] for l in cluster2])

    if abs(np.rad2deg(mean1) - 90) < abs(np.rad2deg(mean2) - 90):
        horizontals = cluster1
        verticals = cluster2
    else:
        horizontals = cluster2
        verticals = cluster1

    # Must have at least 2 of each
    if len(horizontals) < 2 or len(verticals) < 2:
        return None, None

    # Choose extreme lines (min/max rho) in each group → sides of rectangle
    horizontals = sorted(horizontals, key=lambda x: x[0])[:2] + sorted(horizontals, key=lambda x: x[0])[-2:]
    verticals   = sorted(verticals,   key=lambda x: x[0])[:2] + sorted(verticals,   key=lambda x: x[0])[-2:]

    # Pick best 2 unique lines
    horizontals = sorted(horizontals, key=lambda x: x[0])
    verticals   = sorted(verticals,   key=lambda x: x[0])

    top, bottom = horizontals[0], horizontals[-1]
    left, right = verticals[0], verticals[-1]

    # Compute rectangle corners by intersections
    pts = [
        intersect(top, left),
        intersect(top, right),
        intersect(bottom, left),
        intersect(bottom, right)
    ]

    # Order points properly
    pts = order_points(pts)

    # Draw on output
    out = img.copy()
    cv2.polylines(out, [pts], True, (0, 255, 0), 3)

    return pts, out
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("couldn't grab frame")
            break
        pts, img = detect_rectangle(frame)
        if img is None:
            cv2.imshow("Capture", frame)
            continue
        cv2.imshow("Capture", img)



        #exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            # print(contours)
            # M, nsize = rectify(contours[0]) # This doesn't work. This is considered a decomposition problem and trying to make a destination matrix isn't a solution.
            # warped = cv2.warpPerspective(frame, M, nsize)
            cv2.imwrite('test.jpg',img)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()