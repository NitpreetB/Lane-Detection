import cv2 
import numpy as np 

vidcap = cv2.VideoCapture("Test_vVdeos\LaneVid2.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Define perspective transformation matrix




while success:

    success, image = vidcap.read()
    frame = cv2.resize(image, (640, 480))

    tl = (200, 400)
    bl = (150, 450)
    tr = (400, 400)
    br = (450, 450)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    


    # Apply perspective transformation
    ##transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Apply color thresholding to detect white lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Define region of interest (ROI)
    roi_vertices = np.array([[0, 480], [0, 300], [640, 300], [640, 480]], dtype=np.int32)
    roi_mask = np.zeros_like(binary)
    cv2.fillPoly(roi_mask, [roi_vertices], 255)
    roi_image = cv2.bitwise_and(binary, roi_mask)

    # Apply Canny edge detection
    edges = cv2.Canny(roi_image, 50, 150)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    # Filter and draw lanes
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("LaneVideo", frame)
    cv2.imshow("White Lane Detection", roi_image)

    if cv2.waitKey(25) == 27:
        break
