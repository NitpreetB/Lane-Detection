import cv2 
import numpy as np 

vidcap = cv2.VideoCapture("Test_vVdeos\LaneVid2.mp4")
success , image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while success:
    success, image = vidcap.read()
    frame = cv2.resize(image,(640,480))
    
    ## marking the region of interest for the lane detection
    ## Choosing points for perspective transformation
    # tl = (200,400)
    # bl = (150 ,450)
    # tr = (400,400)
    # br = (450,450)

    tl = (265,375)
    bl = (150 ,450)
    tr = (370,375)
    br = (450,450)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## applying perspective transformation
    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

    ## matrix to warp the image for birds eye window

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0) ## only consider the bottom half of the mask window, such that we only consider the road lane
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:])+midpoint

    #sliding windows
    
    y = 480
    lx = []
    rx = []
    
    msk = mask.copy()
    while y>0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
        
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        y -= 40
    
    # Applying inverse perspective transformation to sliding windows
    original_mask_resized = cv2.resize(msk, (frame.shape[1], frame.shape[0]))
    original_mask_resized = cv2.cvtColor(original_mask_resized, cv2.COLOR_GRAY2BGR)
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1) 
    original_mask_resized = cv2.warpPerspective(original_mask_resized, inv_matrix, (frame.shape[1], frame.shape[0]))

    # Overlaying the sliding windows on the original image
    result = cv2.addWeighted(frame, 1, original_mask_resized, 0.5, 0)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    cv2.imshow("Result", result)

    if cv2.waitKey(25) == 27:
        break
