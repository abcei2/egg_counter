import cv2
from matplotlib import pyplot as plt
import glob

def count_eggs(image, morph_shape=(5,5), gauss_kernel=(21,21), borderSize = 50, gap = 20,  draw=True):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)   
    grayhsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)   
    # Smoothing image
    blurred = cv2.GaussianBlur(grayhsv, gauss_kernel ,0)
    # Threshold otsu
    th, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Morphologic close ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_shape)
    morph_close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    # Distance Transform
    dist = cv2.distanceTransform(morph_close, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)    
    # Normalize Transform
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)  
    # Still not understanding after this point.
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
                                      
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED) 
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
    peaks8 = cv2.convertScaleAbs(peaks)    
    contours, _ = cv2.findContours(peaks8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for i in range(len(contours)):
        contour = contours[i]

        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)

        (x, y, w, h) = cv2.boundingRect(contour)

        egg_index = i


        if len(contour) >= 5:
            if radius> 15 and radius < 40 :
                new_contours.append(contour)
    if draw:
        original = image.copy()
        cv2.drawContours(original,new_contours,-1,(0,0,255), 2)
        return len(new_contours), original
   
    return len(new_contours), None
        
def load_images():
    frame_list = glob.glob("data/*.jpeg")
    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        amount, egg_frame = count_eggs(frame)
        cv2.imshow("egg_frame",egg_frame)
        print(f"there are {amount} eggs")
        #display_gray_hist(frame)
        cv2.waitKey(0)


load_images()
