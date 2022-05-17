
def display_gray_hist(image):
    
    gray_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    # normalize the histogram
    gray_hist /= gray_hist.sum()
    # plot the normalized histogram
    plt.figure()
    plt.title("Grayscale Histogram (Normalized)")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()

class EggDetector:    

    current_image = None
    list_images = None

    normal_thresh_name = "normal_thresh:"
    gap_name = "gap:"
    border_name = "border"
    morph_shape_name = "morph:"
    erosion_iteration_name = "erosion_iteration:"

    def __init__(self, namedWindow = "main"):     
        self.source_window = namedWindow   
        cv2.namedWindow(namedWindow)

    def create_trackbars(self):        
        border = 255
        thresh = 100
        cv2.createTrackbar(self.normal_thresh_name, self.source_window, 184, border, self.param_callback)
        
        border = 255
        thresh = 100
        cv2.createTrackbar(self.gap_name, self.source_window, 20, border, self.param_callback)
        cv2.createTrackbar(self.border_name, self.source_window, 50, border, self.param_callback)   
        
        border = 21
        thresh = 0
        cv2.createTrackbar(self.morph_shape_name, self.source_window, 5, border, self.param_callback)    
        cv2.createTrackbar(self.erosion_iteration_name, self.source_window, 0, border, self.param_callback)    
        
    def param_callback(self, val):
        self.normal_thresh = cv2.getTrackbarPos(self.normal_thresh_name, self.source_window)
        self.gap = cv2.getTrackbarPos(self.gap_name, self.source_window)
        self.border = cv2.getTrackbarPos(self.border_name, self.source_window)
        self.morph_shape = cv2.getTrackbarPos(self.morph_shape_name, self.source_window)
        if self.morph_shape % 2 != 1:
            self.morph_shape = self.morph_shape +1

        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV_FULL)   
        gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)   
        gray = cv2.GaussianBlur(gray,(21,21),0)
        cv2.imshow("nxcor", gray)  
        th, draw_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_shape,self.morph_shape))
        draw_image = cv2.morphologyEx(draw_image, cv2.MORPH_CLOSE, kernel)
        draw_image = cv2.distanceTransform(draw_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        cv2.normalize(draw_image, draw_image, 0, 1.0, cv2.NORM_MINMAX)
        

        borderSize = self.border
        distborder = cv2.copyMakeBorder(draw_image, borderSize, borderSize, borderSize, borderSize,
                                        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
                                        
        cv2.imshow("distborder", distborder)   
        gap = self.gap
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
        kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

        distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)

        cv2.imshow("nxcor", nxcor)   
        mn, mx, _, _ = cv2.minMaxLoc(nxcor)
        th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
        draw_image = cv2.convertScaleAbs(peaks)
        cv2.imshow("wtf", draw_image)   
        
        contours, _ = cv2.findContours(draw_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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
        
        original = self.current_image.copy()
        cv2.drawContours(original,new_contours,-1,(0,0,255), 2)
        cv2.imshow(self.source_window,original)
        cv2.imshow("distance", original)   
    
    def next_image(self):
        if self.list_images is None or len(self.list_images) == 0:
            return False
        if len(self.list_images) != 0:
            self.current_image = cv2.imread(self.list_images[0])
            print(self.list_images[0])
            self.list_images.pop(0)  
            self.param_callback(None) 
            return True    

    def set_list_images_from_path(self, glob_query):
        self.list_images = glob.glob(glob_query)

