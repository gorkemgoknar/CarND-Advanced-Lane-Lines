#%reset -f

# <codecell> Module import header

#import os
#import sys
import cv2
#import glob
import pickle
import numpy as np
#import matplotlib as mpl
import moviepy.editor as mpy
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from moviepy.video.io.bindings import mplfig_to_npimage


# <codecell> Cmera calibration and perspective transform

# Load camera calibration data
# The camera was calibrated using images provided in the project repository
# OpenCV function used are cv2.findChessboardCorners and  cv2.calibrateCamera

dist_pickle = pickle.load(open('./camera_cal.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# Settings for undistorted test image (straight_lines1_ud.jpg)
# These settings were obtained by using the following procedure:
# 0. Assumptions:
#    a. Car is driving at the center of the lane and camera is mounted on the
#       center-line of the car.
#    b. The same camera and mouting system was used in all project videos.
# 1. Undistort the straight_lines1.jpg test image provided in the project
#    repository using the above camera calibration data.
# 2. Source points were obtained by tracing a polygon along the edges of the
#    straight lane lines.
# 3. The region of image showing the the hood is ignored (~60px from bottom).
# 4. The height of the polygon was maximized to get the best estimate of
#    curvature (~0.66 of height to horizon).
#    Note: Excessive distortion in color and gradients of lane lines was
#    observed if the top line was kept close to the horizon.
# 5. Minor adjustment to polygon points was necessary to ensure the lane lines
#    were vertical in the perspective transformed (warped) image.

# Backup:
# Settings for project video
# src = np.float32([[256, 676],  [612, 438], [669, 438], [1051, 676]])
# dst = np.float32([[240, 720],  [240,   0], [1040,  0], [1040, 720]])
# Settings for challenge video
# src = np.float32([[313, 677],  [616, 477], [736, 477], [1099, 677]])
# dst = np.float32([[240, 720],  [240,   0], [1040,  0], [1040, 720]])
# Old Settings
# src = np.float32([[240, 670],  [600, 420], [660, 420], [1080, 670]])
# dst = np.float32([[256, 720],  [256,   0], [1051,  0], [1051, 720]])

src = np.float32([[267, 670],  [580, 460], [705, 460], [1039, 670]])
dst = np.float32([[240, 720],  [240,   0], [1040,  0], [1040, 720]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = np.linalg.inv(M)


# <codecell> Line class definition


# Define a class to receive the characteristics of each line detection
# Here, the recommended class structure was adapted, and only the most critical
# properties were utilized to define the line class.

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last 5 fits of the line
        self.recent_fits = deque(maxlen=5)
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.empty((0,3),np.int8)  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.empty((0,3),np.int8)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 


# <codecell> Undistort and warp input image

# Define function to undistort the image using camera calibraiton data and
# warp the region of interest on the road to full image size (1280x720)

def undistort_and_warp_image(img, M):

    global src, dst, mtx, dist
    
    # Retrieve the image size
    h, w = img.shape[:2]
    
    # Undistort image using camera calibration data
    undist = cv2.undistort(img, mtx, dist, None)
    plotimg_org = undist.copy()
    pts = np.array(src, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(plotimg_org, [pts], True, (0,0,255),2) 

    # Warp image based on the perspective transformation map
    warped = cv2.warpPerspective(undist, M, (w, h), flags=cv2.INTER_LINEAR)
    #plotimg_wrp = warped.copy()
    plotimg_wrp = get_clahe_rgb(warped)
    pts = np.array(dst, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(plotimg_wrp, [pts], True, (0,0,255), 2)
    
    return (undist, warped, plotimg_wrp, plotimg_org)

# Additional code for debug/testing (load pre-requisites & run in cell mode):
#img = cv2.imread('test_images/straight_lines1.jpg')
#cv2.imshow('input', img) ; cv2.waitKey()
#img_ud = cv2.undistort(img, mtx, dist, None)
#cv2.imwrite('test_images/straight_lines1_ud.jpg', img_ud)
#cv2.imshow('undistorted', img_ud) ; cv2.waitKey()
#img_udwrp = undistort_and_warp_image(img, M)[1]
#cv2.imwrite('test_images/straight_lines1_udwrp.jpg', img_udwrp)
#cv2.imshow('undistorted and warped', img_udwrp) ; cv2.waitKey()
#cv2.destroyAllWindows()


# <codecell> Magnitude and color thresholds

# Define graident magnitude and color threshold functions

def mag_thresh(img):
    
    # Show the input image for debugging purpose
    #cv2.imshow('debug', img) ; cv2.waitKey() ; cv2.destroyAllWindows()
    
    # Settings for Sobel gradient transform
    # Set kernel size for transform - must be an odd number
    sobel_kernel = 5 # setting for harder_challenge=9, previous=5
    # Set the low/high cut-off gradient values
    #mag_thresh = (5, 110) # setting for project and challenge
    mag_thresh = (25, 110) # setting for harder_challenge
    
    # Following steps were applied to the input image using above settings.
    # Notes:
    # a. In addition to Sobel, cv2.Laplacian() and cv2.Scharr() gradient
    #    transforms were investigated. In conclusion, Sobel gave best results.
    # b. Scaled magnitude of Sobel gradients in X & Y direction provided the
    #    best estimate of the lane marking boundaries.
    # c. As the graident operation preserves a narrow regions near edges of
    #    each lane mark, a morphology operation (dialate) was used to improve
    #    detail in the final binary image. This information will be combined
    #    with color data to get a better estimate of the lane marking extent.
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # addition for harder_challenge
    #gray_blur = cv2.medianBlur(gray,5)
    
    # 2) Take gradient in x and y separately (can try for color channels too)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # change for harder_challenge - gray -> gray_blur
        
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scl_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scl_sobel)
    binary_output[(scl_sobel >= mag_thresh[0]) & (scl_sobel <= mag_thresh[1])] = 1
    
    # 6) Perform morphology operation (dilate) and return image
    binary_output = cv2.dilate(binary_output, np.ones((8,8),np.uint8), 2)
    #binary_output = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE,
    #                                  np.ones((10,10),np.uint8))
    
    # Show the output image for debugging purpose
    #cv2.imshow('debug', binary_output*255) ; cv2.waitKey() ; cv2.destroyAllWindows()
    
    return binary_output


def color_thresh(img):
    
    # Show the input image for debugging purpose
    #cv2.imshow('debug', img) ; cv2.waitKey() ; cv2.destroyAllWindows()
    
    # Following steps were applied to the input image using above settings.
    # Notes:
    # a. Four color spaces were investigated: RGB, HLS, HSV, YUV.
    #    Out of the four - R, G, S (HLS) and V (HSV) were found to be the most
    #    suitable channels/layers to isolate the lane markings
    # b. Trials with S & V layers did not generalize well across all project
    #    videos, epecially challenge_video.mp4 due washout of color graidents
    # c. Since its difficult to yellow and white lane markings in a single
    #    pass, the following implementation uses two passes to filter the
    #    yellow and white line using appropriate thresolds and then combines
    #    the threshold images for return
    # d. Similar to implementation for the gradients, the color binary
    #    threshold image feature are dilated to get better results when
    #    combined (ANDed) with graident threshold image.
    
    img = get_clahe_rgb(img)
    
    R = img[:,:,0] ; G = img[:,:,1] ; B = img[:,:,2]
    #S = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,1]
    #V = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    yline = np.zeros_like(R) ; wline = np.zeros_like(R)
    # Identify only the yellow lines by taking advanced of RGB distribution
    #yline[(R > 150) & (G > 140) & (B < 130)] = 1 # setting for project & challenge
    yline[(R > 220) & (G > 210) & (B < 160)] = 1 # setting for harder_challenge
    # Identify only the white lines using highest values in RGB channels
    #wline[(R > 170) & (G > 170) & (B > 170)] = 1 # setting for project & challenge
    wline[(R > 200) & (G > 200) & (B > 200)] = 1 # setting for harder_challenge
    # Combine (ORed) both image to include all (yellow/white) lane markings
    
    # Setting for harder_challenge - clear unwanted region
    yline[:,880:] = 0 ;  wline[:,:400] = 0
    
    binary_output = (yline|wline)
    
    # Older recepie used for binary_output. Both these gave reasonable results
    #binary_output = (Vb|(Rb&Gb))
    #binary_output = ((Sb&Vb)|(Rb&Gb))
    
    # Older settings used for project_video.mp4
    #yline[(R > 170) & (G > 150) & (B < 140)] = 1
    #wline[(R > 170) & (G > 170) & (B > 170)] = 1
    
    # Save images to test output folder for debug purpose
    #cv2.imwrite('my_test_images/yline.jpg',yline*255)
    #cv2.imwrite('my_test_images/wline.jpg',wline*255)
    
    # 6) Perform morphology operation (dilate) and return image
    binary_output = cv2.dilate(binary_output, np.ones((10,10),np.uint8), 1)
    
    # Show the output image for debugging purpose
    #cv2.imshow('debug', binary_output*255) ; cv2.waitKey() ; cv2.destroyAllWindows()
    
    return binary_output

# Additional code for debug/testing (load pre-requisites & run in cell mode):
# Note: For testing, exchange R & B layers for cv2.imread to work properly
#
#img = cv2.imread('my_test_images/project_video_0640.jpg')
#img = cv2.imread('my_test_images/challenge_video_0115.jpg') 
#img = cv2.imread('my_test_images/harder_challenge_video_0170.jpg') 
#img = cv2.imread('output_images/input_frame_0.00.jpg')
#img_udwrp = undistort_and_warp_image(img, M)[1]
#cv2.imshow('uawimg', img_ud) ; cv2.waitKey()
#
#img_udwrp_ch = cv2.cvtColor(img_udwrp, cv2.COLOR_RGB2HLS)[:,:,2]
#cv2.imwrite('my_test_images/challenge_video_0115_udwrp_ch.jpg', img_udwrp_ch)
#img_udwrp_ch_bin = np.zeros_like(img_udwrp_ch)
#img_udwrp_ch_bin[(img_udwrp_ch > 90) & (img_udwrp_ch < 255)] = 1
#cv2.imshow('channel',img_udwrp_ch_bin*255) ; cv2.waitKey()
#cv2.imwrite('my_test_images/challenge_video_0115_udwrp_ch_bin.jpg', img_udwrp_ch_bin*255)
#
#img_udwrp_mth = mag_thresh(img_udwrp)
#img_udwrp_cth = color_thresh(img_udwrp)
#img_udwrp_mcth = (mag_thresh(img_udwrp)&color_thresh(img_udwrp))
#cv2.imwrite('my_test_images/project_video_0640_mag_thresh.jpg',img_udwrp_mth*255)
#cv2.imwrite('my_test_images/project_video_0640_color_thresh.jpg',img_udwrp_cth*255)
#cv2.imwrite('my_test_images/project_video_0640_composite.jpg',img_udwrp_mcth*255)
#cv2.imshow('composite',img_udwrp_mcth*255) ; cv2.waitKey()
#cv2.destroyAllWindows()


# <codecell>

def get_clahe_rgb(img):
    
    clahe_l = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(10,10))
    clahe_b = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe_l.apply(lab[:,:,0])
    lab[:,:,2] = clahe_b.apply(lab[:,:,2])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    
# <codecell>

def cielab_iat_thresh(img):
    
    L = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,0]
    L_IAT = cv2.bitwise_not(cv2.adaptiveThreshold(L, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5))
    dLx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=5)
    dLy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=5)
    dL = np.sqrt(np.square(dLx) + np.square(dLy))
    dL_scl = np.uint8(255*dL/np.max(dL))
    L_sbl = np.zeros_like(L) ; L_sbl[dL_scl>35] = 1
    
    binary_output = np.zeros_like(L)
    binary_output[((L_IAT==255)&(L_sbl==1))] = 1
    binary_output = cv2.dilate(binary_output, np.ones((5,5),np.uint8), 1)
    #binary_output = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE,
    #                                  np.ones((10,10),np.uint8))
    
    return binary_output


# <codecell> Window masks
    
# Reusable function to get windowed mask at a given level long image height

def window_mask(width, height, img_ref, cx, cy):
    
    output = np.zeros_like(img_ref)
    hmin = int(cy-height/2)
    hmax = int(cy+height/2)
    wmin = max(0,int(cx-width/2))
    wmax = min(int(cx+width/2),img_ref.shape[1])
    output[hmin:hmax,wmin:wmax] = 1
    
    return output


# <codecell> Window centroids along lane lines

# Function to determine the centroid of sliding window. This function uses
# the convolution method to find the centroids that best represent the lane
# location along different level of image height

def find_window_centroids(image, window_width, window_height, margin):
    
    # Define parameters used in the following steps:
    h = image.shape[0] # height of the input image
    w = image.shape[1] # width of the input image
    frac = 0.1 # Lower fraction of image for estimating locaitono of lanes
    # setting for project and challenge is 0.25, harder_challenge is 0.1
    
    # Sliding window settings
    window = np.ones(window_width) # Current window for convolution
    # Note: window_width/2 offset used as convolution signal reference
    # is at right side of window, not center of window
    offset = window_width/2
    # The following parameter can be tune to increase/decrese sensitivity to
    # noise from gradient/color thresholding. The sliding window method will
    # only register the centroid coordinates if a minimum (2%) of pixels
    # are filled with non-zero (or 1) values
    window_fill_threshold = (2.0*window_width*window_height/100.0)*255
    
    # Left centroids initialization
    l_lvls = np.empty((0,1),int)
    l_cnts = np.empty((0,2),int)
    l_sum = np.sum(image[int((1-frac)*h):,:int(w/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum)) - window_width/2
    
    # PeakUtils for getting first peak
    # install peakutils (tar.gz) using following
    #pip install <path-to-peakutils-on-pypi.python.org>
    # More information on following links
    #https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
    #http://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
    #https://plot.ly/python/peak-finding/
    #https://github.com/MonsieurV/py-findpeaks
    # setting for harder_challenge
    #from peakutils.peak import indexes as pkids
    #plt.plot(np.convolve(window,l_sum)) ; plt.show()
    #print(pkids(np.convolve(window,l_sum),
    #                 thres=0, min_dist=200))
    
    # Reight centroids initialization
    r_lvls = np.empty((0,1),int)
    r_cnts = np.empty((0,2),int)
    r_sum = np.sum(image[int((1-frac)*h):,int(w/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum)) - window_width/2 + int(w/2)
    
    #plt.plot(np.convolve(window,r_sum)) ; plt.show()
    #print(pkids(np.convolve(window,r_sum),
    #                thres=window_fill_threshold, min_dist=2))
    
    # setting for harder_challenge
    #r_center = pkidx(np.convolve(window,r_sum),
    #                 thres=window_fill_threshold, min_dist=2)[0]
    
    for l in range(int(h/window_height)): # Loop over all the levels (~9)
        
        # 1. Convolve the window into the vertical slice of the image
        hupper = int(h - (l+1)*window_height)
        hlower = int(h - l*window_height)
        hmid = int(0.5*(hlower+hupper))
        image_layer = np.sum(image[hupper:hlower,:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Setting for harder_challenge
        #if l<5:
        #    conv_signal[:200] = 0 ; conv_signal[1000:] = 0
        
        # The following two steps identify the starting point for the sliding
        # windows method for current image layer level.
        # Notes:
        # a. The method discussed in class has been augmented to handle large
        #    variations in curvature.
        # b. Following implementation project (using linear extrapolation)
        #    into current image layer for the starting point using centroids
        #    detected in last two layers.
        # c. A check is used to make sure that at least two previously
        #    detected centroids are available. If not, default to the method
        #    discussed in class using previous center + offset + margin
        # d. Margin + offset around projected point in then used find the
        #    lane center using sliding window convolution. This enables
        #    to easily detect continuous lane centroids even if the lane
        #    turns sharply (small radius of curvature)
        
        # 1a. Find the best left lane centroid search starting position
        # Settings for harder_challenge : margin -> 1.5*margin
        if len(l_cnts) < 2:
            l_min = int(max(l_center - offset - 1.5*margin, 0))
            l_max = int(min(l_center + offset + 1.5*margin, w))
        else:
            l_proj = l_cnts[-1,0] + (l_cnts[-1,0]-l_cnts[-2,0]) \
                   * ((l - l_lvls[-1])/(l_lvls[-1]-l_lvls[-2]))
            if l_proj < 0: l_proj = 0
            if l_proj > w: l_proj = w
            l_min = int(max(l_proj - offset - 1.5*margin, 0))
            l_max = int(min(l_proj + offset + 1.5*margin, w))
        if (l_min >= 0 and l_max <= w) \
        and np.argmax(conv_signal[l_min:l_max]) > 0 \
        and np.max(conv_signal[l_min:l_max]) > window_fill_threshold:
            l_center = np.argmax(conv_signal[l_min:l_max]) + l_min - offset
            l_lvls = np.append(l_lvls, l)
            l_cnts = np.append(l_cnts, [[l_center, hmid]], axis=0)
        
        # 1b. Find the best right lane centroid search starting position
        # Settings for harder_challenge : margin -> 1.5*margin
        if len(r_cnts) < 2:
            r_min = int(max(r_center - offset - 1.5*margin, 0))
            r_max = int(min(r_center + offset + 1.5*margin, w))
        else:
            r_proj = r_cnts[-1,0] + (r_cnts[-1,0]-r_cnts[-2,0]) \
                   * ((l - r_lvls[-1])/(r_lvls[-1]-r_lvls[-2]))
            if r_proj < 0: r_proj = 0
            if r_proj > w: r_proj = w
            r_min = int(max(r_proj - offset - 1.5*margin, 0))
            r_max = int(min(r_proj + offset + 1.5*margin, w))
        if (r_min >= 0 and r_max <= w) \
        and np.argmax(conv_signal[r_min:r_max]) > 0 \
        and np.max(conv_signal[r_min:r_max]) > window_fill_threshold:
                r_center = np.argmax(conv_signal[r_min:r_max]) + r_min - offset
                r_lvls = np.append(r_lvls, l)
                r_cnts = np.append(r_cnts, [[r_center, hmid]], axis=0)
    
    # If not centroids found, return with false flag and original image itself
    if len(l_cnts) == 0 or len(r_cnts) == 0:
        output = np.array(cv2.merge((image,image,image)),np.uint8)
        return False, output, l_cnts, r_cnts

    # If centroids found, create a mask image with rectangles showing centroid
    # locaiton and windows size superimposed on the binary image
    template = np.zeros_like(image)
    for nc in range(len(l_cnts)):
        l_mask = window_mask(window_width, window_height,
                             image, l_cnts[nc,0], l_cnts[nc,1])
        template[l_mask == 1] = 255
    for nc in range(len(r_cnts)):
        r_mask = window_mask(window_width, window_height,
                             image, r_cnts[nc,0], r_cnts[nc,1])
        template[r_mask == 1] = 255
    mask_template = np.array(cv2.merge((np.zeros_like(template), template,
                                        np.zeros_like(template))), np.uint8)
    binary_warped_color = np.array(cv2.merge((image,image,image)),np.uint8)
    output = cv2.addWeighted(binary_warped_color, 1, mask_template, 0.5, 0.0)
    
    # Return flag, image and centroids for both left/right lane markings
    return True, output, l_cnts, r_cnts

# Additional code for debug/testing (load pre-requisites & run in cell mode):
#img = cv2.imread('project_video_1144.jpg')
#cv2.imshow('input', img) ; cv2.waitKey()
#cv2.imshow('uawimg', undistort_and_warp_image(img, M)[1]) ; cv2.waitKey()
#binary_warped = color_thresh(undistort_and_warp_image(img, M)[1]) \
#              & mag_thresh(undistort_and_warp_image(img, M)[1])
#cv2.imshow('binary_warped', binary_warped*255) ; cv2.waitKey()
#flg, bwcimg, lcs, rcs = find_window_centroids(binary_warped*255, 50, 80, 120)
#cv2.imshow('output', bwcimg) ; cv2.waitKey()
#cv2.destroyAllWindows()


# <codecell> Fit polynomial to centroid points
    
# Helper function to perform fitting on points identified for left
# and right lane marking using the sliding windows method

def polyfit_and_compute_radcurv(sws_flag, lcpts, rcpts, lline, rline, margin):
    
    global debug_log, df_flag, num_drop_frames, max_drop_frames, all_drop_frames
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.66/90 # mt/px in y (12 ft broken white lane in 90 pxs)
    xm_per_pix = 3.66/770 # mt/px in x (12 ft wide lane in in 770 pxs for ROI)
    y_eval = 600 # Evaluate radius of curvature at ~0.8 of image height
    
    # Procedure to determine meters/pixel coefficients using highway markings
    # 1. From video feed, select a frame that has best centerline view of a
    #    straight section of the freeway with atleast one broken white lane
    # 2. Undistrot and warp image using calibraton data and a chosen ROI
    # 3. Determine the center to center distance of one two lane marking in px
    #    -> xm_per_pix will be 3.66m divided by measures px value
    # 4. Determine the length of at least 3 long borken lane markings in px
    #    -> ym_per_pix will be 3.66m divided by measures px value
    # See following links for more information on multi-lane highway markings
    # 1. http://www.dot.ca.gov/trafficops/camutcd/camutcd2014rev2.html
    # 2. https://mutcd.fhwa.dot.gov/htm/2009r1r2/html_index.htm
    #    (see Chap3A section 3A.06 for broken white line with and
    #     figure 3A-102 Detal 11 for broken white line length)

    # Return with dropped frame flag set if no centroids were found
    if len(lcpts)==0 or len(rcpts)==0:
        df_flag = True
        num_drop_frames += 1
        all_drop_frames += 1
        lline.current_fit = lline.best_fit
        lline.detected = False
        rline.current_fit = rline.best_fit
        rline.detected = False
        return df_flag, lline, rline
    
    # Fit 2nd order polynomial using centroids from sliding window (OR)
    # the points within margin of previously detected lanes
    lfit = np.polyfit(lcpts[:,1], lcpts[:,0], 2)
    rfit = np.polyfit(rcpts[:,1], rcpts[:,0], 2)
    
    #print('\n -->', lfit, '\n' rfit, '\n')
    
    # Calculate the new radii of curvature using standard formula and pixel
    # calibration data obtained from above (xm_per_pix, ym_per_pix)
    lftmp = np.polyfit(lcpts[:,1]*ym_per_pix, lcpts[:,0]*xm_per_pix, 2)
    rftmp = np.polyfit(rcpts[:,1]*ym_per_pix, rcpts[:,0]*xm_per_pix, 2)
    lcrv = ((1 + (2*lftmp[0]*y_eval*ym_per_pix + lftmp[1])**2)**1.5) \
    / np.absolute(2*lftmp[0])
    rcrv = ((1 + (2*rftmp[0]*y_eval*ym_per_pix + rftmp[1])**2)**1.5) \
    / np.absolute(2*rftmp[0])
    
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    # The following section of code perform sanity checks on the newly fitted
    # lane lines. The steps are indicated using numerals. High level overview
    # of the philosophy is in the notes below.
    # Notes:
    # a. Three sanity checks are performed to make sure that the new fit is
    #    not significantly different from previously identified fit, and that
    #    the left and the right lanes in the current fit are separated by same
    #    distance along the height of image and within regulation lane width
    # b. Update is performed if conditions for sanity checks are met or the
    #    number of drop frames have exceeded maximum allowed
    
    # 1. Check if fit is similar to previously fit line. In time, the lane
    #    curvature of the current fit should be within 1/2 order magnitude
    #    of curvature of previously fit lane line. For straight lines, which
    #    have a large curvature, this check is ignored (R > 10 kms)
    # Note: Disabling this check on absolute value of curvature, as it is
    #       sensitive to minor change/shift in the detected centroids
    #       especially when the lane lines are straight.
    #       Rely only on margin based checks
    chk1_left = True ; chk1_right = True
    # Change for harder_challenge
    #y_test = np.linspace(0,720,9)
    y_test = np.linspace(300,700,9)
    if len(lline.current_fit) == 0:
        chk1_left = False
    else:
        lx_prv = lline.current_fit[0]*y_test**2 \
        + lline.current_fit[1]*y_test + lline.current_fit[2]
        lx_new = lfit[0]*y_test**2 + lfit[1]*y_test + lfit[2]
        lx_tdiff = np.mean(np.abs(lx_prv - lx_new))
        #lcrv_tdiff = 0
        #if lline.radius_of_curvature < 10000 and lcrv < 10000:
        #    lcrv_tdiff = abs(np.log10(lline.radius_of_curvature)-np.log10(lcrv))
        chk1_left = (lx_tdiff > 2.0*margin) # or (lcrv_tdiff > 0.5)
        # setting for project and challenge was 0.5*margin
        if debug_log:
            with open("log.txt", "a") as myfile:
                myfile.write('\n --- chk1_left ---')
                myfile.write('\n x_tdiff {0:.4f} (>{1})'.format(lx_tdiff,2.0*margin))
                #myfile.write('\n log_roc_tdiff {0:.4f} (>{1})'.format(lcrv_tdiff,0.5))
    if len(rline.current_fit) == 0:
        chk1_right = False
    else:
        rx_prv = rline.current_fit[0]*y_test**2 \
        + rline.current_fit[1]*y_test + rline.current_fit[2]
        rx_new = rfit[0]*y_test**2 + rfit[1]*y_test + rfit[2]
        rx_tdiff = np.mean(np.abs(rx_prv - rx_new))
        #rcrv_tdiff = 0
        #if rline.radius_of_curvature < 10000 and rcrv < 10000:
        #    rcrv_tdiff = abs(np.log10(rline.radius_of_curvature)-np.log10(rcrv))
        chk1_right = (rx_tdiff > 2.0*margin) # or (rcrv_tdiff > 0.5)
        # setting for project and challenge was 0.5*margin
        if debug_log:
            with open("log.txt", "a") as myfile:
                myfile.write('\n --- chk1_right ---')
                myfile.write('\n x_tdiff {0:.4f} (>{1})'.format(rx_tdiff,2.0*margin))
                #myfile.write('\n log_roc_tdiff {0:.4f} (>{1})'.format(rcrv_tdiff,0.5))
    
    # 2. Check if left and right curvature is similar (within half log-order)
    # Note: Disabling check 2 as the changes in curvature are sensitive to
    #       minor shift in the points - especially when the line is straight
    #lr_crv_diff = 0
    #if lcrv < 10000 and rcrv < 10000:
    #    lr_crv_diff = abs(np.log10(lcrv)-np.log10(rcrv))
    #chk2 =  (lr_crv_diff < 0.5)
    chk2 = True
    
    # 3. Check if mean separation distance is within 2 ft of regulation for
    #    lane width on typical roads/freeways, which is ~12 ft (10 ft - 14ft)
    # setting for harder_challenge is 9ft - 15ft
    y_test = np.linspace(400,700,7) ; lwmin = 6*0.3048 ; lwmax = 15*0.3048
    lx_test = lfit[0]*y_test**2 + lfit[1]*y_test + lfit[2]
    rx_test = rfit[0]*y_test**2 + rfit[1]*y_test + rfit[2]
    dx_LR = (rx_test-lx_test)*xm_per_pix
    chk3 = (dx_LR>0).all() and (dx_LR>lwmin).all() and (dx_LR<lwmax).all()
    #chk3 = (dx_LR>0).all() # settings for harder_challenge
    
    if debug_log:
        with open("log.txt", "a") as myfile:
            myfile.write('\n --- chk2 ---')
            myfile.write('\n *** DISABLED ***')
            #myfile.write('\n ROCs: {0:.4f} {1:.4f}'.format(lcrv, rcrv))
            #myfile.write('\n log_roc_lrdiff: {0:.4f} (<{1})'.format(lr_crv_diff,0.5))
            myfile.write('\n --- chk3 ---')
            myfile.write('\n dx_LR: avg = {0:.4f}, std = {1:.4f}'.format( \
                         np.mean(dx_LR), np.std(dx_LR)))
            myfile.write('\n dx_LR: min = {0:.4f}, max = {1:.4f}'.format( \
                         np.min(dx_LR), np.max(dx_LR)))
            myfile.write('\n --- all_flags ---')
            myfile.write('\n sws_flag = {}'.format(sws_flag))
            myfile.write('\n chk1_left = {}'.format(chk1_left))
            myfile.write('\n chk1_right = {}'.format(chk1_right))
            myfile.write('\n chk2 = {}'.format(chk2))
            myfile.write('\n chk3 = {}'.format(chk3))
            myfile.write('\n --- drop_frame_status ---')
            myfile.write('\n num(max):all = {0}({1}):{2}'.format( \
                         num_drop_frames, max_drop_frames, all_drop_frames))
            myfile.write('\n ----------- \n')
    
    # The final decision to update the lane fit is based on following logic:
    # a. First check to see if recent sliding window search was successful
    #    while the dropped frames exceed the maximum allowed. In this case,
    #    the lane fit has to be update immediately, even if last two sanity
    #    check (2 & 3) are not satisfied. Only check is made to ensure that
    #    the new fits are not significantly away due to errors.
    # b. If the number of dropped frames are still within limit, update the
    #    current fit using sliding window / margin point based fit if all the
    #    three sanity checks are satisfied.
    ret_flag = False
    if sws_flag and (num_drop_frames > max_drop_frames):
        if chk2 and chk3: ret_flag = True
        else: ret_flag = False
    elif not (chk1_left or chk1_right) and (chk2 and chk3):
        ret_flag = True
    
    # Update based on the return flag set by pervious logic.
    if ret_flag:
        df_flag = False
        num_drop_frames = 0
        # Update left and right lines
        lline.detected = True ; rline.detected = True
        lline.current_fit = lfit ; rline.current_fit = rfit
        lline.radius_of_curvature = lcrv ; rline.radius_of_curvature = rcrv
        lline.recent_fits.append(lfit) ; rline.recent_fits.append(rfit)
        lline.best_fit = np.mean(lline.recent_fits, axis=0)
        rline.best_fit = np.mean(rline.recent_fits, axis=0)
        return df_flag, lline, rline
    else:
        df_flag = True
        lline.detected = False
        rline.detected = False
        num_drop_frames += 1
        all_drop_frames += 1
        return df_flag, lline, rline


# <codecell> Plot lane region on warped image ROI
        
def plot_warped_lane_regions(warped, lline, rline, margin=100):
    
    mask1 = np.zeros_like(warped)
    mask2 = np.zeros_like(warped)
    
    lfit = lline.current_fit
    rfit = rline.current_fit
    
    # Generate x and y values for plotting
    ly = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    ly = ly.astype(int)
    lx = lfit[0]*ly**2 + lfit[1]*ly + lfit[2]
    lx = lx.astype(int)
    idx = (lx > 0) & (lx < warped.shape[1])
    plx = lx[idx] ; ply = ly[idx]
    
    ry = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    ry = ry.astype(int)
    rx = rfit[0]*ry**2 + rfit[1]*ry + rfit[2]
    rx = rx.astype(int)
    idx = (rx > 0) & (rx < warped.shape[1])
    prx = rx[idx] ; pry = ry[idx]
    
    pts = np.transpose(np.vstack((plx,ply)))
    cv2.polylines(mask1, [pts], False, (255,0,0), 20)
    pts = np.transpose(np.vstack((prx,pry)))
    cv2.polylines(mask1, [pts], False, (0,0,255), 20)
    
    # Generate a polygon to visualize search window area
    lw1 = np.array([np.transpose(np.vstack([plx-margin, ply]))])
    lw2 = np.array([np.flipud(np.transpose(np.vstack([plx+margin, ply])))])
    lpoly = np.hstack((lw1, lw2))
    rw1 = np.array([np.transpose(np.vstack([prx-margin, pry]))])
    rw2 = np.array([np.flipud(np.transpose(np.vstack([prx+margin, pry])))])
    rpoly = np.hstack((rw1, rw2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(mask2, np.int_([lpoly]), (0,255, 0))
    cv2.fillPoly(mask2, np.int_([rpoly]), (0,255, 0))
    
    output = cv2.addWeighted(warped, 1.0, mask1, 1.0, 0.0)
    output = cv2.addWeighted(output, 1.0, mask2, 0.1, 0.0)
    
    return output
    

# <codecell> Visualize lanes on undistored image
    
# Plot lanes (left:red, right:blue) and drivable zone between lanes (green)
# superimposed on top of the original image. Note that inverse perspective
# transform matrix (Minv) is required for proper visualization

# Notes:
# 1. Implemented display of radius of curvature for left and right lanes
# 2. Added current status and total count for dropped frames

def plot_lanes_on_undist(undist, warped, Minv, lline, rline):
    
    global df_flag, num_drop_frames, max_drop_frames, all_drop_frames 
    
    lfit = lline.current_fit
    rfit = rline.current_fit
    lcrv = lline.radius_of_curvature
    rcrv = rline.radius_of_curvature
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped)
    color_mask = np.zeros_like(warped)
    
    ly = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    ly = ly.astype(int)
    lx = lfit[0]*ly**2 + lfit[1]*ly + lfit[2]
    lx = lx.astype(int)
    idx = (lx > 0) & (lx < warped.shape[1])
    plx = lx[idx] ; ply = ly[idx]
    
    ry = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    ry = ry.astype(int)
    rx = rfit[0]*ry**2 + rfit[1]*ry + rfit[2]
    rx = rx.astype(int)
    idx = (rx > 0) & (rx < warped.shape[1])
    prx = rx[idx] ; pry = ry[idx]
    
    pts = np.transpose(np.vstack((plx,ply)))
    cv2.polylines(color_mask, [pts], False, (255,0,0), 20)
    pts = np.transpose(np.vstack((prx,pry)))
    cv2.polylines(color_mask, [pts], False, (0,0,255), 20)
    
    # Generate a polygon to visualize search window area
    l_line = np.array([np.transpose(np.vstack([plx, ply]))])
    r_line = np.array([np.flipud(np.transpose(np.vstack([prx, pry])))])
    lane_poly = np.hstack((l_line, r_line))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([lane_poly]), (0,255, 0))
    
    color_warp = cv2.addWeighted(color_warp, 0.5, color_mask, 1.0, 0.0)
    
    # Warp the blank back to original image space using the
    # inverse perspective matrix (Minv)
    undist_lane = cv2.warpPerspective(color_warp, Minv,
                                      (undist.shape[1],undist.shape[0])) 
    # Combine the result with the original image
    output = cv2.addWeighted(undist, 1, undist_lane, 0.5, 0)
    
    xm_per_pix = 3.66/770
    center_offset = (0.5*(lx[-1]+rx[-1])-640)*xm_per_pix
    cv2.putText(output,'LEFT-ROC = {0:.2f} m'.format(lcrv), (50,70),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA, False)
    cv2.putText(output,'RIGHT-ROC = {0:.2f} m'.format(rcrv), (50,120),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA, False)
    cv2.putText(output,'CENTER-OFFSET = {0:.2f} m'.format(center_offset), (50,170),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA, False)
    
    if df_flag:
        cc = (255,0,0)
    else:
        cc = (0,255,0)
    cx = output.shape[1]-130 ; cy = 80 ; cd = 40
    cv2.circle(output, (cx, cy), cd, cc, -1)
    msg1 = 'Drop Frame'
    msg2 = 'Count: {:03d}'.format(all_drop_frames)
    font = cv2.FONT_HERSHEY_SIMPLEX
    loc1 = (output.shape[1]-200, 150)
    loc2 = (output.shape[1]-200, 180)
    colr = (255,255,255)
    cv2.putText(output, msg1, loc1, font, 0.8, colr, 2, cv2.LINE_AA)
    cv2.putText(output, msg2, loc2, font, 0.8, colr, 2, cv2.LINE_AA)
    
    
    return output


# <codecell> MoviePy video clip frame maker
    
def make_frame(t):
    
    # Load image for testing
    #img = cv2.imread('./test_images/straight_lines1.jpg')
    
    global debug_log, df_flag, left_line, right_line
    global num_drop_frames, max_drop_frames, all_drop_frames
    
    # Open a file for logging debug output
    if debug_log:
        with open("log.txt", "a") as myfile:
            myfile.write('\n ====== {:.2f} ======'.format(t))
    
    # Load the frame from video clip for processing
    img = src_clip.get_frame(t)
    
    # Undistort, warp (perspective transform) and apply gradient/color
    # thresholds to the input image
    undist, warped, plotimg_wrp, plotimg_org = undistort_and_warp_image(img, M)
    binary_warped = (color_thresh(warped) & mag_thresh(warped))
    #binary_warped = cielab_iat_thresh(warped)
    
    # Debug code for saving / displaying warped image
    #plt.imsave('output_images/binary_warped_{:.2f}.jpg'.format(t), binary_warped)
    #cv2.imshow('debug', binary_warped*255) ; cv2.waitKey() ; cv2.destroyAllWindows()
    
    # Sliding window attributes
    # setting for harder_challenge : window_height 80 -> 40
    window_width = 50 ; window_height = 40 # 9 layers for height 720
    margin = 60 # How much to slide left and right for searching
    # setting for project and challenge was margin = 120
    
    # Steps for the algorightms are as below (shown using numerals 1-6)
    
    # 1a. Check if the current frame is due for refresh. If the previous
    #     searches did not find a line and frame drop limit is reach, then
    #     force a search for lane lines using sliding widnow method.
    # 1b. If not due for refresh, update line using image points that fall
    #     within the margins.
    if not left_line.detected or not right_line.detected \
    or num_drop_frames > max_drop_frames:
        sws_flag, bwcimg, lcpts, rcpts = find_window_centroids(binary_warped*255,
                                        window_width, window_height, margin)
        #print(len(lcpts), len(rcpts))
        #print(lcpts, rcpts)
    else:
        sws_flag = False
        nz = np.nonzero(binary_warped*255)
        bwcimg = np.array(cv2.merge((binary_warped,binary_warped,
                                     binary_warped)),np.uint8)*255
        
        # If the current frame is dropped (did not pass sanity checks), then
        # update the with best_fit (average of last 5 frames).
        # Otherwise, use current_fit (previous frame)
        if df_flag:
            lfit = left_line.best_fit
            rfit = right_line.best_fit
        else:
            lfit = left_line.current_fit
            rfit = right_line.current_fit
        # Get left
        lxmin = lfit[0]*(nz[0]**2) + lfit[1]*nz[0] + lfit[2] - margin
        lxmax = lfit[0]*(nz[0]**2) + lfit[1]*nz[0] + lfit[2] + margin
        idx = ((nz[1] > lxmin) & (nz[1] < lxmax))
        lcpts = np.transpose(np.vstack([nz[1][idx],nz[0][idx]]))
        bwcimg[nz[0][idx],nz[1][idx]] = (0, 255, 0)
        # Get right
        rxmin = rfit[0]*(nz[0]**2) + rfit[1]*nz[0] + rfit[2] - margin
        rxmax = rfit[0]*(nz[0]**2) + rfit[1]*nz[0] + rfit[2] + margin
        idx = ((nz[1] > rxmin) & (nz[1] < rxmax))
        rcpts = np.transpose(np.vstack([nz[1][idx],nz[0][idx]]))
        # Create temp image
        bwcimg[nz[0][idx],nz[1][idx]] = (0, 255, 0)
        
    # 2. Fit the lines through the points identified in previous step
    df_flag, left_line, right_line = polyfit_and_compute_radcurv(sws_flag,
                                lcpts, rcpts, left_line, right_line, margin)
    #print(left_line.current_fit, right_line.current_fit)
    
    # 3. Plot the identified left and right lanes on warped ROI region
    wlrimg = plot_warped_lane_regions(warped, left_line, right_line, margin)
    
    # 4. Plot the identified left and right lanes on undistorted image
    ulrimg = plot_lanes_on_undist(undist, warped, Minv, left_line, right_line)
    
    # 5. Assemble the sub-plot style figure for project output submission
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[2,2])
    ax5 = fig.add_subplot(gs[2,1])
    ax6 = fig.add_subplot(gs[2,0])
    
    # Axes 1: Undistorted Image with Lane Region & Curvature
    ax1.imshow(ulrimg, cmap='gray')
    ax1.set_title('Undistorted Image with Lane Region & Curvature', fontsize=16)
    
    # Axes 2: Original image (recieved as-is)
    ax2.imshow(img, cmap='gray')
    ax2.set_title('Original', fontsize=16)
    
    # Axes 3: Undistorted image with ROI region polgon superimposed
    ax3.imshow(plotimg_org, cmap='gray')
    ax3.set_title('Undistorted Image with ROI', fontsize=16)
    
    # Axes 4: Warped image with ROI region superimposed
    ax4.imshow(plotimg_wrp, cmap='gray')
    ax4.set_title('Warped Image', fontsize=16)
    
    # Axes 5: Binary warped image - overlaps either the centroids & windows
    # identified using sliding window search or the set of points that lie
    # within the margin-based search zone
    ax5.imshow(bwcimg, cmap='gray')
    ax5.set_title('Binary Warped Image with Centroids', fontsize=16)
    
    # Axes 6: Warped image with lane lines and margins superimposed
    ax6.imshow(wlrimg, cmap='gray')
    ax6.set_title('Warped Image Lines with Search Zone', fontsize=16)
    
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    #cv2.imwrite('lane_detection.jpg', mplfig_to_npimage(fig))
    #plt.imsave('lane_detection.jpg', mplfig_to_npimage(fig))
    #cv2.imshow('img',mplfig_to_npimage(fig))
    
    # Save the figure to output directory for backup or GIF creation & return
    fig.savefig('output_images/lane_detection{:.2f}.jpg'.format(t), bbox_inches='tight')
    output = mplfig_to_npimage(fig)
    # Following line prevents the figure echo in the IPython console
    plt.close(fig)
    
    return output


# <codecell> Script main code

# Prevent plot figure pop-up and print to IPython console
plt.ioff()
#mpl.use('Agg')

# Initialize left and right lane markings
left_line = Line()
right_line = Line()

# Markers for Project Video (50 secs)
# First bad pactch: (19.0, 25.0)
# Second bad patch: (37.0, 43.0)

# Markers for Challenge Video (16 secs)
# Until start of bridge: (0.0, 16.0)

# Markers for Harder Challenge Video (47 secs)
# Tree shadows: (10.0, 18.0)
# Glare & dash washout: (25.52, 37.08)
# Sharp U-turn: (37.08, 45.88)

file_basename = 'harder_challenge'

df_flag = False
debug_log = True
if debug_log:
    with open('log.txt', 'w') as myfile:
        myfile.write('\n\n ##### Log @ {} ##### \n\n'.format(datetime.now()))

src_vid = mpy.VideoFileClip(file_basename+'_video.mp4', audio=False)
start_time = 0.2 ; end_time = None
src_clip = src_vid.subclip(t_start=start_time, t_end=end_time)

num_drop_frames = 0
all_drop_frames = 0
# Budget 500ms sec worth dropped frames for contingencies (glare, shake, etc)
max_drop_frames = int(src_vid.fps*0.08)
# settings for project and harder are 0.5 of fps

dst_clip = mpy.VideoClip(make_frame, duration=src_clip.duration)
dst_clip.write_videofile(file_basename+'_output.mp4', fps=src_vid.fps)

print('Total dropped frames: {}'.format(all_drop_frames))

# Reset IPython console settings to original
plt.ion()

# Unused test code for implementation of restart functionality
# 1. Open previously saved history information
#restart = False
#if restart:
#    lndata = pickle.load(open('./linehist.p', 'rb'))
#    left_line = lndata['left_line']
#    right_line = lndata['right_line']
# 2. Write history (line) information to file
#pfile = open('linehist.p','wb')
#pickle.dump({'left_line':left_line, 'right_line':right_line}, pfile)


# <codecell>

# Alternately, use VirtualDub. Final GIF size is half of that made by MoviePy.
# 1. Open MP4 and set selection start (Home key) and end (End key) frames
# 2. Add a 2:1 reduction filter (Video -> Filters -> Add -> 2:1 Reduction)
#    to compress/reduce the GIF file size
# 3. Export selection to output gif using File -> Export -> Animated GIF...

import moviepy.editor as mpy
make_gif = False
if make_gif:
    vfile = mpy.VideoFileClip('project_video.mp4')
    vclip = vfile.subclip(t_start=15.04, t_end=18.24)
    #vclip = vclip.resize(0.5).crop(x1=50,x2=50,y1=50,y2=50)
    vclip.write_gif('project_video.gif', fps=vfile.fps, program='ffmpeg', opt='nq')
