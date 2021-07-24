# import the necessary packages
from helpers import pyramid
from helpers import sliding_window
from helpers import get_match
import glob
import json
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# DELETEEEEE ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--crops", required=True,
	help="crops image location")
ap.add_argument("-f", "--full", required=True,
	help="full image location")
args = vars(ap.parse_args())

# load the image and define the window width and height
image = cv2.imread(args["full"])
(winW, winH) = (round(image[0].size*.2), round(image[1].size*.15)) # Scales window according to full image resolution

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# Parameter set
    stepSize=round(.02*resized.shape[0])
    winW = round(resized[0].size*.04) # Scales window width according to full image resolution
    winH = round(resized[1].size*.04) # Scales window height according to full image resolution
    windowSize = (winW, winH)
    
    prev_y1 = 0
    rowNum = 1
    
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
        print(x, y) # DELEEEEEETE Just print coordinates
		# since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (180, 0, 255), 60) # ORIGINAL COLOR IS 0,255,0
        cloneResize = cv2.resize(clone, (1728, 972))
        cv2.imshow("Window", cloneResize)
        cv2.waitKey(1)
# <-- DELETE COMMENT        time.sleep(0.025)
        
        # Scans window for matched image
        # ================================================================
        # Grabs crop location provided as argument
        cropsLocation = args['crops']
        
        # Scans window and grabs cropped image coordinates relative to window
        win_x1, win_y1, win_x2, win_y2 = get_match(window, cropsLocation, x, y)
        
        # Saves cropped image and names with coordinates
        if win_x1 != "null":
            # Turns cropped image coordinates relative to window to full image coordinates
            x1 = x + win_x1
            y1 = y + win_y1
            x2 = x + win_x2
            y2 = y + win_y2
            
            if y1 >= (prev_y1 + round(stepSize/3) ) and y1 <= (prev_y1 - round(stepSize/3) ):
                rowNum += 1
            prev_y1 = y1
            croppedImage = window[win_y1:win_y2, win_x1:win_x2]
            cv2.imwrite("./Images/Cropped_Images/Row{}_x{}.jpg".format(y1, x1), croppedImage)
        # ================================================================
        

# Example argument
#   cd C:\Users\troya\.spyder-py3\Die_Crop_From_Wafer
#   python Sliding_Window.py -f Images/Images_to_Scan/Original.jpg -c Images/Images_to_Compare_for_Cropping/train_2.jpg