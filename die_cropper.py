# import the necessary packages
import imutils
import glob
import json
import argparse
import time
import cv2
import os
import numpy as np

# User Parameters/Constants to Set
MATCH_CL = 0.50 # Minimum confidence level (CL) required to match image to scanned image
SPLIT_MATCHES_CL =  0.80 # Splits MATCH_CL to SPLIT_MATCHES_CL (defects) to one folder, rest (no defects) other folder
FULL_IMAGE_PATH = "Images/Images_to_Scan/Original.jpg"
GOLDEN_IMAGE_PATH = "Images/Images_to_Compare_for_Cropping/toCompare.jpg"
SLEEP_TIME = 0.0 # Time to sleep in seconds between each window step

def deleteDirContents(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


def pyramid(image, compareCrop, scale=1.5, minSize=(500, 500)):
    # yield the original image
    yield (image, compareCrop)
    
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w1 = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w1)
        w2 = int(compareCrop.shape[1] / scale)
        compareCrop = imutils.resize(compareCrop, width=w2)
        
        # if the resized image does not meet the supplied minimum
        #     size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0] \
            or compareCrop.shape[0] < minSize[1] or compareCrop.shape[1] < minSize[0]:
            break
        
        # yield the next image in the pyramid
        yield (image, compareCrop)


def slidingWindow(image, stepSizeX, stepSizeY, windowSize): # stepSize normally 4 to 8 (pixels)
    # slide a window across the image
    for y in range(0, image.shape[0], stepSizeY):
        for x in range(0, image.shape[1], stepSizeX):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison Scan
def getMatch(window, compareCrop, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = compareCrop.shape
    
    if c1==c2 and h2<=h1 and w2<=w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, compareCrop, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > MATCH_CL: 
            print("\nFOUND MATCH")
            print("max_val = ", max_val)
            print("Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0]+w2, "y2:", y + max_loc[1]+h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")


# MAIN():
# =============================================================================
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Deletes contents in cropped and split folders
deleteDirContents("./Images/Cropped_Images/")
deleteDirContents("./Images/Splitted_Cropped_Images/No_Defects/")
deleteDirContents("./Images/Splitted_Cropped_Images/Potential_Defects/")

# load the full and comparing crop images
image = cv2.imread(FULL_IMAGE_PATH)
compareCrop = cv2.imread(GOLDEN_IMAGE_PATH)

# Predefine next for loop's parameters 
layer = 0 # Layer of resolution-downscaled

# loop over the image pyramid
for (resized, resizedCrop) in pyramid(image, compareCrop, scale=1.5):
    # Parameter set
    winW = round(resizedCrop.shape[1] * 1.5) # Scales window width according to full image resolution
    winH = round(resizedCrop.shape[0] * 1.5) # Scales window height according to full image resolution
    windowSize = (winW, winH)
    stepSizeX = round(winW / 3)
    stepSizeY = round(winH / 3)
    
    # Predefine next for loop's parameters 
    prev_y1 = 0
    prev_x1 = 0
    rowNum = 0
    colNum = 0
    prev_matchedCL = 0
    # TESTING BELOW
    BadX1 = 0
    BadY1 = 0
    BadX2 = 0
    BadY2 = 0
    
    # TESTING BELOW FOR SAVING IMAGE WITH BAD DIE BOXES
    clone2 = image.copy()
    
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in slidingWindow(resized, stepSizeX, stepSizeY, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        # Draw rectangle over sliding window for debugging and easier visual
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 180), 20)
        # TESTING BELOW
        # Add rect to failing area already saved
        cv2.rectangle(clone, (BadX1, BadY1), (BadX2, BadY2), (0, 100, 255), 20)
        cloneResize = cv2.resize(clone, (1728, 972))
        cv2.imshow("Window", cloneResize)
        cv2.waitKey(1)
        time.sleep(SLEEP_TIME) # sleep time in ms after each window step
        
        # Scans window for matched image
        # ==================================================================================
        # Scans window and grabs cropped image coordinates relative to window
        win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, resizedCrop, x, y)
        
        # Saves cropped image and names with coordinates
        if win_x1 != "null":
            # Turns cropped image coordinates relative to window to full image coordinates
            x1 = x + win_x1
            y1 = y + win_y1
            x2 = x + win_x2
            y2 = y + win_y2
            
            # Makes sure same image does not get saved as different names
            if y1 >= (prev_y1 + round(stepSizeY/3) ) or y1 <= (prev_y1 - round(stepSizeY/3)):
                rowNum += 1
                colNum = 1
                sameCol = False
            else:
                if x1 >= (prev_x1 + round(stepSizeX/3) ) or x1 <= (prev_x1 - round(stepSizeX/3)):
                    colNum += 1
                    prev_matchedCL = 0
                    sameCol = False
                else: 
                    sameCol = True
            
            # NEEDS A CHECK TO SEE IF FIRST X IN PREVIOUS Y-ROW IS THE SAME
            #   IF IT ISN'T, THEN MAKE PREVIOUS FIRST X IN PREVIOUS ROW
            #   HAVE A COLUMN_NUMBER += 1 AND DELETE OLD SAVE AND RESAVE
            #   WITH NEW NAME
            
            if (sameCol == False) or (sameCol == True and matchedCL > prev_matchedCL): 
                # Gets cropped image and saves cropped image
                croppedImage = window[win_y1:win_y2, win_x1:win_x2]
                cv2.imwrite("./Images/Cropped_Images/L{}-Row_{}-Col_{}.jpg".format(layer, rowNum, colNum), croppedImage)
                # Splits cropped images to folders with potential defects and no defects
                if matchedCL > SPLIT_MATCHES_CL:
                    cv2.imwrite("./Images/Splitted_Cropped_Images/No_Defects/L{}-Row_{}-Col_{}-CL_{}.jpg".format(layer, rowNum, colNum, round(matchedCL, 2)), croppedImage)
                else: 
                    # TESTING BELOW
                    BadX1 = x1
                    BadY1 = y1
                    BadX2 = x2
                    BadY2 = y2
                    cv2.imwrite("./Images/Splitted_Cropped_Images/Potential_Defects/L{}-Row_{}-Col_{}-CL_{}.jpg".format(layer, rowNum, colNum,  round(matchedCL, 2)), croppedImage)
                # If previous same Row and Column will be saved twice, deletes first one
                if sameCol == True and matchedCL > prev_matchedCL:
                    if "L{}-Row_{}-Col_{}-CL_{}.jpg".format(layer, rowNum, colNum, round(prev_matchedCL, 2)) in os.listdir("./Images/Splitted_Cropped_Images/No_Defects/"): 
                        os.remove("./Images/Splitted_Cropped_Images/No_Defects/L{}-Row_{}-Col_{}-CL_{}.jpg".format(layer, rowNum, colNum, round(prev_matchedCL, 2)))
                    if "L{}-Row_{}-Col_{}-CL_{}.jpg".format(layer, rowNum, colNum, round(prev_matchedCL, 2)) in os.listdir("./Images/Splitted_Cropped_Images/Potential_Defects/"): 
                        os.remove("./Images/Splitted_Cropped_Images/Potential_Defects/L{}-Row_{}-Col_{}-CL_{}.jpg".format(layer, rowNum, colNum, round(prev_matchedCL, 2)))
            
            prev_y1 = y1
            prev_x1 = x1
            prev_matchedCL = matchedCL
            
            # Draws orange boxes around bad dies
            if layer == 0:
                # Separate copy of resized with all bad dies showing orange boxes
                # TESTING BELOW
                # Add rect to failing area already saved
                cv2.rectangle(clone2, (BadX1, BadY1), (BadX2, BadY2), (0, 100, 255), 20)
        # ==================================================================================
    
    cv2.imwrite("./Images/Image_with_Failing_Dies_Overlay/ImageWithBoxes.jpg", clone2)
    layer += 1