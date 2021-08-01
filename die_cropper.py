# import the necessary packages
import imutils
import glob
import json
import argparse
import time
import cv2

# Example argument
#   cd C:\Users\troya\.spyder-py3\Die_Crop_From_Wafer
#   python die_cropper.py -f Images/Images_to_Scan/Original.jpg -c Images/Images_to_Compare_for_Cropping/train_2.jpg
#   run die_cropper.py -f Images/Images_to_Scan/Original.jpg -c Images/Images_to_Compare_for_Cropping/train_2.jpg

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

def sliding_window(image, stepSize, windowSize): # stepSize normally 4 to 8 (pixels)
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



# Comparison Scan
# ======================
def get_match(window, compareCrop, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = compareCrop.shape
    
    if c1==c2 and h2<=h1 and w2<=w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, compareCrop, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val>0.65: # ORIGINALLY: 0.98
            print("\nFOUND MATCH")
            print("max_val = ", max_val)
            print("Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0]+w2, "y2:", y + max_loc[1]+h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2)
            # yield max_loc[0]
            # yield max_loc[1]
            # yield max_loc[0]+w2
            # yield max_loc[1]+h2 # Gets coordinates of cropped image
        
        else:
            return ("null", "null", "null", "null")
            # yield "null"
            # yield "null"
            # yield "null"
            # yield "null"


# def main(): # MAKE SURE TO INDENT THE REST
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# DELETEEEEE ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--compareCrops", required=True,
    help="crop image to compare to location")
ap.add_argument("-f", "--full", required=True,
    help="full image location")
args = vars(ap.parse_args())

# load the full and comparing crop images
image = cv2.imread(args["full"])
compareCrop = cv2.imread(args["compareCrops"])

windowSizeFraction = 0.06 # What fraction of full image the window should be
stepSizeFraction = 0.02 # What fraction of full image the window should step by

# Predefine next for loop's parameters 
layer = 0 # Layer of resolution-downscaled

# loop over the image pyramid
for (resized, resizedCrop) in pyramid(image, compareCrop, scale=1.5):
    # Parameter set
    stepSize=round(resized.shape[1] * stepSizeFraction)
    winW = round(resized.shape[1] * windowSizeFraction) # Scales window width according to full image resolution
    winH = round(resized.shape[1] * windowSizeFraction) # Scales window height according to full image resolution
    windowSize = (winW, winH)
    
    # Predefine next for loop's parameters 
    prev_y1 = 0
    prev_x1 = 0
    rowNum = 0
    colNum = 0
    
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        # since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 180), 20) # ORIGINAL COLOR IS 0,255,0
        # Add rect to area already saved
        clone = cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 20)
        cloneResize = cv2.resize(clone, (1728, 972))
        cv2.imshow("Window", cloneResize)
        cv2.waitKey(1)
        # time.sleep(0.5) # sleep time in ms after each window step
        
        # Scans window for matched image
        # ==================================================================================
        # Scans window and grabs cropped image coordinates relative to window
        win_x1, win_y1, win_x2, win_y2 = get_match(window, resizedCrop, x, y)
        
        # Saves cropped image and names with coordinates
        if win_x1 != "null":
            # Turns cropped image coordinates relative to window to full image coordinates
            x1 = x + win_x1
            y1 = y + win_y1
            x2 = x + win_x2
            y2 = y + win_y2
            
            # Makes sure same image does not get saved as different names
            if y1 >= (prev_y1 + round(stepSize/3) ) or y1 <= (prev_y1 - round(stepSize/3) ):
                rowNum += 1
                colNum = 1
            else:
                if x1 >= (prev_x1 + round(stepSize/3) ) or x1 <= (prev_x1 - round(stepSize/3) ):
                    colNum += 1
            
            # NEEDS A CHECK TO SEE IF FIRST X IN PREVIOUS Y-ROW IS THE SAME
            #   IF IT ISN'T, THEN MAKE PREVIOUS FIRST X IN PREVIOUS ROW
            #   HAVE A COLUMN_NUMBER += 1 AND DELETE OLD SAVE AND RESAVE
            #   WITH NEW NAME
            
            # Gets cropped image and saves cropped image
            croppedImage = window[win_y1:win_y2, win_x1:win_x2]
            cv2.imwrite("./Images/Cropped_Images/L{}-Row_{}-Col_{}.jpg".format(layer, rowNum, colNum), croppedImage)
            
            prev_y1 = y1
            prev_x1 = x1
            
        # ==================================================================================
    
    layer += 1


# if __name__ == "__main__":
#     main()
    
    
    