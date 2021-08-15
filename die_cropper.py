# import the necessary packages
import os
import glob
import imutils
import cv2
import time

# User Parameters/Constants to Set
MATCH_CL = 0.60 # Minimum confidence level (CL) required to match golden-image to scanned image
SPLIT_MATCHES_CL =  0.85 # Splits MATCH_CL to SPLIT_MATCHES_CL (defects) to one folder, rest (no defects) other folder
FULL_IMAGE_DIRECTORY = "Images/to_Scan_Image/"
GOLDEN_IMAGE_DIRECTORY = "Images/Golden_Image/"
SLEEP_TIME = 0.0 # Time to sleep in seconds between each window step


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


# PYRAMID NOT IN USE - It scales down images to crop downscale for ML
def pyramid(fullImage, goldenImage, scale=1.5, minSize = (500, 500)):
    # Yield the original full image
    yield (fullImage, goldenImage)
    
    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        w1 = int(fullImage.shape[1] / scale)
        resizedFullImage = imutils.resize(fullImage, width = w1)
        w2 = int(goldenImage.shape[1] / scale)
        resizedCropImage = imutils.resize(goldenImage, width = w2)
        
        # If the resized image does not meet the supplied minimum
        #     size, then stop constructing the pyramid
        if resizedFullImage.shape[0] < minSize[1] or resizedFullImage.shape[1] < minSize[0] \
            or resizedCropImage.shape[0] < minSize[1] or resizedCropImage.shape[1] < minSize[0]:
            break
        
        # yield the next image in the pyramid
        yield (resizedFullImage, resizedCropImage)


def slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # Slide a window across the resized full image
    for y in range(0, fullImage.shape[0], stepSizeY):
        for x in range(0, fullImage.shape[1], stepSizeX):
            # Yield the current window
            yield (x, y, fullImage[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison scan of scanning window-image to golden-image
def getMatch(window, goldenImage, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = goldenImage.shape
    
    if c1 == c2 and h2 <= h1 and w2 <= w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, goldenImage, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > MATCH_CL: 
            print("\nFOUND MATCH")
            print("max_val = ", max_val)
            print("Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0] + w2, "y2:", y + max_loc[1] + h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")


# MAIN():
# =============================================================================
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Deletes contents in cropped and split folders
deleteDirContents("./Images/Cropped_Die_Images/")
deleteDirContents("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/")
deleteDirContents("./Images/Splitted_Cropped_Die_Images/Potential_Defects/")
deleteDirContents("./Images/Failing_Dies_Overlayed_on_Wafer_Image")

# load the full and comparing crop images
fullImagePath = glob.glob(FULL_IMAGE_DIRECTORY + "*")
fullImage = cv2.imread(fullImagePath[0])
goldenImagePath = glob.glob(GOLDEN_IMAGE_DIRECTORY + "*")
goldenImage = cv2.imread(goldenImagePath[0])

# Parameter set
winW = round(goldenImage.shape[1] * 1.5) # Scales window width according to full image resolution
winH = round(goldenImage.shape[0] * 1.5) # Scales window height according to full image resolution
windowSize = (winW, winH)
stepSizeX = round(winW / 2.95)
stepSizeY = round(winH / 2.95)

# Predefine next for loop's parameters 
prev_y1 = stepSizeY * 9 # Number that prevents y = 0 = prev_y1
prev_x1 = stepSizeX * 9
rowNum = 0
colNum = 0
prev_matchedCL = 0
# TESTING BELOW
BadX1 = 0
BadY1 = 0
BadX2 = 0
BadY2 = 0

# TESTING BELOW FOR SAVING FULL IMAGE WITH BAD DIE BOXES
clone2 = fullImage.copy()

# loop over the sliding window
for (x, y, window) in slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    
    # Draw rectangle over sliding window for debugging and easier visual
    clone = fullImage.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 180), 30)
    # TESTING BELOW
    # Add rect to failing area already saved
    cv2.rectangle(clone, (BadX1, BadY1), (BadX2, BadY2), (0, 100, 255), 30)
    cloneResize = cv2.resize(clone, (round(fullImage.shape[1] / fullImage.shape[0] * 950), 950))
    cv2.imshow("Window", cloneResize)
    cv2.waitKey(1)
    time.sleep(SLEEP_TIME) # sleep time in ms after each window step
    
    # Scans window for matched image
    # ==================================================================================
    # Scans window and grabs cropped image coordinates relative to window
    # Uses each golden image in the file if multiple part types are present
    for goldenImagePath in glob.glob(GOLDEN_IMAGE_DIRECTORY + "*"):
        goldenImage = cv2.imread(goldenImagePath)
        win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, goldenImage, x, y)
        
        # Saves cropped image and names with coordinates
        if win_x1 != "null":
            # Turns cropped image coordinates relative to window to full image coordinates
            x1 = x + win_x1
            y1 = y + win_y1
            x2 = x + win_x2
            y2 = y + win_y2
            
            # Makes sure same image does not get saved as different names
            if y1 >= (prev_y1 + round(stepSizeY / 2.95) ) or y1 <= (prev_y1 - round(stepSizeY / 2.95)):
                rowNum += 1
                colNum = 1
                sameCol = False
            else:
                if x1 >= (prev_x1 + round(stepSizeX / 2.95) ) or x1 <= (prev_x1 - round(stepSizeX / 2.95)):
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
                cv2.imwrite("./Images/Cropped_Die_Images/Row_{}-Col_{}.jpg".format(rowNum, colNum), croppedImage)
                # Splits cropped images to folders with potential defects and no defects
                if matchedCL > SPLIT_MATCHES_CL:
                    cv2.imwrite("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/R{}-C{}-CL{}.jpg".format(rowNum, colNum, round(matchedCL * 100)), croppedImage)
                else: 
                    # TESTING BELOW
                    BadX1 = x1
                    BadY1 = y1
                    BadX2 = x2
                    BadY2 = y2
                    cv2.imwrite("./Images/Splitted_Cropped_Die_Images/Potential_Defects/R{}-C{}-CL{}.jpg".format(rowNum, colNum,  round(matchedCL * 100)), croppedImage)
                # If previous same Row and Column will be saved twice, deletes first one
                if sameCol == True and matchedCL > prev_matchedCL:
                    if "R{}-C{}-CL{}.jpg".format(rowNum, colNum, round(prev_matchedCL * 100)) in os.listdir("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/"): 
                        os.remove("./Images/Splitted_Cropped_Die_Images/Not_Likely_Defects/R{}-C{}-CL{}.jpg".format(rowNum, colNum, round(prev_matchedCL * 100)))
                    if "R{}-C{}-CL{}.jpg".format(rowNum, colNum, round(prev_matchedCL * 100)) in os.listdir("./Images/Splitted_Cropped_Die_Images/Potential_Defects/"): 
                        os.remove("./Images/Splitted_Cropped_Die_Images/Potential_Defects/R{}-C{}-CL{}.jpg".format(rowNum, colNum, round(prev_matchedCL * 100)))
            
            prev_y1 = y1
            prev_x1 = x1
            prev_matchedCL = matchedCL
            
            # Draws orange boxes around bad dies
            # Separate copy of resized full image with all bad dies showing orange boxes
            # TESTING BELOW
            # Add rect to failing area already saved
            cv2.rectangle(clone2, (BadX1, BadY1), (BadX2, BadY2), (0, 50, 255), 20)
        # ==================================================================================

# Saves window with orange boxes around potential bad dies
cv2.imwrite("./Images/Failing_Dies_Overlayed_on_Wafer_Image/ImageWithBoxes.jpg", clone2)