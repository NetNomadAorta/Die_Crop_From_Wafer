# Import the necessary packages
import os
import glob
import imutils
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,100000).__str__()
import cv2
import time
# TESTING SVD FROM NUMPY
import numpy as np

# User Parameters/Constants to Set
MATCH_CL = 0.60 # Minimum confidence level (CL) required to match golden-image to scanned image
SPLIT_MATCHES_CL =  0.98 # Splits MATCH_CL to SPLIT_MATCHES_CL (defects) to one folder, rest (no defects) other folder
STICHED_IMAGES_DIRECTORY = "Images/Stitched_Images/"
GOLDEN_IMAGES_DIRECTORY = "Images/Golden_Images/"
SLEEP_TIME = 0.5 # Time to sleep in seconds between each window step
SHOW_IMAGE_CROPPING = True


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


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
    # Slides a window across the stitched-image
    for y in range(0, fullImage.shape[0], stepSizeY):
        for x in range(0, fullImage.shape[1], stepSizeX):
            # Yield the current window
            yield (x, y, fullImage[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison scan window-image to golden-image
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
            print("Window Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0] + w2, "y2:", y + max_loc[1] + h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")


# MAIN():
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Clears some of the screen for asthetics
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Deletes contents in cropped- and split-image folders
deleteDirContents("./Images/Cropped_Die_Images/")
# deleteDirContents("./Images/SVD_Cropped_Die_Images/")

goldenImagePath = glob.glob(GOLDEN_IMAGES_DIRECTORY + "*")
goldenImage = cv2.imread(goldenImagePath[0])



# Parameter set
winW = round(goldenImage.shape[1] * 1.5) # Scales window width with full image resolution
# BELOW DEFAULT IS 1.5 CHANGE BACK IF NEEDED
winH = round(goldenImage.shape[0] * 1.5) # Scales window height with full image resolution
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
prev_BadX1 = 0
prev_BadY1 = 0
prev_BadX2 = 0
prev_BadY2 = 0
# BELOW IS FINE
BadX1 = 0
BadY1 = 0
BadX2 = 0
BadY2 = 0

for fullImagePath in glob.glob(STICHED_IMAGES_DIRECTORY + "*"):
    fullImage = cv2.imread(fullImagePath)

    # TESTING BELOW FOR SAVING FULL IMAGE WITH BAD DIE BOXES
    fullImageClone = fullImage.copy()
    
    cv2.destroyAllWindows()
    
    # loop over the sliding window
    for (x, y, window) in slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        # Draw rectangle over sliding window for debugging and easier visual
        displayImage = fullImage.copy()
        cv2.rectangle(displayImage, (x, y), (x + winW, y + winH), (255, 0, 180), 40)
        # TESTING BELOW
        # Add rect to failing area already saved
        cv2.rectangle(displayImage, (BadX1, BadY1), (BadX2, BadY2), (0, 100, 255), 10)
        displayImageResize = cv2.resize(displayImage, (1000, round(fullImage.shape[0] / fullImage.shape[1] * 1000)))
        if SHOW_IMAGE_CROPPING:
            cv2.imshow(str(fullImagePath), displayImageResize) # TOGGLE TO SHOW OR NOT
        cv2.waitKey(1)
        time.sleep(SLEEP_TIME) # sleep time in ms after each window step
        
        # Scans window for matched image
        # ==================================================================================
        # Scans window and grabs cropped image coordinates relative to window
        # Uses each golden image in the file if multiple part types are present
        for goldenImagePath in glob.glob(GOLDEN_IMAGES_DIRECTORY + "*"):
            goldenImage = cv2.imread(goldenImagePath)
            # Gets coordinates relative to window of matched dies within a Stitched-Image
            win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, goldenImage, x, y)
            
            # Saves cropped image and names with coordinates
            if win_x1 != "null":
                # Turns cropped image coordinates relative to window to stitched-image coordinates
                x1 = x + win_x1
                y1 = y + win_y1
                x2 = x + win_x2
                y2 = y + win_y2
                
                # Makes sure same image does not get saved as different names
                if (y1 >= (prev_y1 + round(goldenImage.shape[0] * .9) ) 
                    or y1 <= (prev_y1 - round(goldenImage.shape[0] * .9) ) ):
                    rowNum += 1
                    colNum = 1
                    prev_matchedCL = 0
                    sameCol = False
                else:
                    if x1 >= (prev_x1 + round(goldenImage.shape[1] * .9) ):
                        colNum += 1
                        prev_matchedCL = 0
                        sameCol = False
                    else: 
                        sameCol = True
                
                # NEEDS A CHECK TO SEE IF FIRST X IN PREVIOUS Y-ROW IS THE SAME
                #   IF IT ISN'T, THEN MAKE PREVIOUS FIRST X IN PREVIOUS ROW
                #   HAVE A COLUMN_NUMBER += 1 AND DELETE OLD SAVE AND RESAVE
                #   WITH NEW NAME
                
                # Puts 0 in front of single digit row nad column number
                if rowNum < 10:
                    rZ = 0
                else: 
                    rZ = ""
                if colNum < 10:
                    cZ = 0
                else: 
                    cZ = ""
                
                if (sameCol == False) or (sameCol == True and matchedCL > prev_matchedCL): 
                    # Gets cropped image and saves cropped image
                    croppedImage = window[win_y1:win_y2, win_x1:win_x2]
                    
                    # SAVES CROPPED TO NEW FOLDER WITH CROP NAME
                    cv2.imwrite("./Images/Cropped_Die_Images" +\
                        "/Row_{}{}-Col_{}{}.jpg".format(rZ, rowNum, cZ, colNum), croppedImage)
                
                if sameCol == False: 
                    prev_y1 = y1
                    prev_x1 = x1
                    prev_matchedCL = matchedCL
                    
                elif sameCol == True and matchedCL > prev_matchedCL:
                    prev_y1 = y1
                    prev_x1 = x1
                    prev_matchedCL = matchedCL
                
                
                # Draws orange boxes around bad dies
                # Separate copy of resized full image with all bad dies showing orange boxes
                # TESTING BELOW
                # Saves backup of fullImageClone before next rectangle write
                fullImageClone_Backup = fullImageClone.copy()
                # Add rect to failing area already saved
                cv2.rectangle(fullImageClone, (BadX1, BadY1), (BadX2, BadY2), (0, 50, 255), 35)
                prev_BadX1 = BadX1
                prev_BadY1 = BadY1
                prev_BadX2 = BadX2
                prev_BadY2 = BadY2
            # ==================================================================================
    rowNum += 1
    colNum = 0
    sameCol = False
    
    # Saves window with orange boxes around potential bad dies



    # cv2.imwrite("./Images/Failing_Dies_Overlayed_on_Wafer_Image/ImageWithBoxes.jpg", fullImageClone)
    # # TESTING BELOW
    # # SAVES Overlay Window TO NEW FOLDER WITH Stitched-Image NAME
    # os.makedirs("./Images/" + slotDir[23:-3] + "/" + \
    #     slotDir[-2:] + "/Failing_Dies_Overlayed_on_Wafer_Image", exist_ok=True)
    # cv2.imwrite("./Images/" + slotDir[23:-3] + "/" + \
    #     slotDir[-2:] + "/Failing_Dies_Overlayed_on_Wafer_Image" +\
    #     "/ImageWithBoxes.jpg", fullImageClone)



print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)