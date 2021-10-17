import cv2


import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


# Store all photos in Array
imageArray = [cv2.imread(file) for file in glob.glob('Pictures/*.jpg')]
print(len(imageArray))


# Stack function
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# Example to use stackfunction
#imgStack = stackImages(0.5, ([imageArray[0], imageArray[1], imageArray[2], imageArray[3], imageArray[4]], [imageArray[5],
                            #imageArray[6], imageArray[7], imageArray[8], imageArray[9]], [imageArray[10],
                            #imageArray[11], imageArray[12], imageArray[13], imageArray[14]], [imageArray[15],
                            #imageArray[16], imageArray[17], imageArray[18], imageArray[19]], [imageArray[20],
                            #imageArray[21], imageArray[22], imageArray[23], imageArray[24]]))

imgStack = stackImages(0.2, ([imageArray[0]]))

# Display the stack of images
#cv2.imshow('Original img stack', imgStack)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


def task2_3():
    img_blur_stack = cv2.GaussianBlur(imgStack, (3, 3), 0)

    sobelxy = cv2.Sobel(src=img_blur_stack, ddepth=cv2.CV_8U,
                        dx=1, dy=1, ksize=5)

    cv2.imshow("Sobel X Y", sobelxy)


    edges = cv2.Canny(image=img_blur_stack, threshold1=100, threshold2=200)

    cv2.imshow("Canny Edge 100-200", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



task2_3()


def task3():
    src = cv2.imread("Pictures/20211011_212722.jpg")

    src = cv2.pyrDown(src, dstsize=(2 // src.shape[1], 2 // src.shape[0]))
    src = cv2.pyrDown(src, dstsize=(2 // src.shape[1], 2 // src.shape[0]))
    src = cv2.pyrUp(src, dstsize=(2 * src.shape[1], 2 * src.shape[0]))
    src = cv2.pyrUp(src, dstsize=(2 * src.shape[1], 2 * src.shape[0]))

    cv2.imshow("Pyramid Scaled", src)
    cv2.waitKey(0)


#task3()


def task4():
    img = cv2.imread("Pictures/20211011_212722.jpg", 0)
    img2 = img.copy()

    methods = ['cv2.TM_CCOEFF']

    template = cv2.imread('Pictures/Wolf.jpg', 0)
    w, h = template.shape[::-1]

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()


#task4()
