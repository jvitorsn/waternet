import numpy as np
import cv2 as cv

def nothing(x):
    pass

# Create a black image, a window
img_src = cv.imread('./samples/section1/section1_full/frame_000075.jpg', cv.IMREAD_COLOR)
img_src = cv.cvtColor(img_src, cv.COLOR_BGR2RGB)
img_src = cv.resize(img_src, (img_src.shape[1]//2, img_src.shape[0]//2), interpolation=cv.INTER_AREA)
img = img_src.copy()
cv.namedWindow('image')
cv.resizeWindow('image', img.shape[1], img.shape[0])

# create trackbars for color change
cv.createTrackbar('gamma','image',0, 255,nothing)
cv.createTrackbar('gain_alpha','image',0, 255,nothing)
cv.createTrackbar('gain_beta','image',0, 255,nothing)
cv.createTrackbar('contrast','image',0, 255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)

def adjust_gamma(image, gamma=1.0):
    try:
        invGamma = 1.0 / gamma
    except:
        invGamma = 0.0001

    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def adjust_contrast(image, contrast=1.0):
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)

    return cv.addWeighted(image, alpha_c, image, 0, gamma_c)

def adjust_brightness(image, gain=1.0, beta=0):
    return cv.convertScaleAbs(image, alpha=gain, beta=beta)
    
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    gamma = cv.getTrackbarPos('gamma','image')
    gain_alpha = cv.getTrackbarPos('gain_alpha','image')
    gain_beta = cv.getTrackbarPos('gain_beta','image')
    contrast = cv.getTrackbarPos('contrast','image')
    s = cv.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = img_src
    else:
        img[:] = adjust_gamma(img_src, gamma/127.0)
        img = adjust_contrast(img, contrast - 127.0)
        img = adjust_brightness(img, gain_alpha/255.0, gain_beta/64.0)

cv.destroyAllWindows()