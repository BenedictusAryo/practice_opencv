import cv2
import numpy as np


def passFuntion(*args):
    pass


def do_scalling(scaleFactor, scaletype):
    if scaletype == 0:
        scaleFactor = 1 + scaleFactor/100.0
    if scaletype == 1:
        scaleFactor = 1 - scaleFactor/100.0
    return scaleFactor


def main():
    img = np.zeros((400, 450), np.uint8)
    windowName = 'OpenCV Image Resizer'
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('Size', windowName, 1, 100, passFuntion)
    cv2.createTrackbar('Type', windowName, 0, 1, passFuntion)

    while True:

        scaleType = cv2.getTrackbarPos('Type', windowName)
        scaleFactor = cv2.getTrackbarPos('Size', windowName)
        scaleFactor = do_scalling(scaleFactor, scaleType)

        scaledImg = cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor,
                               interpolation=cv2.INTER_LINEAR)
        cv2.imshow(windowName, scaledImg)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
