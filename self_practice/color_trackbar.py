import cv2
import numpy as np


def passFunction(*args):
    pass


def main():
    img = np.zeros((400, 512, 3), np.uint8)
    windowName = 'OpenCV BGR Color Generation'
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('B', windowName, 0, 255, passFunction)
    cv2.createTrackbar('G', windowName, 0, 255, passFunction)
    cv2.createTrackbar('R', windowName, 0, 255, passFunction)

    while True:
        cv2.imshow(windowName, img)

        if cv2.waitKey(1) == 27:
            break

        blue = cv2.getTrackbarPos('B', windowName)
        green = cv2.getTrackbarPos('G', windowName)
        red = cv2.getTrackbarPos('R', windowName)

        img[:] = [blue, green, red]

    # save the color
    cv2.imwrite('outColor.jpg', img)
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
