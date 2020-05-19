import cv2
import numpy as np

# Feladat dominok detektalasa


# Kivagja a megtalalt dominokat es megjeleniti oket kulon ablakban
def getSubImage(rect, src):

    # Get center, size, and angle from rect
    center, size, theta = rect

    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)

    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)

    # Rotate to horizontal
    height, width = out.shape[:2]
    if height > width:
        out = np.rot90(out)

    return out

# Konturok megkeresese
def getcontours(img):

    gamma = 1.8
    if gamma < 1:
        gamma = -1.0 / (gamma - 2.0)
    print("======================")
    print("Gamma value:", gamma)
    print("======================")

    temp = cv2.normalize(img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    gamma_corr = temp ** gamma
    out = cv2.normalize(gamma_corr, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    box = []
    dominos = []

    # kiszuri a tul kicsi talalt objektumokat
    for cntrIdx in range(0, len(contours)):
        if 3500 < cv2.contourArea(contours[cntrIdx]):       # filter out small 'noise'
            print("Area: " + str(cv2.contourArea(contours[cntrIdx])))
            rect = cv2.minAreaRect(contours[cntrIdx])
            rectbox = cv2.boxPoints(rect)
            box.append(np.int0(rectbox))
            dominos.append(getSubImage(rect, img))



    print('Kontúrok száma:', len(box))

    cv2.drawContours(img, box, -1, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)

    limit = len(dominos)

    for i in range(limit):          #  change the limit to show fewer cropped dominos
        cv2.imshow("Domino " + str(i + 1), dominos[i])
        cv2.waitKey(0)


# innen indul a program
if __name__ == '__main__':
    images = []
    images.append(cv2.imread('domino_1.jpg'))
    images.append(cv2.imread('domino_2.jpg'))

    for img in images:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        getcontours(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
