import cv2
import numpy as np

def image_preprocessing(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (960, 540))
    image = cv2.resize(image, (960, 540))
    return image, gray

def getAutoEdge(image, sigma=0.23):
    v = np.median(image)
    l = int(max(0, (1.0 - sigma) * v))
    u = int(min(255, (1.0 + sigma) * v))
    e = cv2.Canny(image, l, u)
    print("lower " + str(l))
    print("upper " + str(u))
    return e

def show_images(original, final):
    cv2.imshow("Original image", original)
    cv2.imshow("Detection image", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def skeletonize(canny_image):
    size = np.size(canny_image)
    skel = np.zeros(canny_image.shape, np.uint8)

    ret, img = cv2.threshold(canny_image, 127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def main():
    original_image, preprocessed_image = image_preprocessing("test_images_2022_11_18/20221118_215732.jpg")
    canny_edge_image = getAutoEdge(preprocessed_image)
    skel_image = skeletonize(canny_edge_image)
    show_images(original_image, skel_image)

if __name__ == "__main__":
    main()