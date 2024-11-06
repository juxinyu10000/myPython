import cv2

img_url = r"C:\Users\¾ÏĞÂÓî\Pictures\1000\fishBig.jpg"
img = cv2.imread(img_url)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

