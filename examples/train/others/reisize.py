from glob import glob
import cv2


filelist = glob("*.jpg")

for fname in filelist:
    img = cv2.imread(fname)
    img = cv2.resize(img, (160, 120))
    cv2.imwrite(fname, img)
