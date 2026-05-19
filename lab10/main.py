import numpy as np
import cv2
import matplotlib as plt

photos = ['corgie.jpg','gory.jpg','kaktusy.jpg']

def JPG_comp(photo):
    img = cv2.imread(photo)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)

    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(decimg)

    cv2.imwrite(photo+'_1', decimg)

def main():
    for photo in photos:
        JPG_comp(photo)