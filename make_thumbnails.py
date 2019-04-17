from PIL import Image
from resizeimage import resizeimage
import os

hansel = os.listdir('/Users/peternygaard/Desktop/github/shot_charts/images')
hansel.pop(1483)

for file in hansel:
    fd_img = open(file, 'rb')
    img = Image.open(fd_img)
    img = resizeimage.resize_thumbnail(img, [100, 100])
    img.save('thumbnails/thumbnail_{0}'.format(file), img.format)
    fd_img.close()
    print(file, len(hansel))
