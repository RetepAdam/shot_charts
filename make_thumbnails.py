from PIL import Image
from resizeimage import resizeimage
import os

hansel = os.listdir('/Users/Peter/Desktop/github/shot_charts')
skips = [9820, 9654, 9425, 8597, 8438, 8266, 8157, 7581, 6830, 6494, 5522, 5097, 4545, 3139, 2529, 2495, 1594, 1484, 176]
for i in skips:
    hansel.pop(i)

for file in hansel:
    fd_img = open(file, 'r')
    img = Image.open(fd_img)
    img = resizeimage.resize_thumbnail(img, [200, 200])
    img.save('thumbnail_{0}'.format(file), img.format)
    fd_img.close()
    print(file, len(hansel))
