from PIL import Image
from os import listdir

def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.close()

if __name__ == '__main__':
    hello = []
    for i in range(len(listdir())):
        if listdir()[i][-3:] == 'png':
            hello.append(i)
    for i in hello:
        print(i)
        image = '{0}'.format(listdir()[hello[i]])
        crop(image, (108, 95, 779, 694), '{0}'.format(listdir()[hello[i]]))
