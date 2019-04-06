from PIL import Image
from os import listdir

def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.close()

if __name__ == '__main__':
    images = listdir('images')
    images.pop(1483)
    for i in images:
        print(i)
        image = 'images/{0}'.format(i)
        crop(image, (108, 95, 779, 694), '{0}'.format(i))
