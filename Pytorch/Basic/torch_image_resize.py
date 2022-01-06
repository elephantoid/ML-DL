from PIL import Image
import os
from tqdm import tqdm
root_dir='C:/Users/all7j/PycharmProjects/torch_tutorials/dataset/train/train/'
file_list= os.listdir(root_dir)
save_path='C:/Users/all7j/PycharmProjects/torch_tutorials/dataset/cats_dogs/resized/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

print("Resizing... ")
for file in tqdm(file_list):

    image=Image.open(root_dir+file)
    image=image.resize((256, 256))
    image=image.rotate(-90)

    # Change format JPG to PNG
    file = file[:-3] + "png"
    save_point= save_path+file

    image.save(save_point)
print('Done')