import os
import pandas as pd

# Change dir
#os.chdir(os.getcwd()+'/dataset')

# make a dataframe
train_DogCats = pd.DataFrame(columns=['file_name', 'target'])
# print(train_DogCats)

# load file names
filenames = os.listdir(os.getcwd()+'/dataset/train/train')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


print(train_DogCats.head())
print(train_DogCats.tail())
train_DogCats.to_csv('DogandCat.csv')