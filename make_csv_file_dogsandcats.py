import os
import pandas as pd

# Change dir
#os.chdir(os.getcwd()+'/dataset')

# make a dataframe
train_DogCats = pd.DataFrame(columns=['file_name', 'target'])
# print(train_DogCats)

# load file names
filenames = os.listdir("C:/Users/all7j/PycharmProjects/datasets/cats_dogs/resized/")
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


print(df.head())
print(df.tail())
df.to_csv('DogsandCats.csv')