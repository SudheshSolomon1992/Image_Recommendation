import os
from PIL import Image

path = '/media/sudhesh/ML/Visual Similarity/Image_Recommendation/images/'
output_path = '/media/sudhesh/ML/Visual Similarity/Image_Recommendation/resized_images/'
files = []

# Create directory if not exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
            files.append(os.path.join(r, file))

for f in files:
    print("Resizing " + str(f))
    try:
        filename = str(f).replace('jpg', 'png').split('/')[-1]
        img = Image.open(f) # image extension *.png,*.jpg
        new_width  = 800
        new_height = 800
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(output_path + filename) # format may what u want ,*.png,*jpg,*.gif
    except:
        print ("Exception for " + str(f))
        