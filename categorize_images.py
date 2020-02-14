import os
import subprocess

path = "/media/sudhesh/ML/Object_Detection_Recommendation/resized_images"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    filename = str(f).split('/')[-1]
    base_sku = filename.split('-')[0]
    if len(base_sku) == 7:
        category = base_sku[:2]
    else:
        category = "misc"

    output_directory = "/media/sudhesh/ML/Object_Detection_Recommendation/Categorized_Images/" + category + "/"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        move_file = "cp " + f + " " +output_directory
        os.system (move_file)
    
    else:
        move_file = "cp " + f + " " +output_directory 
        os.system (move_file)

    
    