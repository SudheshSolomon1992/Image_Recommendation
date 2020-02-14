# importing os module 
import os 
  
# Function to rename multiple files 
directory = "/media/sudhesh/ML/Object_Detection_Recommendation/resized_images/"
def main(): 
    i = 0
      
    for filename in os.listdir(directory): 
        old_filename = directory + filename
        new_filename = directory + str(filename).replace("png", "jpg")
        
        os.rename(old_filename, new_filename) 
        print ("Renamed" + old_filename)
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 