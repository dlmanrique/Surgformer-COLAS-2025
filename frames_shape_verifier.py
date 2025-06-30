import glob
import os
from PIL import Image
from tqdm import tqdm

# Load all possible paths
images_paths = glob.glob(os.path.join('Cholec80', 'frames_cutmargin', '**', '*.jpg')) 

# Iterate over each path, open the image and verify shape = (250, 250,3)

for image_pth in tqdm(images_paths):
    img = Image.open(image_pth)
    if img.size != (250,250):
        breakpoint()