import glob
import random
from shutil import copy, move

source_folder = 'images/resized/'
file_path = glob.glob(source_folder + "/*.png")

from PIL import Image

for f in file_path:
    img = Image.open(f)
    print(img.size)
