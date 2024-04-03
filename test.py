import pyximport
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pyximport.install()
import main

blurred_array = main.gaussian()  # Assuming gaussian() returns a NumPy array representing the blurred image
blurred_img = Image.fromarray(blurred_array.astype(np.uint8))
plt.imshow(blurred_img)
plt.show()
