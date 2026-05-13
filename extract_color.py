#
# This program is used to extract unique colors from images
# and plot them in a visual 3D graph.
#

import sys
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Converts RGB to Hex code.
def convert_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

# Extracts the colors and assigns them to a list.
# Makes sure that all the colors are unique.
def extract(img_np):
    x = []
    y = []
    z = []
    color = []
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            hex = convert_to_hex(img_np[i][j])
            if hex not in color:
                color.append(hex)
                x.append(img_np[i][j][0])
                y.append(img_np[i][j][1])
                z.append(img_np[i][j][2])
    
    return x, y, z, color
    

def main(argv):
    fig = plt.figure(figsize=(12, 5))
    
    # Creates a subplot for each image
    for i in range(len(argv)):    
        img = Image.open(argv[i]).convert("RGB")
        img_np = np.asarray(img)
        
        x, y, z, color = extract(img_np)
        
        ax = fig.add_subplot(1, len(argv), i + 1, projection='3d')
        ax.scatter(x, y, z, c=color)
        ax.set_title(argv[i])
        ax.set_xlabel('red')
        ax.set_ylabel('green')
        ax.set_zlabel('blue')
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
    
    plt.savefig("plots.png")
                
if __name__ == "__main__":
    main(sys.argv[1:])