import os

# Directory containing the images
image_dir = "object-detection/data/images"

# Directory containing the label files
label_dir = "object-detection/data/labels"

# Output text file name
output_file = "output.txt"

# Get list of all files in the image directory
file_names = os.listdir(image_dir)

# Filter out directories and keep only filenames with corresponding labels
file_names_with_labels = []
for file_name in file_names:
    # Check if label file exists for the current image
    label_file = os.path.join(label_dir, file_name.replace(".jpg", ".txt"))
    if os.path.isfile(label_file):
        file_names_with_labels.append(file_name)

# Write file names to output text file
with open(output_file, "w") as f:
    for file_name in file_names_with_labels:
        f.write(file_name + "\n")

print(f"File names with labels written to {output_file}")

