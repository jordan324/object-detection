import os
import csv

# Read train.txt and limit to 500 lines
with open("output.txt", "r") as file:
    train_lines = file.readlines()[:500]

# Write train data to train.csv
with open("YOLO/data/train.csv", mode="w", newline="") as train_file:
    writer = csv.writer(train_file)
    for line in train_lines:
        image_file = line.strip().split("/")[-1]
        text_file = image_file.replace(".jpg", ".txt")
        writer.writerow([image_file, text_file])

# Read test.txt and limit to 500 lines
with open("output.txt", "r") as file:
    test_lines = file.readlines()[:500]

# Write test data to test.csv
with open("YOLO/data/test.csv", mode="w", newline="") as test_file:
    writer = csv.writer(test_file)
    for line in test_lines:
        image_file = line.strip().split("/")[-1]
        text_file = image_file.replace(".jpg", ".txt")
        writer.writerow([image_file, text_file])
