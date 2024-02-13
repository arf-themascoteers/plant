import torch
import pandas as pd
import os

df = pd.read_csv("data/info.csv")
filenames = df["File.Name"]
csv_files = list(filenames)
csv_files = sorted(csv_files)

disk_files = os.listdir("data/images")

disk_files = sorted(disk_files)

print(f"CSV files {len(csv_files)}")
print(f"Disk files {len(disk_files)}")

csv_found = 0
csv_not_found = 0
disk_found = 0
disk_not_found = 0

for image in csv_files:
    image_png = f"{image}.png"
    if image_png in disk_files:
        csv_found = csv_found + 1
    else:
        csv_not_found = csv_not_found + 1
        print(image_png)

for image in disk_files:
    ind = image.index(".png")
    image_png = image[0:ind]
    if image_png in csv_files:
        disk_found = disk_found + 1
    else:
        disk_not_found = disk_not_found + 1
        print(image_png)

print(f"csv_found {csv_found}")
print(f"csv_not_found {csv_not_found}")
print(f"disk_found {disk_found}")
print(f"disk_not_found {disk_not_found}")