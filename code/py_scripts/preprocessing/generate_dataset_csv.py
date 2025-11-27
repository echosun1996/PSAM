from collections import OrderedDict
import argparse
import os
from rich import print
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--img-path", type=str, default=None, help="image path")
parser.add_argument("--seg-path", type=str, default=None, help="seg path")
parser.add_argument("--csv-path", type=str, default=None, help="csv path")
args = parser.parse_args()


img_path = args.img_path
image_folder_name = img_path.split("/")[-1]

seg_path = args.seg_path
seg_folder_name = seg_path.split("/")[-1]

dataset_type = img_path.split("/")[-2]


csv_path = args.csv_path

# Read the Images list from img_path
images_list = os.listdir(img_path)

# Save the Images list to csv_path
id_i = 0
os.remove(csv_path) if os.path.exists(csv_path) else None

with open(csv_path, "w") as f:
    f.write("id" + "," + "image_path" + "," + "seg_path" + "\n")
    for image in tqdm(images_list, desc="Generating to: " + csv_path):
        if image.endswith(".jpg"):
            seg_name = image.replace(".jpg", "_segmentation.png")
            if not os.path.exists(os.path.join(seg_path, seg_name)):
                print(f"{seg_name} not found")
                exit(-1)
            f.write(
                str(id_i)
                + ","
                + (dataset_type + "/" + image_folder_name + "/" + image)
                + ","
                + (dataset_type + "/" + seg_folder_name + "/" + seg_name)
                + "\n"
            )
            id_i += 1
