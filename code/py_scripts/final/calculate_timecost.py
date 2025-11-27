import argparse
import pandas as pd
from rich import print

parser = argparse.ArgumentParser()

parser.add_argument("--input-csv-path", type=str, help="input csv path")
parser.add_argument("--output-txt-path", type=str, help="output txt path")
parser.add_argument("--images-num", type=int, help="sum of images")

args = parser.parse_args()

input_csv_path = args.input_csv_path
output_txt_path = args.output_txt_path
images_num = args.images_num


df = pd.read_csv(input_csv_path, header=0)
df = df.dropna()
row_sum = df.shape[0]
# 找到重复的行
duplicate_rows = df[df.duplicated(subset=["img_name"], keep=False)]
assert (
    len(duplicate_rows) == 0
), f"There are duplicate rows in the csv file:{duplicate_rows}"

assert (
    row_sum == images_num
), f"The sum_of_images [{images_num}] is not equal to the sum_of_rows [{row_sum}]"
total_timecost = df["timecost"].sum()
with open(output_txt_path, "w") as file:
    file.write(f"Total cost [{total_timecost}] seconds")
