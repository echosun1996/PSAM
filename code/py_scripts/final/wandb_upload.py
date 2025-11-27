import time
import pandas as pd
import numpy as np
from scipy import stats
import wandb
import os
import re

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default=None, help="model name")
parser.add_argument("--metrics-csv", type=str, default=None, help="metrics csv file")
parser.add_argument("--timecost-txt", type=str, default=None, help="metrics txt file")
parser.add_argument("--images-num", type=int, help="number of images")
parser.add_argument("--comments", type=str, default=None, help="comments")
parser.add_argument("--labeller", type=int, default=None, help="labeller No.")

args = parser.parse_args()
comments = args.comments


def extract_first_bracket_content(text):
    # 使用正则表达式查找第一个方括号中的内容
    match = re.search(r"\[(.*?)\]", text)
    return match.group(1) if match else None


text_path = args.timecost_txt
text = open(text_path, "r").read()
images_num = args.images_num

os.environ["WANDB_SILENT"] = "true"

current_time = time.strftime("%m-%d %H:%M", time.localtime())
if comments is not None:
    upload_name = args.model_name + "[" + comments + "]-" + str(current_time)
else:
    upload_name = args.model_name + "-" + str(current_time)

model_name = args.model_name.split("-")[0]
dataset_name = args.model_name.split("-")[1]
wandb.init(
    project="PSAM-results",
    name=upload_name,
    tags=[model_name, dataset_name, "labeller=" + str(args.labeller)],
    notes=comments,
)


# 提取第一个方括号中的内容
extracted_content = extract_first_bracket_content(text)
if extracted_content is None:
    print(f"Error: No content found in {text_path}")
    exit(-1)

ave_timecost = float(extracted_content) / images_num

df = pd.read_csv(args.metrics_csv)
shape = df["Shape"][0]

numeric_df = df.select_dtypes(include="number")

means = numeric_df.mean()
variances = numeric_df.var()
std_devs = numeric_df.std()
confidence_intervals = {}

for column in numeric_df.columns:
    n = len(numeric_df[column])
    mean = means[column]
    std_err = stats.sem(numeric_df[column])
    h = std_err * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
    confidence_intervals[column] = (mean - h, mean + h)

table = wandb.Table(
    columns=[
        "Model",
        "Column",
        "Mean",
        "Variance",
        "Standard Deviation",
        "95% CI Lower",
        "95% CI Upper",
        "Shown",
        "Time",
    ]
)
# 添加数据到表格
for column in numeric_df.columns:
    table.add_data(
        upload_name + "-" + str(current_time),
        column,
        means[column],
        variances[column],
        std_devs[column],
        confidence_intervals[column][0],
        confidence_intervals[column][1],
        str(
            str(means[column].round(3))
            + "("
            + f"{confidence_intervals[column][0].round(3):.3f}"
            + "-"
            + f"{confidence_intervals[column][1].round(3):.3f}"
            + ")"
        ),
        f"{round(ave_timecost,3):.3f}",
    )
wandb.log(
    {
        "Name": upload_name,
        "Shape": shape,
        "Run Time": current_time,
        "Accuracy": means.get("Accuracy"),
        "Sensitivity": means.get("Sensitivity"),
        "Specificity": means.get("Specificity"),
        "Jaccard": means.get("Jaccard"),
        "Dice": means.get("Dice"),
        "Statistics": table,
        "Average Inference Time": ave_timecost,
    }
)
print(f"[√] {upload_name} upload success!")
wandb.finish()
