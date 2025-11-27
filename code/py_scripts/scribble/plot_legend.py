import matplotlib.pyplot as plt


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-path", type=str, default=None, help="scibble output path")

args = parser.parse_args()

# 创建图形和轴
fig, ax = plt.subplots()

# 创建虚拟的线条对象，仅用于图例，不实际绘制
ax.plot([], [], color=[0, 1, 0], label="ground truth")
ax.plot([], [], color="red", label="PSAM output")
ax.plot([], [], color=[1, 1, 0], label="positive scribble")
ax.plot([], [], color=[0.5, 0, 0.5], label="negative scribble")
# 添加图例，图例显示在图形中央，四个图例分布在同一行
ax.legend(loc="center", bbox_to_anchor=(0.5, 0.5), ncol=4)

# 隐藏坐标轴
ax.axis("off")
# 显示图像
# plt.show()
# 调整子图边距，确保图例不会被截断
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.savefig(args.output_path, bbox_inches="tight")
