import matplotlib.pyplot as plt
import numpy as np

# Data preparation
# labels = ["HAM10000\n(test set)", "ISIC2016", "ISIC2017", "PH2", "Dermofit", "STI Atlas"]
# stats = {
#     "nnUNet": [0.898, 0.868, 0.800, 0.864, 0.817, 0.704],
#     "ScribbleSaliency": [0.853, 0.826, 0.783, 0.894, 0.793, 0.470],
#     "Medical SAM": [0.770, 0.789, 0.770, 0.787, 0.687, 0.741],
#     "SAM": [0.642, 0.675, 0.637, 0.706, 0.599, 0.732],
#     "PSAM": [0.901, 0.890, 0.785, 0.882, 0.841, 0.769],
# }

# Data preparation - v2 only mantain "HAM10000", "ISIC2016", "Dermofit", "STI Atlas"
labels = ["HAM10000\n(test set)", "ISIC2016", "Dermofit", "STI Atlas"]
stats = {
    "nnUNet": [0.898, 0.868, 0.817, 0.704],
    "ScribbleSaliency": [0.853, 0.826, 0.793, 0.470],
    "Medical SAM": [0.770, 0.789, 0.687, 0.741],
    "SAM": [0.642, 0.675, 0.599, 0.732],
    "PSAM": [0.901, 0.890, 0.841, 0.769],
}

# Number of variables we're plotting
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The radar chart is a circle, so we need to "complete the loop"
angles += angles[:1]

# Setup the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Remove border (spines)
ax.spines["polar"].set_visible(False)

# Draw one axe per variable and add labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw the axes with the labels
plt.xticks(angles[:-1], labels, fontsize=25)


# Draw ylabels
ax.set_rlabel_position(30)
plt.yticks(
    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    color="grey",
    size=20,
)
ax.tick_params(axis="x", pad=30)  # 增加标签与轴的距离
ax.tick_params(axis="y", pad=30)
plt.ylim(0.4, 0.95)

# Plot each individual group's data with lighter shading
for name, values in stats.items():
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle="solid", label=name)
    ax.fill(angles, values, alpha=0.1)  # Set alpha to 0.1 for lighter shading

# Adjusting the legend to the bottom-right corner of the whole figure
fig.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), frameon=False, fontsize=20)

plt.tight_layout()

# Save the figure with the legend fully visible
plt.savefig("radar_chart-v2.png", bbox_inches="tight")
