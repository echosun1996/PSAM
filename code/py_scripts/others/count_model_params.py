import torch


def count_model_params(pth_file):
    # 加载 .pth 文件
    checkpoint = torch.load(pth_file, map_location=torch.device("cpu"))

    # 如果 checkpoint 中有 'state_dict'，就提取模型的状态字典
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 计算参数总量
    total_params = sum(p.numel() for p in state_dict.values())

    # 将参数量转换为百万
    total_params_in_millions = total_params / 1e6

    return total_params_in_millions


if __name__ == "__main__":
    print("Counting model parameters...")
    # Replace with your .pth file path
    import os
    parent_dir = os.environ.get('PSAM_PARENT_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    pth_file = f"{parent_dir}/code/compare_models/checkpoints/ScribbleSaliency/vgg16-397923af.pth"
    total_params = count_model_params(pth_file)
    print(
        f"ScribbleSaliency: Total number of parameters in {pth_file}: {total_params:.2f}M"
    )

    # Replace with your checkpoint path
    pth_file = f"{parent_dir}/code/compare_models/reps/OnePrompt/checkpoint/HAM10000_point_2024_07_26_14_14_30/Model/best_checkpoint"
    total_params = count_model_params(pth_file)
    print(f"OnePrompt: Total number of parameters in {pth_file}: {total_params:.2f}M")
