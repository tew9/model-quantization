import torch
import matplotlib.pyplot as plt
import seaborn as sns

def get_q_scale_and_zeropoint(tensor, dtype=torch.int8):
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    r_min = tensor.min().item()
    r_max = tensor.max().item()

    # get the scale
    scale = (r_max - r_min) / (q_max - q_min)
    # get the zero point
    zero_point = q_min - (r_min / scale)

    #handle zero point overflow
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = int(round(zero_point))
    return scale, zero_point

def plot_tensors(original_tensor, quantized_tensor, dequantized_tensor, error_tensor):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Plot original tensor
    sns.heatmap(original_tensor.numpy(), annot=True, fmt=".2f", ax=axs[0], cmap="coolwarm", cbar=False)
    axs[0].set_title("Original Tensor")

    # Plot quantized tensor
    sns.heatmap(quantized_tensor.numpy(), annot=True, fmt=".2f", ax=axs[1], cmap="coolwarm", cbar=False)
    axs[1].set_title("8-bit Linear Quantized Tensor")

    # Plot dequantized tensor
    sns.heatmap(dequantized_tensor.numpy(), annot=True, fmt=".2f", ax=axs[2], cmap="coolwarm", cbar=False)
    axs[2].set_title("Dequantized Tensor")

    # Plot dequantization error tensor
    sns.heatmap(error_tensor.numpy(), annot=True, fmt=".2f", ax=axs[3], cmap="coolwarm", cbar=False)
    axs[3].set_title("Quantization Error Tensor")

    plt.show()

def calculate_dequantization_error(original_tensor, dequantized_tensor):
    # Calculate the absolute dequantization error
    error_tensor = torch.abs(dequantized_tensor - original_tensor)
    return error_tensor

def linear_dequantizer(q_tensor, scale, zeropoint):
    return scale * (q_tensor.float() - zeropoint)