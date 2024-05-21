import torch
import torch.nn as nn
import torch.nn.functional as F
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

def get_q_scale_symmetric(tensor, dtype=torch.int8):
  r_max = tensor.abs().max().item()
  q_max = torch.iinfo(dtype).max
  scale = r_max / q_max
  return scale

def linear_q_with_scale_and_zeropoint(tensor, scale, zeropoint, dtype=torch.int8):
    scaled_and_linear_shifted_tensor = tensor / scale + zeropoint
    rounded_tensor = torch.round(scaled_and_linear_shifted_tensor)

    # Make sure the quantized tensor is in the range of the min and max quantized value
    # Get min and max quantized value
    min_val = torch.iinfo(dtype).min
    max_val = torch.iinfo(dtype).max

    # Clamp the quantized tensor to the min and max quantized value
    q_tensor = rounded_tensor.clamp(min_val, max_val).to(dtype)
    return q_tensor

def linear_q_symmetric(tensor, dtype=torch.int8):
  scale = get_q_scale_symmetric(tensor, dtype)
  q_tensor = linear_q_with_scale_and_zeropoint(tensor, scale, 0, dtype)
  return q_tensor, scale

def quantize_linear_W8A32_without_bias(input, quantized_w, scale_w, zeropoint_w):
  # make sure the input is float32
  assert input.type() == torch.float32
  # make sure the quanted weights are int8
  assert quantized_w.type() == torch.int8
  # dequantize the weights
  dequantized_weights = quantized_w.float() * scale_w + zeropoint_w
  #make inference using linear layer
  output = torch.nn.functional.linear(input, dequantized_weights)
  return output

def w8_a16_forward(weight, input, scale, bias=None):
  # Cast weight to input type
  casted_weight = weight.to(input.dtype)
  # Perform matrix multiplication
  matrix_mul = F.linear(input, casted_weight) * scale
  # Add bias
  if bias is not None:
    matrix_mul += bias
  return matrix_mul

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                bias=True, dtype=torch.float32):
        super().__init__()
        
        self.register_buffer( 
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features)).to(torch.int8)
        )

        self.register_buffer( 
            "scale",
            torch.randn((out_features), dtype=dtype)
        )
        if bias:
            self.register_buffer( 
                "bias",
                torch.randn((1, out_features), dtype=dtype)
            )
        else:
            self.bias = None

    # Add quantizer
    def quantize(self, weights):
        # cast weight to float32
        w_fp32 = weights.to(torch.float32)
        # get the scale by getting absolute max value of the the last dimension
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)
        self.int8_weights = int8_weights

        self.scale = scales

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scale, self.bias)
    

# loop over torch.nn.Module children and replace them with a new module
def replace_linear_with_target_and_quantize(module, target_class, module_names_to_exclude):
  '''
  Replace all linear layers in a module with a target class.

  Parameters:
  - module: The module containing the linear layers to be replaced.
  - target_class: The target class to replace the linear layers with.
  - module_names_to_exclude: A list of module names to exclude from replacement.

  Returns:
  - The modified module with linear layers replaced by the target class.
  '''
  for name, child in module.named_children():
    if isinstance(child, torch.nn.Linear) and not any([x ==name for x in module_names_to_exclude]):
      # get old module bias
      old_bias = child.bias
      # retrieve the old weight
      old_weight = child.weight

      # create new module 
      new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)

      # replace current module name with new module
      setattr(module, name, new_module)

      # Once the old module is replaced above, we can now set the old weight to the new module
      # Get this new module, and quantize it's old weight
      getattr(module, name).quantize(old_weight)

      if old_bias is not None:
        # if old module had bias, set the bias of new module to the old bias
        getattr(module, name).bias.data = old_bias
    else:
      # Recursively apply the function to the child module for nested modules
      replace_linear_with_target_and_quantize(child, target_class, module_names_to_exclude)