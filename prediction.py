#!/usr/bin/env python
import os
from datetime import datetime

__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
import h5py

# -----------------------------
# Image parameters
# -----------------------------
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# -----------------------------
# Generator (exact copy from gans.py)
# -----------------------------
def build_generator():
    noise_shape = (100,)

    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)

# Build the generator
generator = build_generator()
generator.build((None, 100))

# Manual weight loading
print("Loading weights manually...")
with h5py.File('generator_epoch_13389.weights.h5', 'r') as f:
    if 'sequential_1' in f.keys():
        seq_group = f['sequential_1']
        
        # Get the sequential model from the functional model
        seq_model = generator.layers[1]
        
        print(f"Available weight groups: {list(seq_group.keys())}")
        
        # Map saved weights to layers
        # Based on the output, weights are: dense_3, dense_4, dense_5, dense_6
        # and batch_normalization, batch_normalization_1, batch_normalization_2
        
        weight_mapping = {
            'dense_3': 0,      # First Dense(256)
            'batch_normalization': 2,  # First BatchNorm
            'dense_4': 3,      # Second Dense(512)
            'batch_normalization_1': 5,  # Second BatchNorm
            'dense_5': 6,      # Third Dense(1024)
            'batch_normalization_2': 8,  # Third BatchNorm
            'dense_6': 9       # Fourth Dense(784)
        }
        
        for weight_name, layer_idx in weight_mapping.items():
            if weight_name in seq_group:
                layer = seq_model.layers[layer_idx]
                layer_group = seq_group[weight_name]
                
                weight_names = list(layer_group.keys())
                print(f"Loading {weight_name} -> layer {layer_idx} ({layer.__class__.__name__})")
                
                if 'kernel:0' in layer_group and 'bias:0' in layer_group:
                    # Dense layer
                    kernel = np.array(layer_group['kernel:0'])
                    bias = np.array(layer_group['bias:0'])
                    layer.set_weights([kernel, bias])
                    print(f"  ✓ Loaded Dense weights: kernel {kernel.shape}, bias {bias.shape}")
                    
                elif 'gamma:0' in layer_group:
                    # BatchNormalization layer
                    gamma = np.array(layer_group['gamma:0'])
                    beta = np.array(layer_group['beta:0'])
                    moving_mean = np.array(layer_group['moving_mean:0'])
                    moving_variance = np.array(layer_group['moving_variance:0'])
                    layer.set_weights([gamma, beta, moving_mean, moving_variance])
                    print(f"  ✓ Loaded BatchNorm weights: gamma {gamma.shape}")

print("\n✅ All weights loaded successfully!\n")

# Generate multiple images
fig, axes = plt.subplots(5,2, figsize=(10, 7))
axes = axes.flatten()

for idx in range(10):
    vector = np.random.randn(1, 100)
    X = generator.predict(vector, verbose=0)
    X = 0.5 * X + 0.5
    
    axes[idx].imshow(X[0, :, :, 0], cmap='gray')
    axes[idx].axis('off')
    axes[idx].set_title(f'Generated #{idx+1}')

plt.tight_layout()
output_dir = r"L:\GANs\images"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"generated_digits{timestamp}.png")

plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Image saved as '{output_path}'")

print("✅ Image saved as 'generated_digits.png'")
plt.close()

print("\n🎉 Done! Check 'generated_digits.png' to see your generated MNIST digits!")