#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

# -----------------------------
# Global epoch tracker
# -----------------------------
current_epoch = 0

# -----------------------------
# Image parameters
# -----------------------------
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# -----------------------------
# Generator
# -----------------------------
def build_generator():
    noise_shape = (100,)

    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)

# -----------------------------
# Discriminator
# -----------------------------
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

# -----------------------------
# Save generated images
# -----------------------------
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    os.makedirs("images20k", exist_ok=True)
    fig.savefig(f"images20k/mnist_{epoch}.png")
    plt.close()

# -----------------------------
# Save weights safely
# -----------------------------
def save_weights_safe(epoch):
    os.makedirs("saved_weights", exist_ok=True)
    generator.save_weights(f"saved_weights/generator_epoch_{epoch}.weights.h5")
    discriminator.save_weights(f"saved_weights/discriminator_epoch_{epoch}.weights.h5")
    print(f"\n✅ Weights saved successfully at epoch {epoch}")

# -----------------------------
# Training
# -----------------------------
def train(epochs, batch_size=128, save_interval=1000):
    global current_epoch

    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    half_batch = batch_size // 2

    for epoch in range(epochs):
        current_epoch = epoch

        # ---------------------
        # Train Discriminator
        # ---------------------
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        # Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))
        g_loss = combined.train_on_batch(noise, valid_y)

        print(
            f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] "
            f"[G loss: {g_loss:.4f}]"
        )

        if epoch % save_interval == 0:
            save_imgs(epoch)

# -----------------------------
# Build models
# -----------------------------
optimizer = Adam(0.0002, 0.5)

discriminator = build_discriminator()
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

generator = build_generator()

z = Input(shape=(100,))
img = generator(z)

discriminator.trainable = False
valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

GEN_PATH = "saved_weights/generator_epoch_20000.weights.h5"
DIS_PATH = "saved_weights/discriminator_epoch_20000.weights.h5"

if os.path.exists(GEN_PATH) and os.path.exists(DIS_PATH):
    generator.load_weights(GEN_PATH)
    discriminator.load_weights(DIS_PATH)
    print("✅ Weights loaded successfully")
else:
    print("ℹ️ No saved weights found. Training from scratch.")


# -----------------------------
# Run safely (Ctrl+C safe)
# -----------------------------
try:
    train(epochs=50_000, batch_size=32, save_interval=1000)

except KeyboardInterrupt:
    print("\n⛔ Training interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n❌ Training crashed due to error:\n{e}")

finally:
    print("\n💾 Saving generator and discriminator weights...")
    save_weights_safe(current_epoch)
    print("✅ Done. You can safely resume training later.")
