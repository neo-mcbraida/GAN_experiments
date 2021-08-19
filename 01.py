import tensorflow as tf
from PIL import Image
import os, os.path
import numpy as np
import matplotlib as plt
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.ops.gen_array_ops import InplaceAdd
import random

#image.resize((x, y))

x = 40
y = 40
inpX = 15
inpY = 15

path = "C:/Users/Neo/Documents/original_images"

noise_shape = (100,)

imagesShape = (1600, 3)

genModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=noise_shape),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(momentum=0.8),
    tf.keras.layers.Dense(3200, activation='relu'),
    tf.keras.layers.Dense(4800, activation='tanh'),
    tf.keras.layers.Reshape(imagesShape)
])

#optimizer = Adam(0.0002, 0.5)

genModel.compile(loss='binary_crossentropy', optimizer='adam')

discrimModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=imagesShape),
    tf.keras.layers.LayerNormalization(),
    #tf.keras.layers.InputLayer(input_shape=imagesShape),
    tf.keras.layers.Dense(1600, activation="relu"),
    tf.keras.layers.Dense(750, activation="relu"),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(512, activation="relu"),
    #tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    #tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    #tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

discrimModel.compile(loss='binary_crossentropy', optimizer='adam')
discrimModel.trainable = False

z = tf.keras.layers.Input(shape=noise_shape)
img = genModel(z)

valid = discrimModel(img)
combined = tf.keras.Model(z, valid)

combined.compile(loss='binary_crossentropy', optimizer='adam')
combined.summary()


def save_imgs(epoch):
        #r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, 100))
        #gen_imgs = genModel.predict(noise)

        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        noise = np.random.normal(0, 1, (1, 100))
        img = genModel.predict(noise)
        img = (img.tolist())[0]
        tempIm = [[]]

        u = 0
        for i in range(1600):
            if u == 40:
                u = 0
                tempIm.append([])
            pixel = img.pop(0)
            u += 1
            tempIm[len(tempIm) - 1].append(pixel)

        print("shaped")

        img = np.array(tempIm)

        img = Image.fromarray(img, 'RGB')
        name = "epoch" + str(epoch) + ".png"
        img.save(name)
        #img.show()
        #fig, axs = plt.subplots(r, c)
       # cnt = 0
       # for i in range(r):
        #    for j in range(c):
        #        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        ##        axs[i,j].axis('off')
        #        cnt += 1
       # fig.savefig("C:/Users/Neo/Documents/GAN_hw/imgsmnist_%d.png" % epoch)
       # plt.close()

def trainGan(epochs, batchSize, saveInterval):
    images = []
    
    # load images (jpgs)
    for f in os.listdir(path):
        image = (Image.open(os.path.join(path,f)))
        image = image.resize((40, 40))
        image = np.array(image.getdata())

        images.append(image)

    half_batch = int(batchSize / 2)

    #print((images[1]).shape)

    for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            indexes = np.random.randint(0, len(images), half_batch)
            #imgs = images[idx]
            imgs = np.array([images[i] for i in indexes])

            # noise = np.random.normal(0, 1, (half_batch, 100))
            noise = [[random.random() for u in range(100)] for i in range(half_batch)]
            noise = np.array(noise)
            # Generate a half batch of new images
            gen_imgs = np.array(genModel.predict(noise))

            # Train the discriminator
            ones = np.array([np.array([1]) for i in range(half_batch)])
            d_loss_real = discrimModel.train_on_batch(imgs, ones)
            #d_loss_fake = discrimModel.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            zeros = np.array([np.array([0]) for i in range(half_batch)])
            d_loss_fake = discrimModel.train_on_batch(gen_imgs, zeros)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            #noise = np.random.normal(0, 1, (batchSize, 100))

            noise = [[random.random() for u in range(100)] for i in range(batchSize)]
            noise = np.array(noise)
            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            # valid_y = np.array([0, 1] * batchSize)
            valid_y = np.array([[0] for i in range(batchSize)])
            # Train the generator
            g_loss = combined.train_on_batch(noise, valid_y)
            #combined.summary()
            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, d_loss, g_loss))

            # If at save interval => save generated image samples

            if epoch % saveInterval == 0:
                save_imgs(epoch)
                print("epoch:", epoch, "/", epochs, "g_loss:", g_loss, "d_loss:", d_loss)
            else:
                print("epoch:", epoch, "/", epochs)

trainGan(30000, 16, 64)


















        
    






