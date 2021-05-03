from verification_net import VerificationNet
from stacked_mnist import StackedMNISTData, DataMode

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


task = 2
stacked = False

latent_dim = 4
epochs = 15
tol = 0.8

gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=2048)
if stacked:
    StackedMNISTData(mode=DataMode.COLOR_FLOAT_COMPLETE, default_batch_size=2048)
imgshape = gen.get_random_batch(batch_size=9)[0].shape[1:]


#print(imgshape)

x_train, y_train = gen.get_full_data_set(training=True) 
x_test, y_test = gen.get_full_data_set(training=False)

if task == 3:    #anomaly detection
    gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=2048)
    x_train, y_train = gen.get_full_data_set(training=True) #for anomaly detection

mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = mnist_digits.astype("float32") / 255
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


encoder_inputs = keras.Input(shape=imgshape)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(7 * 7 * 64, activation="relu")(x)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def anomaly_detection(self, x, y, n):
        generated = self.call(x)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )

        #threshold = np.mean(loss) + np.std(loss) #set treshold (maybe use later)
        print(threshold)
        index = np.argsort(reconstruction_loss)
        #loss[index]
        anomalies = x_test[index,...][-n:]
        classes = y_test[index,...][-n:]
        top_losses = reconstruction_loss[index,...][-n:]
        #anomalies = tf.math.greater(loss, threshold) #må hente ut selve bildene
        return anomalies, classes, top_losses


"""
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
"""

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=epochs, batch_size=128)

# nn for testing
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)

# --- recreate images ---

if task==1:     #basic

    # -- reconstruct a set of images --
    reconstructions =9
    img, cls = gen.get_random_batch(batch_size=reconstructions)
    gen.plot_example(images=img, labels=cls)

    rec = vae(img)  #reconstruct images
    rec = np.array(rec)     #convert to numpy array
    gen.plot_example(images=rec, labels=cls)
    
    # -- predictability and accuracy --
    data = vae(x_test)
    data = np.array(data)   #convert to numpy array

    predictability, accuracy = net.check_predictability(data = data, correct_labels=y_test, tolerance=tol)
    print("predictability with a toleranse of ", tol, " is: ", predictability)
    print("accuracy with a toleranse of ", tol, " is: ", accuracy)

# --- generator ---

if task==2:    #generator
    #z = np.random.randint(2, size=(500,28,28,1))
    no_samples = 5000
    z = tf.random.normal(shape=[no_samples, latent_dim]) #np.random.randn(no_samples, latent_dim)
    generated = vae.decoder(z)

    generated = np.array(generated) #convert to numpy array for representation
    #print(generated.shape)

    # -- predictability --
    predictability, accuracy = net.check_predictability(data = generated, tolerance=tol)
    print("predictability with a toleranse of ", tol, " is: ", predictability)

    # -- class coverage --
    coverage = net.check_class_coverage(data = generated, tolerance=tol)
    print("coverage with a toleranse of ", tol, " is: ", coverage)

    # -- example images --
    z_ex = tf.random.normal(shape=[9, latent_dim]) # np.random.randn(9, latent_dim)
    #z_ex =  np.random.randint(2, size=(9,28,28,1))
    #print(z.shape)

    labels_ex = np.zeros(9)
    generated_ex = vae.decoder(z_ex)
    gen.plot_example(images=generated_ex, labels=labels_ex)


if task ==3: # anomaly detection
    #when using MISSING-data that lack the number 8
    #data = autoencoder(x_test)
    #data = np.array(data)
    n = 9
    anomalies, labels, top_losses = autoencoder.anomaly_detection(x_test, y_test, 9)
    print(x_test.shape)
    print(anomalies.shape)
    print(top_losses)

    #labels=np.zeros(n)
    print(labels.shape)
    gen.plot_example(images=anomalies, labels=labels)
    #anomalies = tf.expand_dims(anomalies, -1)
    #print(anomalies.shape)

    #if anomalies[0].shape>9:
    #    anomalies= anomalies[:9]

    #coverage = net.check_class_coverage(data = data, tolerance = tol)
    #print("coverage on missing set is: ", coverage)

    #extract examples
    no_classes_available = np.power(10, data.shape[-1])
    predictions, beliefs = net.predict(data=data)

    # Only keep predictions where all channels were legal
    anomaly = predictions[beliefs < tol]
    print(type(anomaly))
    anomaly = anomaly[:9]
    print(type(anomaly))
    print(anomaly.shape)
    labels=np.zeros(9)
    print(labels.shape)
    #gen.plot_example(images=anomaly, labels=labels)















