from verification_net import VerificationNet
from stacked_mnist import StackedMNISTData, DataMode

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#Tasks = 1="basic", 2="gen", 3="anomaly detection"
task = 3


latent_dim = 5
epochs = 3
tol = 0.8


gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
imgshape = gen.get_random_batch(batch_size=9)[0].shape[1:]


#print(imgshape)

x_train, y_train = gen.get_full_data_set(training=True) 
x_test, y_test = gen.get_full_data_set(training=False)

if task == 3:    #anomaly detection
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    x_train, y_train = gen.get_full_data_set(training=True) #for anomaly detection



class Autoencoder(Model):
    def __init__(self, imgshape):
        super(Autoencoder, self).__init__()
        self.imgshape = imgshape
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=imgshape),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
            ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7*7*32, activation='sigmoid'),
            layers.Reshape((7, 7, 32)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(imgshape[2], kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def anomaly_detection(self, x, y, n):
        generated = self.call(x)
        generated = np.array(generated)
        loss = losses.mean_squared_error(generated, x)
        loss = np.array(loss).mean(axis=(-1,-2))
        
        #threshold = np.mean(loss) + np.std(loss) #set treshold (maybe use later)
        print(threshold)
        index = np.argsort(loss)
        #loss[index]
        anomalies = x_test[index,...][-n:]
        classes = y_test[index,...][-n:]
        top_losses = loss[index,...][-n:]
        #anomalies = tf.math.greater(loss, threshold) #må hente ut selve bildene
        return anomalies, classes, top_losses




# ----- create models -------------


autoencoder = Autoencoder(imgshape)

#autoencoder.encoder.summary()
#autoencoder.decoder.summary()

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=64, shuffle=True, validation_data=(x_test, x_test))

# nn for testing
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)

# --- recreate images ---

if task==1:     #basic

    #reconstruct a set of images
    reconstructions =9
    img, cls = gen.get_random_batch(batch_size=reconstructions)
    gen.plot_example(images=img, labels=cls)
    

    rec = autoencoder(img)  #reconstruct images
    rec = np.array(rec)     #convert to numpy array

    gen.plot_example(images=rec, labels=cls)
    
    #predictability and accuracy
    data = autoencoder(x_test)
    data = np.array(data)   #convert to numpy array

    predictability, accuracy = net.check_predictability(data = data, correct_labels=y_test, tolerance=tol)
    print("predictability with a toleranse of ", tol, " is: ", predictability)
    print("accuracy with a toleranse of ", tol, " is: ", accuracy)


# --- generator ---

if task==2:    #generator
    #z = np.random.randint(2, size=(500,28,28,1))
    no_samples = 500
    z = tf.random.normal(shape=[no_samples, latent_dim]) #np.random.randn(no_samples, latent_dim)


    generated = autoencoder.decoder(z)

    generated = np.array(generated) #convert to numpy array for representation
    #print(generated.shape)

    #predictability
    predictability, accuracy = net.check_predictability(data = generated, tolerance=tol)
    print("predictability with a toleranse of ", tol, " is: ", predictability)

    #class coverage
    coverage = net.check_class_coverage(data = generated, tolerance=tol)
    print("coverage with a toleranse of ", tol, " is: ")
    print(coverage)

    #example images
    z_ex = tf.random.normal(shape=[num_examples_to_generate, latent_dim]) # np.random.randn(9, latent_dim)
    #z_ex =  np.random.randint(2, size=(9,28,28,1))
    #print(z.shape)

    labels_ex = np.zeros(9)
    generated_ex = autoencoder.decoder(z_ex)
    gen.plot_example(images=generated_ex, labels=labels_ex)



# --- anomaly detection ---
"""
def extract_anomalies(num, data, tol):
    if net.predict(data=data)[0]
    no_classes_available = np.power(10, data.shape[-1])
    predictions, beliefs = net.predict(data=data)

    # Only keep predictions where all channels were legal
    anomaly = predictions[beliefs < tol]
"""


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












