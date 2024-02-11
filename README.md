# CVAE
Model that Generates inputs for particular outputs in a sinosoidal function using CVAE

## Algorithm:
An autoencoder is a type of neural network engineered to autonomously grasp the inherent identity of its input without explicit guidance, enabling it to reconstruct the initial data while concurrently condensing the information, thereby revealing a more streamlined and compressed depiction.
It consists of two networks:

Encoder network: It translates the original high-dimension input into the latent low-dimensional code. The input size is larger than the output size.
Decoder network: The decoder network recovers the data from the code, likely with larger and larger output layers.
The encoder network essentially accomplishes the dimensionality reduction, just like how we would use Principal Component Analysis (PCA) or Matrix Factorization (MF)
The model contains an encoder function g(.) parameterized by phi and a decoder function f(.) parameterized by theta. The low-dimensional code learned for input x in the bottleneck layer is z = g(x) and the reconstructed input is x’=f(g(x).

VAE, is actually less similar to all the autoencoder models above, but deeply rooted in the methods of variational bayesian and graphical model.

Instead of mapping the input into a fixed vector z, we want to map it into a distribution.The relationship between the data input x and the latent encoding vector x can be fully defined by:

1. Prior - P(z)
2. Likelihood - P (x/z)
3. Posterior - P (z/x)

Conditional Variational Autoencoders take a different turn and label the input and output data with a labelled given input .


The following is the architecture behind CVAE:

![architecture](vae-gaussian.png)
