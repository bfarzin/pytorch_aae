# Pytorch Adversarial Autoencoders
Replicated the results from [this blog post](https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/) using PyTorch.

Autoencoders can be used to reduce dimensionality in the data.  This example uses the Encoder to fit the data (unsupervised step) and then uses the encoder representation as "features" to train the labels.

The result is not as good as using the raw features with a simple NN.  This example is designed to demonstrate the workflow for AAE and using that as features for a supervised step.


