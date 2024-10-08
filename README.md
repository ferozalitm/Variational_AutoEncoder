Variational_AutoEncoder_MNIST.py

Author: T M Feroz Ali


Network architecture:

    Autoencoder and Decoder uses 3 layer n/w.
    
    Uses 9 dimensional Latent space.
    
    Encoder outputs mean and log of variance.
    
    All RelU non-linearity except Sigmoid for output layer
    


Other details:

    Loss: BCELoss(mean over all elements and batchsize)*28*28 + KL_loss_(mean_over_batchSize).

    Analyzes VAE latent space using PCA

    Generates new data(numbers) by sampling the VAE latent space

    Dataset: MNIST




Plot of train and test losses:

![alt text](https://github.com/ferozalitm/Variational_AutoEncoder/blob/main/Results/Loss.png)



Reconstructrion on training-set samples:

![alt text](https://github.com/ferozalitm/Variational_AutoEncoder/blob/main/Results/train_reconst-145.png)



Reconstructrion on test-set samples:

![alt text](https://github.com/ferozalitm/Variational_AutoEncoder/blob/main/Results/test_reconst-145.png)



Analysis of VAE latent space using PCA:

![alt text](https://github.com/ferozalitm/Variational_AutoEncoder/blob/main/Results/PCA_latentSpace-145.png)



New data generated/sampled from the VAE latent space:

![alt text](https://github.com/ferozalitm/Variational_AutoEncoder/blob/main/Results/sampled-145.png)
