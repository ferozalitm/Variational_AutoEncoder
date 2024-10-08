#  Variational_AutoEncoder_MNIST.py
#     Author: T M Feroz Ali
#     Network architecture:
#          Autoencoder and Decoder uses 3 layer n/w.
#          Uses 9 dimensional Latent space.
#          Encoder outputs mean and log of variance.
#          All RelU non-linearity except Sigmoid for output layer
#
#     Loss = BCELoss(mean over all elements and batchsize)*28*28 + KL_loss_(mean_over_batchSize).
#     Analyzes VAE latent space using PCA
#     Generates new data(numbers) by sampling the VAE latent space



import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import random
from torchvision.utils import save_image
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.decomposition import PCA
import matplotlib.cm as cm

no_classes = 10
colors = cm.rainbow(np.linspace(0, 1, no_classes))

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
ip_dimn = 28*28
batch_size = 256

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = './VAE/VAE_results/Feroz/9dimnLatentSpace/BCE_loss_reductionMean_Generate_LatentPCA_logSigma'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

train_dataset = torchvision.datasets.MNIST(root='../data/',
                                     train=True, 
                                     transform=transforms.ToTensor(),
                                     download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                     train=False, 
                                     transform=transforms.ToTensor(),
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

no_batches_train = len(train_loader)
no_batches_tst = len(test_loader)
print(f"No_batches train: {no_batches_train}")
print(f"No_batches test: {no_batches_tst}")

# Build a fully connected layer and forward pass
class VAE_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.linear1 = nn.Linear(in_features=ip_dimn, out_features=14*14)
        self.linear2 = nn.Linear(in_features=14*14, out_features=7*7)
        self.linear3a = nn.Linear(in_features=7*7, out_features=3*3)
        self.linear3b = nn.Linear(in_features=7*7, out_features=3*3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3a = nn.ReLU()
        self.relu3b = nn.ReLU()

        # Decoder
        self.linear4 = nn.Linear(in_features=3*3, out_features=7*7)
        self.linear5 = nn.Linear(in_features=7*7, out_features=14*14)
        self.linear6 = nn.Linear(in_features=14*14, out_features=ip_dimn)
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        self.mu = self.relu3a(self.linear3a(x))
        self.log_sigma = self.relu3b(self.linear3b(x))

        # Decoder
        mv_std_Guassian = torch.from_numpy(np.random.multivariate_normal(mean = np.zeros(self.mu.shape[1]), cov = np.diag(np.ones(self.mu.shape[1])), size=self.mu.shape[0])).float().to(device)
        z_sampled = torch.mul(mv_std_Guassian,torch.exp(self.log_sigma)) + self.mu

        # mv_std_Guassian = torch.randn_like(mu)
        # z_sampled = mv_std_Guassian*torch.exp(log_sigma) + mu

        x = self.relu4(self.linear4(z_sampled))
        x = self.relu5(self.linear5(x))
        x = self.sigmoid(self.linear6(x))
        return x
    
    def sampleLatentAndGenerate(self):
        # Decoder
        mv_std_Guassian = torch.from_numpy(np.random.multivariate_normal(mean = np.zeros(self.mu.shape[1]), cov = np.diag(np.ones(self.mu.shape[1])), size=100)).float().to(device)
        z_sampled = mv_std_Guassian

        x = self.relu4(self.linear4(z_sampled))
        x = self.relu5(self.linear5(x))
        x = self.sigmoid(self.linear6(x))
        return x
    


# Build model.
model = VAE_Net().to(device)

# Build optimizer.
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

lambda_BCEloss = 28*28      

# Build loss.
criterionM = nn.BCELoss(reduction='mean')  

no_epochs = 150
first_pass = True
epoch_all = []
loss_test_all = []
loss_train_all = []

curr_lr = learning_rate
for epoch in range(no_epochs):

  # Training
  # batch_idx = 0
  total_loss_train = 0
  total_BCEloss_train = 0
  total_KLloss_train = 0
  X_train = []
  X_labels = []

  for batch_idx, (images, labels) in enumerate(train_loader):

    images = images.reshape(-1, 28*28)
    images = images.to(device)

    # Forward pass.
    x_reconst = model(images)

    # Compute loss.
    BCElossM = criterionM(x_reconst, images)*lambda_BCEloss

    KLloss = -0.5*torch.sum(1+torch.log(torch.square(torch.exp(model.log_sigma)) + 0.000000000000000000000000001) - torch.square(model.mu) - torch.square(torch.exp(model.log_sigma)))
    KLlossN = KLloss/images.shape[0]
    loss = BCElossM + KLlossN

    if epoch == 0 and first_pass == True:
      print(f'Initial {epoch} total loss {loss.item()}, BCEloss:{BCElossM}, KLloss:{KLlossN}')
      first_pass = False

    # Compute gradients.
    optimizer.zero_grad()
    loss.backward()

    # 1-step gradient descent.
    optimizer.step()

    # calculating train loss
    total_loss_train += loss.item()
    total_BCEloss_train += BCElossM.item()
    total_KLloss_train += KLlossN.item()

    if epoch == 0 and (batch_idx+1) % 10 == 0:
      print(f"Train Batch:{batch_idx}/{no_batches_train}, total loss: {loss}, BCEloss:{BCElossM}, KLloss:{KLlossN}, accumulated_epoch_total_loss: {total_loss_train}")

    # Accumulate data for PCA
    if batch_idx == 0:
       X_train = model.mu.detach().cpu().numpy()
       X_labels = labels.detach().cpu().numpy()
    else:
       X_train = np.concatenate((X_train, model.mu.detach().cpu().numpy()), axis=0)
       X_labels = np.concatenate((X_labels, labels.detach().cpu().numpy()), axis=0)

      


  # Decay learning rate
  if (epoch+1) % 50 == 0:
      curr_lr /= 10
      update_lr(optimizer, curr_lr)

  print(f'Train Epoch:{epoch}, Average Train loss:{total_loss_train/no_batches_train}, Average BCEloss:{total_BCEloss_train/no_batches_train}, Average KLloss:{total_KLloss_train/no_batches_train}' )

  if (epoch) % 5 == 0:
    x_concat = torch.cat([images.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
    save_image(x_concat, os.path.join(sample_dir, 'train_reconst-{}.png'.format(epoch)))


  # Testing after each epoch
  model.eval()
  with torch.no_grad():

    total_loss_test = 0
    total_BCEloss_test = 0
    total_KLloss_test = 0

    for images, labels in test_loader:

      images = images.reshape(-1, 28*28)
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass.
      x_reconst = model(images)

      # Compute test loss.
      BCElossM = criterionM(x_reconst, images)*lambda_BCEloss
      KLloss = -0.5*torch.sum(1+torch.log(torch.square(torch.exp(model.log_sigma)) + 0.000000000000000000000000001) - torch.square(model.mu) - torch.square(torch.exp(model.log_sigma)))
      KLlossN = KLloss/images.shape[0]
      loss = BCElossM + KLlossN
      
      total_loss_test += loss.item()
      total_BCEloss_test += BCElossM.item()
      total_KLloss_test += KLlossN.item()

    print(f'Test Epoch:{epoch}, Average Test loss: {total_loss_test/no_batches_tst}, Average BCEloss:{total_BCEloss_test/no_batches_tst}, Average KLloss:{total_KLloss_test/no_batches_tst}')

    if (epoch+1) % 5 == 0:
      x_concat = torch.cat([images.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
      save_image(x_concat, os.path.join(sample_dir, 'test_reconst-{}.png'.format(epoch+1)))

      # Generate samples y sampling latent space
      x_generate = model.sampleLatentAndGenerate().view(-1, 1, 28, 28)
      save_image(x_generate, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)), nrow = 10)


  # PLotting train and test curves
  # breakpoint()
  epoch_all.append(epoch)
  loss_test_all.append(total_loss_test/no_batches_tst)
  loss_train_all.append(total_loss_train/no_batches_train)

  plt.clf()
  plt.plot(epoch_all, loss_train_all, marker = 'o', mec = 'g', label='Average Train loss')
  plt.plot(epoch_all, loss_test_all, marker = 'o', mec = 'r', label='Average Test loss')
  plt.legend()
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()
  plt.savefig(os.path.join(sample_dir, 'Loss.png'))

  # Plotting latent space using PCA
  if (epoch) % 5 == 0:

    X_trainT = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    pca = PCA(n_components = 2)
    X_trainPCA = pca.fit_transform(X_trainT)
    # X_test = pca.transform(X_test)
    # breakpoint()

    plt.clf()
    no_points_plt = 10000
    X = X_trainPCA[0:no_points_plt,0]
    Y = X_trainPCA[0:no_points_plt,1]
    plt.scatter(X, Y, color = colors[X_labels[0:no_points_plt]])
    plt.title('PCA latent space')
    plt.show()
    plt.savefig(os.path.join(sample_dir, f'PCA_latentSpace-{epoch}.png'))

  model.train()
