import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
#from logger import Logger

import torch.nn.functional as F


# MNIST Dataset 
dataset = dsets.MNIST(root='./data', 
                      train=True, 
                      transform=transforms.ToTensor(),  
                      download=True)

# Data Loader (Input Pipeline)
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=100, 
                                          shuffle=True)

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

#Encoder
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss
        
# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = F.log_softmax(self.fc2(out))
        return out

z_red_dims = 120
Q = Q_net(784,1000,z_red_dims).cuda()
Q.load_state_dict(torch.load('Q_encoder_weights.pt'))
Q.eval() #turn off dropout
net = Net(input_size = z_red_dims).cuda()

# # Set the logger
# logger = Logger('./logs/encoder_fit_120_sm_eval')

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters())#, lr=0.00001)  

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

# Start training
for step in range(total_step):
    
    # Reset the data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch the images and labels and convert them to variables
    images, labels = next(data_iter)
    images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)
    
    # Forward, backward and optimize
    optimizer.zero_grad()  # zero the gradient buffer
    outputs = net(Q(images))
    # outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step+1) % 100 == 0:
        print ('Step [%d/%d], Loss: %.4f, Acc: %.2f' 
               %(step+1, total_step, loss.data[0], accuracy.data[0]))

        #============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'loss': loss.data[0],
            'accuracy': accuracy.data[0]
        }

        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, step+1)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in net.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, to_np(value), step+1)
        #     logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)

        # # (3) Log the images
        # info = {
        #     'images': to_np(images.view(-1, 28, 28)[:10])
        # }

        # for tag, images in info.items():
        #     logger.image_summary(tag, images, step+1)

#test
# MNIST Dataset 
dataset_test = dsets.MNIST(root='./data', 
                      train=False, 
                      transform=transforms.ToTensor(),  
                      download=True)

# Data Loader (Input Pipeline)
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, 
                                          batch_size=10000, 
                                               shuffle=True)
data_iter_test = iter(data_loader_test)    
# Fetch the images and labels and convert them to variables
images, labels = next(data_iter_test)
images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

outputs = net(Q(images))
# outputs = net(images)

# Compute accuracy
_, argmax = torch.max(outputs, 1)
accuracy = (labels == argmax.squeeze()).float().mean()

print(accuracy.data[0])

