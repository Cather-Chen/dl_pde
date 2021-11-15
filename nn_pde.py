from torch import nn
import torch
from torch.autograd import grad
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

class softplus_power(nn.Module):
    def __init(self):
        super.__init__()

    def forward(self, x):
        m = nn.Softplus()
        x = m(x)
        return x**1.1
class ICNN(nn.Module):
    '''
    Parameters :        n_inputs --- number of inputs
                        n_outputs -- number of outputs
                        n_layers --- number of hidden layers
                        n_npl ------ number of nodes per hidden layer
    '''
    def __init__(self, n_inputs, n_outputs, n_layers, n_npl):
        super().__init__()
        self.n_inps = n_inputs
        if n_layers == 0:
            self.layers = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.layers.append(nn.Linear(n_inputs, n_outputs))
        else:
            self.layers = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.layers.append(nn.Linear(n_inputs, n_npl))
            self.activations.append(softplus_power())
            for x in range(n_layers-1):
                self.layers.append(nn.Linear(n_npl + n_inputs, n_npl))
                self.activations.append(softplus_power())
            self.layers.append(nn.Linear(n_npl + n_inputs, n_outputs))
    '''
    Makes the all the weights positive.
    Refer to : https://arxiv.org/pdf/1609.07152.pdf
    '''
    def make_convex(self):
        with torch.no_grad():
            for i in range(1, len(self.layers)):
                if i == 0:
                    self.layers[i].weight[:, :][self.layers[i].weight[:, :] < 0] = \
                        torch.abs(self.layers[i].weight[:, :][self.layers[i].weight[:, :] < 0])
                self.layers[i].weight[:, :-self.n_inps][self.layers[i].weight[:, :-self.n_inps] < 0] = \
                    torch.abs(self.layers[i].weight[:, :-self.n_inps][self.layers[i].weight[:, :-self.n_inps] < 0])

    '''
    Sets all negative weights (W^z) to 0
    '''
    def project(self):
        with torch.no_grad():
            for i in range(1, len(self.layers)):
                if i == 0:
                    self.layers[i].weight[:, :][self.layers[i].weight[:, :] < 0] = 0
                self.layers[i].weight[:, :-self.n_inps][self.layers[i].weight[:, :-self.n_inps] < 0] = 0

    '''
    Saves the model in the specified path and name with .pth extension
    '''
    def save(self, path, model_name):
        torch.save(self.state_dict(), path + model_name + '.pth')

    '''
    Loads the model given its path.
    Different Architecture or activation function (with parameters) in the saved model will throw an error
    '''
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def forward(self, x):
        inps = x
        out = x
        for i in range(len(self.layers)-1):
            if i > 0:
                out = torch.cat((out, inps), dim=1)
            out = self.layers[i](out)
            out = self.activations[i](out)

        if len(self.layers) != 1:
            out = torch.cat((out, inps), dim=1)

        out = self.layers[-1](out)
        return out

class PINN(nn.Module):
    '''
    Parameters :        n_inputs --- number of inputs
                        n_outputs -- number of outputs
                        n_layers --- number of hidden layers
                        n_npl ------ number of nodes per hidden layer
    '''
    def __init__(self, n_inputs, n_outputs, n_layers, n_npl):
        super().__init__()
        self.n_inps = n_inputs
        if n_layers == 0:
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(n_inputs, n_outputs))
        else:
            self.layers = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.layers.append(nn.Linear(n_inputs, n_npl))
            for x in range(n_layers-1):
                self.layers.append(nn.Linear(n_npl, n_npl))
            self.layers.append(nn.Linear(n_npl, n_outputs))
    def save(self, path, model_name):
        torch.save(self.state_dict(), path + model_name + '.pth')
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def forward(self, x):
        out = x
        for i in range(len(self.layers)-1):
            out = self.layers[i](out)
            out = F.relu(out)
        out = self.layers[-1](out)
        return out

def tensor_data(func_f,func_u,n,pattern):
    '''
    :param func_f: f in paper
    :param func_u: u in paper
    :param n: grid num
    :param pattern: 0--uniform, 1-random
    :return: tensor dataset
    '''
    if pattern == 0:
        datax = []
        F = []
        U = []
        X_ = np.linspace(-1,1,n)
        Y_ = np.linspace(-1,1,n)
        for i in X_:
            for j in Y_:
                x = [i,j]
                f = func_f(i,j)
                u = func_u(i,j)
                datax.append(x)
                F.append(f)
                U.append(u)
        all_x = torch.tensor(datax,dtype=torch.float)
        all_f = torch.tensor(F,dtype=torch.float)
        all_u = torch.tensor(U,dtype=torch.float)
        dataset = TensorDataset(all_x,all_f,all_u)
        return dataset

    if pattern == 1:
        datax = []
        F = []
        U = []
        X_ = np.random.uniform(-1,1,(n**2,2))
        for t in range(n**2):
            i = X_[t][0]
            j = X_[t][1]
            x = [i, j]
            f = func_f(i, j)
            u = func_u(i, j)
            datax.append(x)
            F.append(f)
            U.append(u)
        all_x = torch.tensor(datax,dtype=torch.float)
        all_f = torch.tensor(F,dtype=torch.float)
        all_u = torch.tensor(U,dtype=torch.float)
        dataset = TensorDataset(all_x,all_f,all_u)
        return dataset

def boundary(func_u,n,device):
    X = []
    U = []
    X1 = np.random.uniform(-1, 1, size=n)
    for i in X1:
        data1 = [-1,i]
        u = func_u(-1, i)
        X.append(data1)
        U.append(u)
    X2 = np.random.uniform(-1, 1, size=n)
    for i in X2:
        data2 = [1, i]
        u = func_u(1, i)
        X.append(data2)
        U.append(u)
    X3 = np.random.uniform(-1, 1, size=n)
    for i in X3:
        data3 = [i, -1]
        u = func_u(i, -1)
        X.append(data3)
        U.append(u)
    X4 = np.random.uniform(-1, 1, size=n)
    for i in X4:
        data4 = [i, 1]
        u = func_u(i, 1)
        X.append(data4)
        U.append(u)
    bound_x = torch.tensor(X, dtype=torch.float).to(device)
    bound_u = torch.tensor(U, dtype=torch.float).to(device)
    return bound_x, bound_u

def init_weights(t):
    with torch.no_grad():
        if type(t) == torch.nn.Linear:
            t.weight.normal_(0, 0.02)
            t.bias.normal_(0, 0.02)


def train_network(model,n_epoch,device,init_weights_flag =False):
    if init_weights_flag == True:
        model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    criterion = nn.MSELoss()
    # model.make_convex() #if model is ICNN, please use make_convex()
    loss_in_list = []
    loss_bound_list = []
    for epoch in range(1,n_epoch+1):
        bound_x, bound_u = boundary(func_u, n=256, device=device)  # 256*4=1024
        train_data = tensor_data(func_f, func_u, n=32, pattern=1)   #32*32 = 1024
        train_loader = DataLoader(train_data, shuffle=True, batch_size=1024, num_workers=8)
        for i,batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            f = batch[1]
            u = batch[2]
            x1 = x[:,0].view(x.shape[0],1)
            x2 = x[:,1].view(x.shape[0],1)
            x1.requires_grad = True
            x2.requires_grad = True
            x_ = torch.cat((x1,x2),dim=1)
            output = model(x_)
            output_bound = model(bound_x)
            loss_bound = criterion(output_bound,bound_u)
            v = torch.ones(output.shape).to(x.device)
            ux = torch.autograd.grad(output,x1,grad_outputs=v,retain_graph = True,create_graph=True,allow_unused=True)[0]
            v2 = torch.ones(ux.shape).to(x.device)
            uxy = torch.autograd.grad(ux,x2,grad_outputs=v2,retain_graph = True,create_graph=True,allow_unused=True)[0]
            uxx = torch.autograd.grad(ux,x1,grad_outputs=v2,retain_graph = True,create_graph=True,allow_unused=True)[0]
            uy = torch.autograd.grad(output,x2,grad_outputs=v,retain_graph = True,create_graph=True,allow_unused=True)[0]
            uyx = torch.autograd.grad(uy,x1,grad_outputs=v2,retain_graph = True,create_graph=True,allow_unused=True)[0]
            uyy = torch.autograd.grad(uy,x2,grad_outputs=v2,retain_graph = True,create_graph=True,allow_unused=True)[0]
            fh = (uxx*uyy) - (uxy*uyx)
            f = f.view(fh.size(0),1)

            loss_in = criterion(fh,f)
            loss = 1000*loss_bound + loss_in
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # model.project()   #if model is ICNN, please use project()
            if loss_bound.item()<1e-5:
                loss_in_list.append(loss_in.item())
                loss_bound_list.append(loss_bound.item())
                test_model( model, device)
                print('error less than threshold')
                return model, loss_in_list, loss_bound_list

        loss_in_list.append(loss_in.item())
        loss_bound_list.append(loss_bound.item())
        print("\n")
        print('Loss in after Epoch {} is : {}'.format(epoch, loss_in_list[-1]))
        print('Loss bound after Epoch {} is : {}'.format(epoch, loss_bound_list[-1]))
        if epoch %100 ==0:
            test_model(model, device)

    return model, loss_in_list, loss_bound_list


def test_model(model,device):
    h = 2**-7
    x_ = np.arange(-1, 1+h, h)
    y_ = np.arange(-1, 1 + h, h)
    x, y = np.meshgrid(x_, y_)
    l = len(x)
    u = []
    X = []
    for j in range(l):
        for i in range(l):
            X.append([x_[i],x_[j]])
            u.append(func_u(x_[i],x_[j]))
    x_input = torch.tensor(X,dtype=torch.float).view((-1,2)).to(device)
    u = torch.tensor(u, dtype=torch.float).view((-1, 1)).to(device)
    u_pre = model(x_input).view((-1,1))
    error = u - u_pre
    absolute_error = torch.abs(error)
    u_pre = u_pre.view((l, l)).detach().cpu().numpy()
    absolute_error = absolute_error.view((l, l)).detach().cpu().numpy()
    u = u.view((l,l)).detach().cpu().numpy()
    fig = plt.figure(figsize=(9, 3))
    cm = plt.cm.get_cmap('viridis')
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()
    ax1.plot_surface(x, y, u, cmap=cm)
    ax1.set_xlabel('Y Label')
    ax1.set_ylabel('X Label')
    ax3.set_zlabel('True value')
    ax2.plot_surface(x, y, u_pre, cmap=cm)
    ax2.set_xlabel('Y Label')
    ax2.set_ylabel('X Label')
    ax3.set_zlabel('predicted value')
    ax3.plot_surface(x, y,absolute_error, cmap=cm)
    ax3.set_xlabel('Y Label')
    ax3.set_ylabel('X Label')
    ax3.set_zlabel('error')
    plt.show()
def func_f(x,y):
    return (x**2+y**2+1)*np.exp(x**2+y**2)
def func_u(x,y):
    return np.exp((x**2+y**2)/2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ICNN(n_inputs=2,n_outputs=1,n_layers=5,n_npl=50)
    model = PINN(n_inputs=2,n_outputs=1,n_layers=5,n_npl=50)
    model.to(device)
    model, loss_in,loss_bound = train_network(model, n_epoch=2000,device=device)
    print('loss_in',loss_in)
    print('loss_bound',loss_bound)


if __name__ == "__main__":
    main()




# def loss_fct(u,x,f):
#     u = u.requires_grad_()
#     x = x.requires_grad_()
#     dy, =grad(u.sum(),x, create_graph=True, retain_graph=True,allow_unused=True)
#     x_grad_u = dy
#
#     der = torch.zeros((x.shape[0],x.shape[1],x.shape[1])).to(x.device)
#     for dim in range(x.shape[1]):
#         der[:,dim,:]=grad(x_grad_u[:,dim].sum(),x,create_graph=True,retain_graph=True,allow_unused=True)[0]
#     det_du = torch.det(der)
#     det_du = det_du.view(det_du.shape[0],1)
#     f = f.view(f.shape[0],1)
#     loss_func = torch.nn.MSELoss(reduce=True, size_average=True)
#     loss = loss_func(det_du, f)
#     return loss

