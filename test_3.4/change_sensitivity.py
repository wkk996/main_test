import torch
import casadi as ca
import numpy as np


class SensitivityNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(SensitivityNN, self).__init__()
        self.fc1 = torch.nn.Linear(3, 32)  
        self.fc2 = torch.nn.Linear(32, 32) 
        self.fc3 = torch.nn.Linear(32, 32) 
        self.fc4 = torch.nn.Linear(32, 1)  

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.tanh(self.fc3(x))  
        x = self.fc4(x)              
        return x


def load_model(model_path='neural_sensitivity.pth'):
    model = SensitivityNN(3)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model


def get_model_parameters(model):
    state_dict = model.state_dict()
    
    W1 = state_dict['fc1.weight'].detach().numpy()
    b1 = state_dict['fc1.bias'].detach().numpy()
    W2 = state_dict['fc2.weight'].detach().numpy()
    b2 = state_dict['fc2.bias'].detach().numpy()
    W3 = state_dict['fc3.weight'].detach().numpy()
    b3 = state_dict['fc3.bias'].detach().numpy()
    W4 = state_dict['fc4.weight'].detach().numpy()
    b4 = state_dict['fc4.bias'].detach().numpy()
    
    return W1, b1, W2, b2, W3, b3, W4, b4


def create_casadi_nn(input_dim, W1, b1, W2, b2, W3, b3, W4, b4):

    x = ca.MX.sym('x', input_dim)  

    W1_casadi = ca.MX(W1)  
    b1_casadi = ca.MX(b1)  
    layer1 = ca.tanh(ca.mtimes(W1_casadi, x) + b1_casadi) 


    W2_casadi = ca.MX(W2)
    b2_casadi = ca.MX(b2)
    layer2 = ca.tanh(ca.mtimes(W2_casadi, layer1) + b2_casadi)  


    W3_casadi = ca.MX(W3)
    b3_casadi = ca.MX(b3)
    layer3 = ca.tanh(ca.mtimes(W3_casadi, layer2) + b3_casadi) 


    W4_casadi = ca.MX(W4)
    b4_casadi = ca.MX(b4)
    output = ca.mtimes(W4_casadi, layer3) + b4_casadi  

    return x, output


def pytorch_to_casadi_sensitivity(model_path='neural_sensitivity.pth'):

    model = load_model(model_path)
    

    W1, b1, W2, b2, W3, b3, W4, b4 = get_model_parameters(model)
    
    x_casadi, output_casadi = create_casadi_nn(3, W1, b1, W2, b2, W3, b3, W4, b4)
    

    nn_sensitivity = ca.Function('nn_function', [x_casadi], [output_casadi])
    
    return nn_sensitivity


