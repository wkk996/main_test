import torch
import casadi as ca
import numpy as np


class ValueFunctionNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(ValueFunctionNN, self).__init__()
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


def load_model(model_path='value_function_model5--30--20w.pth'):
    model = ValueFunctionNN(3)
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

    x = ca.MX.sym('x', 1, input_dim)  
    

    W1_T = W1.T
    W2_T = W2.T
    W3_T = W3.T
    W4_T = W4.T
    

    W1_casadi = ca.MX(W1_T)
    b1_casadi = ca.MX(b1).T
    layer1 = ca.tanh(ca.mtimes(x, W1_casadi) + b1_casadi)


    W2_casadi = ca.MX(W2_T)
    b2_casadi = ca.MX(b2).T
    layer2 = ca.tanh(ca.mtimes(layer1, W2_casadi) + b2_casadi)

    
    W3_casadi = ca.MX(W3_T)
    b3_casadi = ca.MX(b3).T
    layer3 = ca.tanh(ca.mtimes(layer2, W3_casadi) + b3_casadi)

   
    W4_casadi = ca.MX(W4_T)
    b4_casadi = ca.MX(b4).T
    output = ca.mtimes(layer3, W4_casadi) + b4_casadi

    return x, output


def pytorch_to_casadi(model_path='value_function_model5--30--20w.pth'):

    model = load_model(model_path)
    

    W1, b1, W2, b2, W3, b3, W4, b4 = get_model_parameters(model)
    

    x_casadi, output_casadi = create_casadi_nn(3, W1, b1, W2, b2, W3, b3, W4, b4)

    nn_function = ca.Function('nn_function', [x_casadi], [output_casadi])
    
    return nn_function


def test_casadi_nn():
    model_path = 'value_function_model5--30--20w.pth' 
    nn_function = pytorch_to_casadi(model_path)
    

    state_input = np.array([10, 10, 10]) 

    output_value = nn_function(state_input)
    
    print("Predicted output:", output_value)

if __name__ == "__main__":
    test_casadi_nn()