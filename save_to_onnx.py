import torch.onnx
import torch.nn as nn
import torch


class MuhNet(nn.Module):
    def __init__(self):
        super(MuhNet, self).__init__()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.relu1 = nn.ReLU()

        #self.conv1 = nn.Conv2d(1, 1, 4, 1)
        self.linear1 = nn.Linear(28*28, 10*10)
        self.linear2 = nn.Linear(10*10, 10)
    
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.sigmoid1(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid1(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x

net = MuhNet()

net.load_state_dict(torch.load("pretrained/21_1000"))

net.eval()

# Export the model
torch.onnx.export(net,               # model being run
                  torch.randn(32, 28, 28),                         # model input (or a tuple for multiple inputs)
                  "mnist.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})