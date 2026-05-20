from torch import nn

class CVNet(nn.Module):
    def __init__(self, hidden_neurons):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden_neurons),
            nn.Sigmoid(),
            nn.Linear(hidden_neurons, 10)
        )
    
    def forward(self, x):
        return self.net(x)
