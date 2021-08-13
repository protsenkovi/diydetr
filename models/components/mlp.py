from torch import nn

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
    super().__init__()
    assert num_layers > 1
    self.num_layers = num_layers

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for i in range(num_layers-2):
      layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))

    self.body = nn.Sequential(*layers)

  def forward(self, x):
    return self.body(x)