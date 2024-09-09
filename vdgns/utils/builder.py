import torch
import torch.nn as nn
from torch.utils.data import Dataset


ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'identity': nn.Identity
}

LOSS_FUNCTIONS = {
    'L2': nn.MSELoss,
    'L1': nn.L1Loss,
    'CrossEntropy': nn.CrossEntropyLoss
}


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super(SimpleDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_mlp(input_dim: int,
              hidden_layers: list,
              out_dim: int,
              activations='relu'):
    if not isinstance(activations, list):
        activations = [activations] * (len(hidden_layers))
        activations.append('identity')
    elif len(activations) != len(hidden_layers) + 1:
        print(f'To build an MLP, you need to pass the same amount of activations as layers. You want to create {len(hidden_layers) + 1} layers, while providing {len(activations)} activation functions.')
        return None
    
    layer_sizes = [input_dim, *hidden_layers, out_dim]

    layers = []

    # Iterate over the layer sizes and create linear layers
    for i in range(1, len(layer_sizes)):
        in_size = layer_sizes[i - 1]
        out_size = layer_sizes[i]
        layer = nn.Linear(in_size, out_size)

        # Append the linear layer to the list
        layers.append(layer)

        # Add activation after all but the last layer
        if not activations[i - 1] in ACTIVATION_FUNCTIONS.keys():
            print(f"{activations[i - 1]} is not a valid activation function. Should be one of {ACTIVATION_FUNCTIONS.keys()}. Using ReLU as default.")
            activations[i - 1] = 'relu'
        activation_fun = ACTIVATION_FUNCTIONS[activations[i - 1]]
        layers.append(activation_fun())

    # Combine all layers into a Sequential model
    mlp = nn.Sequential(*layers)

    return mlp



def train(model: nn.Module,
          train_loader,
          n_epochs=10,
          lr=0.001,
          lr_decay=1.0,
          loss='L2',
          validation_loader=None,
          early_stop_length=None,
          verbose=False,
          additional_models=[]):

    stats = {
        'loss_iteration': [],
        'loss_epoch': [],
        'loss_validation': []
    }

    criterion = LOSS_FUNCTIONS[loss]()
    
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr)]
    for m in additional_models:
        optimizers.append(torch.optim.Adam(m.parameters(), lr=lr))
    
    schedulers = []
    for o in optimizers:
        schedulers.append(torch.optim.lr_scheduler.ExponentialLR(o, gamma=lr_decay))

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        for i, (img, true) in enumerate(train_loader):
            for o in optimizers:
                o.zero_grad()

            y_pred = model(img)

            loss = criterion(y_pred, true)
            loss.backward()

            for o in optimizers:
                o.step()

            train_loss += loss.item()
            stats['loss_iteration'].append(loss.item())
        
        for s in schedulers:
            s.step()

        train_loss = train_loss / len(train_loader)
        stats['loss_epoch'].append(train_loss)
        if verbose:
            print(f"Epoch: {epoch} \t Loss: {train_loss}")

        if validation_loader is not None:
            validation_loss = 0.0
            for i, (img, true) in enumerate(validation_loader):
                y_pred = model(img)
                loss = criterion(y_pred, true)
                validation_loss += loss.item()
            validation_loss /= len(validation_loader)
            stats['loss_validation'].append(validation_loss)

            if verbose:
                print(f"\tValidation loss: {validation_loss}")

            if early_stop_length is not None and len(stats['loss_validation']) > early_stop_length:
                is_improved = False
                past_result = stats['loss_validation'][-early_stop_length]
                for i in range(early_stop_length):
                    if stats['loss_validation'][-i] < past_result:
                        is_improved = True
                        break
                if not is_improved:
                    print("Early stopped")
                    return stats

    return stats
