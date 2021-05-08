
import pandas as pd
import typing
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def load_array(data_arrays: typing.Tuple[torch.Tensor, torch.Tensor],
               batch_size: int, is_train: bool = True) -> DataLoader:
    """

    :param data_arrays: (features, labels) torch. tensors
        features: num_features x num_examples
        labels: num_features x num_ouputs
    :param batch_size:
    :param is_train:
    :return:
    """
    dataset = TensorDataset(*data_arrays)

    return DataLoader(dataset, batch_size, shuffle=is_train)


def train(data_iter_train, data_iter_eval, model, optimizer, loss, num_epochs, print_every=1, return_params = False):
    """

    :param data_iter_train:
    :param data_iter_eval:
    :param model:
    :param optimizer:
    :param loss:
    :param num_epochs:
    :param print_every:
    :return:
    """
    params_ = []
    train_loss, valid_loss = [], []

    for epoch in range(num_epochs):
        run_loss = 0
        for i, (X, y) in enumerate(data_iter_train):
            optimizer.zero_grad()
            y_hat = model.forward(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            run_loss += l.item()

        model.eval()
        with torch.no_grad():
            for X_valid, y_valid in data_iter_eval:
                y_hat = model.forward(X_valid)
                l_valid = loss(y_hat, y_valid).item()

        if return_params:
            if hasattr(model, 'get_params'):
                params_.append(model.get_params)

        train_loss.append(run_loss)
        valid_loss.append(l_valid)

        if epoch % print_every == 0:
            print(f'epoch: {epoch + 1}, train-loss: {run_loss:f}, valid-loss: {l_valid:f}')

    loss_df = pd.DataFrame({'loss_train': train_loss, 'loss_valid': valid_loss})

    if return_params:
        return model,  loss_df, params_
    else:
        return model, loss_df


class MLPRegressor(torch.nn.Module):
    def __init__(self, num_features, num_outputs, hidden_layers, seed=123):
        super(MLPRegressor, self).__init__()
        self.seed = seed
        self.n_layers = len(hidden_layers)
        # hidden layer parameters: (hi, ho)

        if self.n_layers == 1:
            layer_sizes = [hidden_layers[0]]
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        # Instanciate hidden_layers with input_layer
        input_layer = nn.Linear(num_features, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([input_layer])
        if self.n_layers > 1:
            self.hidden_layers.extend([nn.Linear(hi, ho) for hi, ho in layer_sizes])

        # output layer: regression: a q_est for every action accessible from state s
        self.out = nn.Linear(hidden_layers[-1], num_outputs)

        torch.manual_seed(seed)
        self.reset_parameters()

    def forward(self, X):
        x = X
        for fc in self.hidden_layers:
            x = F.relu(fc(x))

        return self.out(x)

    def reset_parameters(self, mu=0., sigma=0.01):
        for layer in self.hidden_layers:
            layer.weight.data.normal_(mu, sigma)
            layer.bias.data.fill_(0)