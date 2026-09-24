import yfinance as yf
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(100, 10)
        self.output = nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.output(y)
        return y

# create training pairs using sliding window algorithm
def create_dataset(prices):
    X = sliding_window_view(prices, 100)[:-10]
    Y = sliding_window_view(prices[100:], 10)
    return X, Y

# normalization for company data to get them to similar scale
def normalization(data):
    prices = data['Adj Close'].to_numpy()
    
    mean = prices.mean()
    std = prices.std()
    return (prices - mean) / std


def main():
    tickers = ['NDA-FI.HE', 'NOKIA.HE', 'KNEBV.HE', 'SAMPO.HE', 'NESTE.HE',
           'FORTUM.HE', 'WRT1V.HE', 'METSO.HE', 'UPM.HE', 'ORNBV.HE',
           'KESKOB.HE', 'STERV.HE', 'KCR.HE', 'ELISA.HE', 'VALMT.HE',
           'HIAB.HE', 'HUH1V.HE', 'MANTA.HE', 'OUT1V.HE', 'KEMIRA.HE',
           'TYRES.HE', 'LUMO.HE', 'TIETO.HE', 'BITTI.HE', 'QTCOM.HE'
           ]

    X_all_tr = []
    Y_all_tr = []

    # form training data
    for ticker in tickers:
        # load data
        data = yf.download(ticker, '2020-01-01', '2025-12-31',
                           group_by=ticker, auto_adjust=False,
                           threads=False, progress=False)[ticker]

        prices = normalization(data)
    
        # create dataset of prices
        X, Y = create_dataset(prices)

        X_all_tr.append(X)
        Y_all_tr.append(Y)

    X_all_test = []
    Y_all_test = []

    # form testing data
    for ticker in tickers:
        # load data
        data = yf.download(ticker, '2026-01-01', '2026-09-22',
                           group_by=ticker, auto_adjust=False,
                           threads=False, progress=False)[ticker]
        
        prices = normalization(data)

        # create dataset of prices
        X, Y = create_dataset(prices)

        X_all_test.append(X)
        Y_all_test.append(Y)

    # join all OMXH25 company closing price data to one 2D numpy array
    X_all_tr = np.concatenate(X_all_tr, axis=0)
    Y_all_tr = np.concatenate(Y_all_tr, axis=0)
    X_all_test = np.concatenate(X_all_test, axis=0)
    Y_all_test = np.concatenate(Y_all_test, axis=0)

    # turn training data into tensors
    X = torch.from_numpy(X_all_tr.copy()).float()
    Y = torch.from_numpy(Y_all_tr.copy()).float()

    # turn test data into tensors
    X_t = torch.from_numpy(X_all_test.copy()).float()
    Y_t = torch.from_numpy(Y_all_test.copy()).float()

    torch.manual_seed(42)

    model = LinearModel()

    criterion =  nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)

    for epoch in range(1000):
        # set gradients to zero
        optimizer.zero_grad()

        # make prediction with model
        Y_pred = model(X)

        # compute loss of prediction
        loss = criterion(Y_pred, Y)

        loss.backward()

        # compute updates based on optimizer rules
        optimizer.step()

        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss.item():.6f}')

    # test prediction accuracy
    pred = model(X_t)
    print(f'Test loss: {criterion(pred, Y_t).item()}')

    # form Kemira company data
    data = yf.download('KEMIRA.HE', '2026-01-01', '2026-09-24',
                       auto_adjust=False, threads=False, progress=False)

    normalized_prices = normalization(data)
    X = torch.from_numpy(normalized_prices[-100:].copy()).float().reshape(1, 100)

    kemira_normalized_pred = model(X)

    prices = data['Adj Close'].to_numpy()
    mean = prices.mean()
    std = prices.std()

    non_normalized_pred = kemira_normalized_pred * std + mean
    print(f'KEMIRA price prediction for Monday, September 28: {non_normalized_pred[0, 1].item():.6f}€')

    
    


if __name__ == "__main__":
    main()