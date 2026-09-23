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

def create_dataset(prices):
    X = sliding_window_view(prices, 100)[:-10]
    Y = sliding_window_view(prices[100:], 10)
    return X, Y

def main():
    tickers = ['NDA-FI.HE', 'NOKIA.HE', 'KNEBV.HE', 'SAMPO.HE', 'NESTE.HE',
           'FORTUM.HE', 'WRT1V.HE', 'METSO.HE', 'UPM.HE', 'ORNBV.HE',
           'KESKOB.HE', 'STERV.HE', 'KCR.HE', 'ELISA.HE', 'VALMT.HE',
           'HIAB.HE', 'HUH1V.HE', 'MANTA.HE', 'OUT1V.HE', 'KEMIRA.HE',
           'TYRES.HE', 'LUMO.HE', 'TIETO.HE', 'BITTI.HE', 'QTCOM.HE'
           ]

    X_all_tr = []
    Y_all_tr = []

    # Form training data
    for ticker in tickers:
        # Load data
        data = yf.download(ticker, '2020-01-01', '2025-12-31',
                           group_by=ticker, auto_adjust=False,
                           threads=False, progress=False)[ticker]
    
        prices = data['Adj Close'].to_numpy()
        # Create dataset of prices
        X, Y = create_dataset(prices)

        X_all_tr.append(X)
        Y_all_tr.append(Y)

    X_all_test = []
    Y_all_test = []

    for ticker in tickers:
        data = yf.download(ticker, '2026-01-01', '2026-09-22',
                           group_by=ticker, auto_adjust=False,
                           threads=False, progress=False)[ticker]
        prices = data['Adj Close'].to_numpy()

        X, Y = create_dataset(prices)

        X_all_test.append(X)
        Y_all_test.append(Y)

    X = torch.from_numpy(X_all_tr[0].copy()).float()
    Y = torch.from_numpy(Y_all_tr[0].copy()).float()

    X_t = torch.from_numpy(X_all_test[0].copy()).float()
    Y_t = torch.from_numpy(Y_all_test[0].copy()).float()

    model = LinearModel()

    criterion =  nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(1000):
        optimizer.zero_grad()

        Y_pred = model(X)

        loss = criterion(Y_pred, Y)

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss.item():.6f}')

    pred = model(X_t)
    print(criterion(pred, Y_t).item())
    


if __name__ == "__main__":
    main()