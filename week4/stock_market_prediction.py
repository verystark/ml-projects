import yfinance as yf
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear1 = nn.Linear(100, 20)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.output(y)
        y = F.relu(y)
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
    
    for i, ticker in enumerate(tickers):
        # Load data
        data = yf.download(ticker, '2020-01-01', '2025-12-31',
                           group_by=ticker, auto_adjust=False,
                           threads=False, progress=False)[ticker]
    
        prices = data['Adj Close'].to_numpy()
        # Create dataset of prices
        X, Y = create_dataset(prices)

        X_all_tr.append(X)
        Y_all_tr.append(Y)
        


if __name__ == "__main__":
    main()