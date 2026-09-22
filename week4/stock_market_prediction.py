import yfinance as yf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def create_dataset(tickers):

def main():
    tickers = ['NDA-FI.HE', 'NOKIA.HE', 'KNEBV.HE', 'SAMPO.HE', 'NESTE.HE',
           'FORTUM.HE', 'WRT1V.HE', 'METSO.HE', 'UPM.HE', 'ORNBV.HE',
           'KESKOB.HE', 'STERV.HE', 'KCR.HE', 'ELISA.HE', 'VALMT.HE',
           'HIAB.HE', 'HUH1V.HE', 'MANTA.HE', 'OUT1V.HE', 'KEMIRA.HE',
           'TYRES.HE', 'LUMO.HE', 'TIETO.HE', 'BITTI.HE', 'QTCOM.HE'
           ]

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
    

if __name__ == "__main__":
    main()