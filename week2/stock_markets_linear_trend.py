import yfinance as yf

import matplotlib.pyplot as plt

def my_linfit(x, y):
    a = 0
    b = 0
    return a, b

def main():
    tickers = ['NDA-FI.HE'#, 'NOKIA.HE', 'KNEBV.HE', 'SAMPO.HE', 'NESTE.HE',
           #'FORTUM.HE', 'WRT1V.HE', 'METSO.HE', 'UPM.HE', 'ORNBV.HE',
           #'KESKOB.HE', 'STERV.HE', 'KCR.HE', 'ELISA.HE', 'VALMT.HE',
           #'HIAB.HE', 'HUH1V.HE', 'MANTA.HE', 'OUT1V.HE', 'KEMIRA.HE',
           #'TYRES.HE', 'LUMO.HE', 'TIETO.HE', 'BITTI.HE', 'QTCOM.HE'
           ]

    for company in tickers:
        data = yf.download(company, '2022-02-24', '2026-08-31', auto_adjust=False, threads=False, progress=False)
        adj_close = data['Adj Close']
        y = adj_close[company]
        x = []
        for i in range(y.size):
            x.append(i)

        a, b = my_linfit(x, y)

if __name__ == '__main__':
    main()