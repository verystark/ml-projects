import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def plus_or_minus(performance):
    return '+' if performance > 0 else ''

def my_linfit(x, y):
    a = (-sum(x)*sum(y)+len(x)*sum(x*y))/(len(x)*sum(x**2)-(sum(x)**2))
    b = (-a*sum(x)+sum(y))/len(x)

    return a, b

def main():
    tickers = ['NDA-FI.HE', 'NOKIA.HE', 'KNEBV.HE', 'SAMPO.HE', 'NESTE.HE',
           'FORTUM.HE', 'WRT1V.HE', 'METSO.HE', 'UPM.HE', 'ORNBV.HE',
           'KESKOB.HE', 'STERV.HE', 'KCR.HE', 'ELISA.HE', 'VALMT.HE',
           'HIAB.HE', 'HUH1V.HE', 'MANTA.HE', 'OUT1V.HE', 'KEMIRA.HE',
           'TYRES.HE', 'LUMO.HE', 'TIETO.HE', 'BITTI.HE', 'QTCOM.HE'
           ]

    for company in tickers:
        data = yf.download(company, '2022-02-24', '2026-08-31', auto_adjust=False, threads=False, progress=False)
        adj_close = data['Adj Close']
        y = adj_close[company]
        x = []
        for i in range(y.size):
            x.append(i)
        x = np.array(x)

        a, b = my_linfit(x, y)

        performance = (y.iloc[-1] / y.iloc[0] - 1) * 100
        last_linreg = (a*x+b)[-1]
        performance_vs_trend = (y.iloc[-1] / last_linreg - 1) * 100

        adj_close.plot()
        plt.plot(data.index, a*x+b)
        plt.title(f'Stock performace: {plus_or_minus(performance)}{performance:.2f}%, Stock vs. linear trend: {plus_or_minus(performance_vs_trend)}{performance_vs_trend:.2f}%')
        plt.show()

if __name__ == '__main__':
    main()