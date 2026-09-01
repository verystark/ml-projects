import yfinance as yf

import matplotlib.pyplot as plt

tickers = ['NDA-FI.HE'#, 'NOKIA.HE', 'KNEBV.HE', 'SAMPO.HE', 'NESTE.HE',
           #'FORTUM.HE', 'WRT1V.HE', 'METSO.HE', 'UPM.HE', 'ORNBV.HE',
           #'KESKOB.HE', 'STERV.HE', 'KCR.HE', 'ELISA.HE', 'VALMT.HE',
           #'HIAB.HE', 'HUH1V.HE', 'MANTA.HE', 'OUT1V.HE', 'KEMIRA.HE',
           #'TYRES.HE', 'LUMO.HE', 'TIETO.HE', 'BITTI.HE', 'QTCOM.HE'
           ]

best_company = ''
best_return = 0
for i in tickers:
    data = yf.download(i, '2020-01-01', '2026-01-01', auto_adjust=False, progress=False, threads=False)
    print('downloaded', i)
    adj_close = data['Adj Close']
    change = adj_close.iloc[-1] / adj_close.iloc[0]
    percent_change = (change.iloc[0] - 1) * 100
    print(f'{i}: {'+' if percent_change > 0 else ''}{percent_change:.2f}%')
    print(f'If you invested 1000 euros in the start date you would have {(change.iloc[0] * 1000):.2f} euros in the end date')

    if change.iloc[0] > best_return:
        best_return = change.iloc[0]
        best_company = i

    adj_close.plot()
    plt.show()

print(f'Company that provided best value for investment: {best_company}')
