import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

nifty = yf.download("^NSEI", start="2014-01-01", end="2024-06-30")
nifty.to_csv("nifty.csv")

#print(nifty.head())

df = pd.read_csv("nifty.csv", skiprows=[1,2], index_col=0, parse_dates=True)

log_returns = np.log(df['Close'] / df['Close'].shift(1))

df['returns'] = log_returns

df.dropna(inplace=True)

#print(df.head())

df['return_squared'] = df['returns'] ** 2
X = np.column_stack((df['returns'], df['return_squared']))


scaler = StandardScaler()
X = scaler.fit_transform(X)
model = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X)
regimes = model.predict(X)
df['Regime_new'] = regimes

df.to_csv("regime_labels2.csv")
regime_names = {0: "Bull", 1: "Bear", 2: "HighVol", 3: "Crash"}
df['Regime_label'] = df['Regime_new'].map(regime_names)