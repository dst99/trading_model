import pandas as pd

# Load DX
dx = pd.read_csv("data/DX_4h.csv")
dx['time'] = pd.to_datetime(dx['time'], utc=True)
dx = dx.set_index('time')

# Load 6E
eur = pd.read_csv("data/6E_4h.csv")
eur['time'] = pd.to_datetime(eur['time'], utc=True)
eur = eur.set_index('time')

print("DX head:")
print(dx.head(), "\n")
print("6E head:")
print(eur.head(), "\n")

# Check alignment of first few dates
print("Earliest DX time:", dx.index.min())
print("Earliest 6E time:", eur.index.min())
print("Latest DX time:", dx.index.max())
print("Latest 6E time:", eur.index.max())
print("DX timezone:", dx.index.tz)
print("6E timezone:", eur.index.tz)
