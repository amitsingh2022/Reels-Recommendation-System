import pandas as pd
from app.features import save_features

def main():
    df = pd.read_csv("data/interactions.csv")
    save_features(df)

if __name__ == "__main__":
    main()
