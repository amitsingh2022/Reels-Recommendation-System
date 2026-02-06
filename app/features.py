import pandas as pd


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    user_features = (
        df.groupby("user_id")
        .agg(
            avg_watch_time=("watch_time", "mean"),
            likes=("liked", "sum"),
            shares=("shared", "sum"),
            comments=("commented", "sum"),
        )
        .reset_index()
    )
    return user_features


def build_reel_features(df: pd.DataFrame) -> pd.DataFrame:
    reel_features = (
        df.groupby("reel_id")
        .agg(
            avg_watch_time=("watch_time", "mean"),
            likes=("liked", "sum"),
            shares=("shared", "sum"),
            comments=("commented", "sum"),
        )
        .reset_index()
    )
    return reel_features


def save_features(df: pd.DataFrame) -> None:
    user_features = build_user_features(df)
    reel_features = build_reel_features(df)

    user_features.to_csv("data/processed/user_features.csv", index=False)
    reel_features.to_csv("data/processed/reel_features.csv", index=False)

    print("✅ User & reel features saved")
