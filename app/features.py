def build_user_features(df):
    user_features = df.groupby("user_id").agg({
        "watch_time": "mean",
        "liked": "sum",
        "shared": "sum",
        "commented": "sum"
    }).reset_index()

    return user_features

def build_reel_features(df):
    reel_features = df.groupby("reel_id").agg({
        "watch_time": "mean",
        "liked": "sum",
        "shared": "sum",
        "commented": "sum"
    }).reset_index()

    return reel_features

