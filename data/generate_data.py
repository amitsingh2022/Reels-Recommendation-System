import pandas as pd
import numpy as np
import time

NUM_USERS = 5000
NUM_REELS = 2000
NUM_INTERACTIONS = 150_000

np.random.seed(42)

users = np.random.randint(1, NUM_USERS + 1, NUM_INTERACTIONS)
reels = np.random.randint(1, NUM_REELS + 1, NUM_INTERACTIONS)

watch_time = np.random.exponential(scale=15, size=NUM_INTERACTIONS)
liked = (watch_time > 10).astype(int)
shared = (watch_time > 20).astype(int)
commented = (watch_time > 25).astype(int)

timestamps = np.random.randint(
    int(time.time()) - 60*60*24*30,
    int(time.time()),
    NUM_INTERACTIONS
)

df = pd.DataFrame({
    "user_id": users,
    "reel_id": reels,
    "watch_time": watch_time,
    "liked": liked,
    "shared": shared,
    "commented": commented,
    "timestamp": timestamps
})

df.to_csv("data/interactions.csv", index=False)
print("✅ interactions.csv generated")
