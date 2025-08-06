import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("crowd_log.csv")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["Timestamp (s)"], df["People Count"], marker='o', linestyle='-', color='blue')
plt.title("Crowd Count Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("People Count")
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
