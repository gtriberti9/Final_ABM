from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, size, mean, percentile_approx, collect_list
import matplotlib.pyplot as plt
import pandas as pd

# --- Spark Setup ---
spark = SparkSession.builder.appName("ABMAnalysis").getOrCreate()

# --- Load Data ---
# Replace with your S3 path or local path
DATA_PATH = "s3://your-bucket/abm-results/*.json"  # or "file:///path/to/your/data/*.json"
df = spark.read.json(DATA_PATH)

# --- Convert columns to correct types if needed ---
for c in ['informality_rate', 'seed', 'time_to_target', 'inflation_volatility', 'max_deviation']:
    df = df.withColumn(c, col(c).cast("double"))

# --- 1. Distribution of time_to_target by informality_rate ---
box_data = df.select("informality_rate", "seed", "time_to_target").toPandas()
plt.figure(figsize=(10,6))
box_data.boxplot(column="time_to_target", by="informality_rate")
plt.title("Distribution of Equilibrium Steps by Informality Rate")
plt.suptitle("")
plt.xlabel("Informality Rate")
plt.ylabel("Time to Target")
plt.savefig("spark_equilibrium_steps_distribution.png")
plt.show()

# --- 2. Distribution of inflation_volatility by informality_rate ---
vol_data = df.select("informality_rate", "seed", "inflation_volatility").toPandas()
plt.figure(figsize=(10,6))
vol_data.boxplot(column="inflation_volatility", by="informality_rate")
plt.title("Distribution of Inflation Volatility by Informality Rate")
plt.suptitle("")
plt.xlabel("Informality Rate")
plt.ylabel("Inflation Volatility")
plt.savefig("spark_inflation_volatility_distribution.png")
plt.show()

# --- 3. Distribution of max_deviation by informality_rate ---
dev_data = df.select("informality_rate", "seed", "max_deviation").toPandas()
plt.figure(figsize=(10,6))
dev_data.boxplot(column="max_deviation", by="informality_rate")
plt.title("Distribution of Max Deviation by Informality Rate")
plt.suptitle("")
plt.xlabel("Informality Rate")
plt.ylabel("Max Deviation")
plt.savefig("spark_max_deviation_distribution.png")
plt.show()

# --- 4. Distribution of final inflation value by informality_rate ---
if "inflation_history" in df.columns:
    # Get the last value of inflation_history for each run
    df = df.withColumn("final_inflation", col("inflation_history")[size(col("inflation_history"))-1])
    final_inf = df.select("informality_rate", "seed", "final_inflation").toPandas()
    plt.figure(figsize=(10,6))
    final_inf.boxplot(column="final_inflation", by="informality_rate")
    plt.title("Distribution of Final Inflation by Informality Rate")
    plt.suptitle("")
    plt.xlabel("Informality Rate")
    plt.ylabel("Final Inflation (%)")
    plt.savefig("spark_final_inflation_distribution.png")
    plt.show()

# --- 5. Mean/Median Inflation Series by Informality Rate (if stored) ---
if "inflation_history" in df.columns:
    # Explode inflation_history for time series analysis
    exploded = df.select("informality_rate", "seed", explode(col("inflation_history")).alias("inflation"),).withColumn("timestep", col("pos"))
    # Group by informality_rate and timestep, compute mean inflation
    # (If explode doesn't give timestep, you may need to zip with index in your data)
    # For simplicity, collect and plot in pandas:
    pandas_df = df.select("informality_rate", "inflation_history").toPandas()
    plt.figure(figsize=(12,7))
    for rate in sorted(pandas_df["informality_rate"].unique()):
        histories = pandas_df[pandas_df["informality_rate"]==rate]["inflation_history"]
        # Convert to lists of floats
        series = []
        for h in histories:
            if isinstance(h, str):
                h = eval(h)
            series.append([float(x) for x in h])
        maxlen = max(len(s) for s in series)
        padded = [s + [None]*(maxlen-len(s)) for s in series]
        mean_series = [pd.Series([row[i] for row in padded if row[i] is not None]).mean() for i in range(maxlen)]
        plt.plot(mean_series, label=f"Informality {rate:.2f}")
    plt.xlabel("Time Steps")
    plt.ylabel("Inflation (%)")
    plt.title("Average Inflation Path by Informality Rate")
    plt.legend()
    plt.savefig("spark_inflation_series_by_informality.png")
    plt.show()

spark.stop()