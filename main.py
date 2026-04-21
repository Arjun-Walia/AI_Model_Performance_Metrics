import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import t
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load DataSet
df = pd.read_csv("ai_model_experiments.csv")
df.info()
df.describe()

# Data Cleaning and Preprocessing
print(df.isnull().sum())
print(df.duplicated().sum())

# Exploratory Data Analysis (EDA)
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
categorical_features = df.select_dtypes(include=["object"]).columns
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

plt.figure()
sns.histplot(df["accuracy"],bins=10, kde=True)
plt.title("Accuracy Distribution")

plt.figure()
sns.histplot(df["compute_cost_usd"], bins=30,kde=True)
plt.title("Compute Cost Distribution")

plt.figure()
sns.boxplot(y=df["memory_usage_gb"])
plt.title("Outlier Analysis: Memory Usage")

plt.figure()
sns.boxplot(y=df["tokens_per_second"])
plt.title("Outlier Analysis: Tokens per Second")

# Correlation Analysis
df_numerical = df[numerical_features]
plt.figure()
sns.heatmap(df_numerical.corr(), annot=True, cmap="inferno")
plt.title("Correlation Heatmap")

# IQR Testing
iqr_features = [col for col in numerical_features if col != "experiment_id"]
iqr_summary = []

for col in iqr_features:
    data = df[col].dropna()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    iqr_summary.append({
        "feature": col,
        "Q1": round(q1, 3),
        "Q3": round(q3, 3),
        "IQR": round(iqr, 3),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_count": int(outliers.shape[0]),
        "outlier_percent": round((outliers.shape[0] / data.shape[0]) * 100, 2),
    })

iqr_results = pd.DataFrame(iqr_summary).sort_values("outlier_percent", ascending=False)
print("IQR Outlier Summary by Numeric Feature")
print(iqr_results)

# Hypothesis Testing: Compare latency across GPU types
a100_latency = df.loc[df["gpu_type"] == "A100", "latency_ms"].dropna()
v100_latency = df.loc[df["gpu_type"] == "V100", "latency_ms"].dropna()

n1 = len(a100_latency)
n2 = len(v100_latency)

mean1 = np.mean(a100_latency)
mean2 = np.mean(v100_latency)

var1 = np.var(a100_latency, ddof=1)
var2 = np.var(v100_latency, ddof=1)

sp2 = ((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2)

se = np.sqrt(sp2 * (1/n1 + 1/n2))

t_stat = (mean1 - mean2) / se

df_simple = n1 + n2 - 2

p_value = 2 * (1 - t.cdf(abs(t_stat), df=df_simple))

print(f"A100 mean latency: {mean1:.2f} ms (n={n1})")
print(f"V100 mean latency: {mean2:.2f} ms (n={n2})")
print(f"T-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df_simple}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"Conclusion: Reject H0 at alpha={alpha}. GPU type has a statistically significant effect on latency.")
else:
    print(f"Conclusion: Fail to reject H0 at alpha={alpha}. No statistically significant difference in latency between GPU types.")

# Training Model
feature_cols = ["latency_ms", "tokens_per_second", "memory_usage_gb", "batch_size", "Model", "dataset", "gpu_type"]
x = pd.get_dummies(df[feature_cols], columns=["Model", "dataset", "gpu_type"], drop_first=True)
y = df["compute_cost_usd"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, Y_test)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

results = pd.DataFrame({"Actual": Y_test.values, "Predicted": y_pred})
print(results.head())

# Plotting Results
plt.figure(figsize=(7, 7))
plt.scatter(Y_test, y_pred, color='blue', alpha=0.6, s=25, label='Predicted vs Actual')

line_min = min(Y_test.min(), y_pred.min())
line_max = max(Y_test.max(), y_pred.max())
plt.plot([line_min, line_max], [line_min, line_max], color='red', label='Ideal Fit Line')

plt.xlabel("Actual Compute Cost (USD)")
plt.ylabel("Predicted Compute Cost (USD)")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)

# Visualizations
# Bar Chart: Average compute cost by model services vs others
model_services = ["GPT-4o", "Gemini-2.0-Pro", "Llama-3.1-8B", "Command"]
df_model_services = df[["Model", "compute_cost_usd"]].copy()
df_model_services["model_service_group"] = np.where(df_model_services["Model"].isin(model_services), df_model_services["Model"], "Others")

avg_cost_by_model_service = df_model_services.groupby("model_service_group")["compute_cost_usd"].mean().sort_values(ascending=False)
avg_cost_by_model_service.plot(kind='bar', color=["navy", "steelblue", "skyblue", "lightblue", "aliceblue"] )
plt.title("Average Compute Cost by Model Services")
plt.xlabel("Model Service Group")
plt.ylabel("Average Compute Cost (USD)")

# Line Plot: Mean accuracy by batch size
mean_accuracy_by_batch = df.groupby("batch_size")["accuracy"].mean().sort_index()
plt.figure()
plt.plot(mean_accuracy_by_batch.index, mean_accuracy_by_batch.values, marker="*")
plt.title("Mean Accuracy by Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Mean Accuracy")
plt.grid(True)

# Pie Chart: GPU type share in total experiments
gpu_count = df["gpu_type"].value_counts()
plt.figure()
plt.pie(gpu_count, labels=gpu_count.index, autopct="%1.2f%%", colors=["darkblue", "steelblue", "skyblue", "lightblue"] )
plt.title("GPU Type Distribution")

# Scatter Plot: Latency vs tokens per second by GPU type
plt.figure()
sns.scatterplot(x="latency_ms", y="tokens_per_second", hue="gpu_type", data=df)
plt.title("Latency vs Tokens per Second")

# Heat Map: Correlation among numeric performance features
plt.figure()
sns.heatmap(df[["accuracy", "latency_ms", "tokens_per_second", "memory_usage_gb", "compute_cost_usd"]].corr(), annot=True, cmap="inferno")
plt.title("Feature Correlation Heat Map")
