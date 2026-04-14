import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("Gaming and Mental Health.csv")
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
print(df.isnull().sum())
df = df.dropna()

# Outlier visualization (Before)
plt.boxplot(df['daily_gaming_hours'])
plt.title("Outliers Before Removal (Gaming Hours)")
plt.show()

# Outlier Detection (IQR Method)
Q1 = df['daily_gaming_hours'].quantile(0.25)
Q3 = df['daily_gaming_hours'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['daily_gaming_hours'] < lower) | (df['daily_gaming_hours'] > upper)]
print("Outliers:\n", outliers)

# Remove Outliers
df = df[(df['daily_gaming_hours'] >= lower) & (df['daily_gaming_hours'] <= upper)]

# Verify Outliers Removed
plt.boxplot(df['daily_gaming_hours'])
plt.title("Outliers After Removal")
plt.show()

# Objective 1: Gaming Time Distribution
plt.hist(df['daily_gaming_hours'])
plt.title("Gaming Hours Distribution")
plt.xlabel("Daily Gaming Hours")
plt.ylabel("Frequency")
plt.show()

# Objective 2: Gaming vs Mental Stress (Isolation)
plt.scatter(df['daily_gaming_hours'], df['social_isolation_score'])
plt.title("Gaming Hours vs Social Isolation")
plt.xlabel("Gaming Hours")
plt.ylabel("Isolation Score")
plt.show()

# Objective 3: Gaming VS Sleep Quality
plt.scatter(df['daily_gaming_hours'], df['sleep_quality'])
plt.title("Gaming Hours vs Sleep Quality")
plt.xlabel("Gaming Hours")
plt.ylabel("Sleep Quality")
plt.show()

# Objective 4: Hypothesis Testing (T-test)
high = df[df['daily_gaming_hours'] > df['daily_gaming_hours'].mean()]['social_isolation_score']
low = df[df['daily_gaming_hours'] <= df['daily_gaming_hours'].mean()]['social_isolation_score']

t_stat, p_val = ttest_ind(high, low)

print("T-Statistic:", t_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Reject H0 → Gaming affects isolation")
else:
    print("Fail to Reject H0 → No significant effect")

# Objective 5: Linear Regression
X = df[['daily_gaming_hours']]
y = df['social_isolation_score']

model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Regression Visualization
plt.scatter(df['daily_gaming_hours'], df['social_isolation_score'])
plt.plot(df['daily_gaming_hours'], model.predict(X))
plt.title("Linear Regression: Gaming vs Isolation")
plt.xlabel("Gaming Hours")
plt.ylabel("Isolation Score")
plt.show()