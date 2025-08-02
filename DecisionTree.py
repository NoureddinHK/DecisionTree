import pandas as pd
import numpy as np

# Function to calculate entropy for a given target series
def calculate_entropy(y):
    # Count the frequency of each class
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    # Calculate entropy: -sum(p * log2(p))
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

# Function to calculate conditional entropy for a feature
def calculate_conditional_entropy(df, feature, target='shares'):
    # Get unique values (bins) of the feature
    unique_values = df[feature].unique()
    total_samples = len(df)
    conditional_entropy = 0
    
    # Calculate entropy for each bin
    for value in unique_values:
        subset = df[df[feature] == value]
        if len(subset) == 0:
            continue
        subset_entropy = calculate_entropy(subset[target])
        # Weight by the proportion of samples in this bin
        weight = len(subset) / total_samples
        conditional_entropy += weight * subset_entropy
    
    return conditional_entropy

# Function to calculate information gain
def calculate_information_gain(df, feature, target='shares'):
    # Calculate overall entropy
    overall_entropy = calculate_entropy(df[target])
    # Calculate conditional entropy
    cond_entropy = calculate_conditional_entropy(df, feature, target)
    # Information gain = overall entropy - conditional entropy
    return overall_entropy - cond_entropy

# Load the training dataset
try:
    train_df = pd.read_csv('Train_Discretized_SyntheticNewsPopularity.csv')
except FileNotFoundError:
    print("فایل 'Train_Discretized_SyntheticNewsPopularity.csv' یافت نشد.")
    exit()

# Verify data integrity
print("بررسی دیتاست آموزش:")
print(f"تعداد نمونه‌ها: {train_df.shape[0]}")
print(f"تعداد ویژگی‌ها: {train_df.shape[1] - 1}")
print(f"توزیع کلاس‌ها:\n{pd.Series(train_df['shares']).value_counts()}")

# Check for missing values
if train_df.isnull().sum().sum() > 0:
    print("هشدار: مقادیر گمشده در دیتاست یافت شد. لطفاً داده‌ها را بررسی کنید.")
    exit()

# Get feature columns (excluding 'shares')
feature_columns = [col for col in train_df.columns if col != 'shares']

# Calculate overall entropy of the target variable
overall_entropy = calculate_entropy(train_df['shares'])
print(f"\nآنتروپی کلی دیتاست (برای ستون 'shares'): {overall_entropy:.4f}")

# Calculate conditional entropy and information gain for each feature
print("\nآنتروپی شرطی و سود اطلاعاتی برای هر ویژگی:")
info_gains = {}
for feature in feature_columns:
    cond_entropy = calculate_conditional_entropy(train_df, feature)
    info_gain = calculate_information_gain(train_df, feature)
    info_gains[feature] = info_gain
    print(f"\nویژگی {feature}:")
    print(f"آنتروپی شرطی: {cond_entropy:.4f}")
    print(f"سود اطلاعاتی: {info_gain:.4f}")

# Find the feature with the highest information gain
best_feature = max(info_gains, key=info_gains.get)
print(f"\nبهترین ویژگی برای تقسیم اولیه: {best_feature} (سود اطلاعاتی: {info_gains[best_feature]:.4f})")

# Optional: Visualize the distribution of the best feature
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.countplot(x=best_feature, hue='shares', data=train_df)
plt.title(f'Distribution of {best_feature} by Shares')
plt.xlabel(best_feature)
plt.ylabel('Count')
plt.legend(title='Shares', labels=['Not Popular (0)', 'Popular (1)'])
plt.show()