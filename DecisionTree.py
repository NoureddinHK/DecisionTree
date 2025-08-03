import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import os

# Define functions for entropy, Gini index, and information gain
def entropy(y):
    """Calculate entropy of a target variable."""
    if not np.all(np.isclose(y, np.round(y))):
        print(f"Error in entropy: y contains non-integer values: {np.unique(y)}")
        raise ValueError("Non-integer values in y for entropy calculation")
    y = y.astype(np.int64, casting='unsafe')
    counts = np.bincount(y)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def gini_index(y):
    """Calculate Gini index of a target variable."""
    if not np.all(np.isclose(y, np.round(y))):
        print(f"Error in gini_index: y contains non-integer values: {np.unique(y)}")
        raise ValueError("Non-integer values in y for Gini index calculation")
    y = y.astype(np.int64, casting='unsafe')
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def information_gain(X, y, feature_idx):
    """Calculate information gain for a feature."""
    if not np.all(np.isclose(X[:, feature_idx], np.round(X[:, feature_idx]))):
        print(f"Error in information_gain: feature {feature_idx} contains non-integer values: {np.unique(X[:, feature_idx])}")
        raise ValueError(f"Non-integer values in feature {feature_idx}")
    X_col = X[:, feature_idx].astype(np.int64, casting='unsafe')
    parent_entropy = entropy(y)
    n = len(y)
    values = np.unique(X_col)
    weighted_child_entropy = 0
    
    for value in values:
        subset_indices = X_col == value
        subset_y = y[subset_indices]
        if len(subset_y) > 0:
            weight = len(subset_y) / n
            weighted_child_entropy += weight * entropy(subset_y)
    
    return parent_entropy - weighted_child_entropy

# Read the original dataset
try:
    df = pd.read_csv('OnlineNewsPopularity.csv')
except FileNotFoundError:
    print("فایل 'OnlineNewsPopularity.csv' یافت نشد. لطفاً مطمئن شوید که فایل در پوشه درست آپلود شده است.")
    exit()

# Print column names for inspection
print("ستون‌های دیتاست:")
print(df.columns)

# Drop non-numeric 'url' column if it exists
if 'url' in df.columns:
    df = df.drop(columns=['url'])
else:
    print("ستون 'url' یافت نشد.")

# Check for 'shares' column
if 'shares' not in df.columns:
    possible_columns = [col for col in df.columns if 'shares' in col.lower()]
    if possible_columns:
        print(f"ستون 'shares' یافت نشد، اما ستون‌های مشابه پیدا شدند: {possible_columns}")
        target_column = possible_columns[0]
        print(f"استفاده از ستون '{target_column}' به عنوان ستون هدف.")
    else:
        print("هیچ ستونی مرتبط با 'shares' یافت نشد. لطفاً فایل CSV را بررسی کنید.")
        exit()
else:
    target_column = 'shares'

# Select features and target
X_original = df.drop(columns=[target_column])
y_original = df[target_column]

# Extract numeric column names
feature_columns = X_original.columns.tolist()

# Check if there are enough numeric columns
if len(feature_columns) < 20:
    print(f"خطا: دیتاست اصلی تنها {len(feature_columns)} ستون عددی دارد، اما 20 ستون مورد نیاز است.")
    exit()

# Select first 20 columns
feature_columns = feature_columns[:20]

# Convert 'shares' to binary classification (threshold 1400)
y_binary = (y_original >= 1400).astype(np.int64)  # Use pandas astype without 'casting'

# Check number of classes
n_classes = len(np.unique(y_binary))
print(f"تعداد کلاس‌ها: {n_classes}")
print(f"تDistribution of classes in original dataset:\n{pd.Series(y_binary).value_counts()}")

# Generate new dataset with make_classification
X_new, y_new = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=n_classes,
    random_state=42,
    weights=np.bincount(y_binary) / len(y_binary)
)

# Validate and force integer casting for y_new
print(f"y_new dtype before casting: {y_new.dtype}")
print(f"Unique values in y_new: {np.unique(y_new)}")
if not np.all(np.isclose(y_new, np.round(y_new))):
    print("خطا: مقادیر غیرصحیح (غیرصفر یا یک) در y_new یافت شد.")
    exit()
y_new = y_new.astype(np.int64, casting='unsafe')

# Standardize data
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

# Discretize features using quantile strategy
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
X_discretized = discretizer.fit_transform(X_new)

# Ensure X_discretized is integer type
X_discretized = X_discretized.astype(np.int64, casting='unsafe')
print(f"X_discretized dtype after casting: {X_discretized.dtype}")

# Create DataFrame for discretized dataset
df_discretized = pd.DataFrame(X_discretized, columns=feature_columns)
df_discretized['shares'] = y_new

# Split dataset into train (80%) and test (20%)
train_df, test_df = train_test_split(df_discretized, test_size=0.2, random_state=42, stratify=df_discretized['shares'])

# Save train and test datasets with error handling
def save_with_fallback(df, filename, fallback_dir='~'):
    try:
        df.to_csv(filename, index=False)
        print(f"فایل '{filename}' با موفقیت ذخیره شد.")
    except PermissionError:
        print(f"خطا: عدم دسترسی برای ذخیره فایل '{filename}'.")
        fallback_path = os.path.join(os.path.expanduser(fallback_dir), filename)
        try:
            df.to_csv(fallback_path, index=False)
            print(f"فایل در مسیر جایگزین ذخیره شد: '{fallback_path}'.")
        except Exception as e:
            print(f"خطا در ذخیره فایل در مسیر جایگزین: {e}")
            print("لطفاً یک مسیر با دسترسی نوشتن مناسب مشخص کنید.")

save_with_fallback(train_df, 'Train_Discretized_SyntheticNewsPopularity.csv')
save_with_fallback(test_df, 'Test_Discretized_SyntheticNewsPopularity.csv')

# Display information about the discretized dataset
print("\nاطلاعات دیتاست گسسته‌شده (کل):")
print(f"تعداد نمونه‌ها: {df_discretized.shape[0]}")
print(f"تعداد ویژگی‌ها: {df_discretized.shape[1] - 1}")
print(f"تDistribution of classes:\n{pd.Series(df_discretized['shares']).value_counts()}")

# Display information about the training dataset
print("\nاطلاعات دیتاست آموزش (80%):")
print(f"تعداد نمونه‌ها: {train_df.shape[0]}")
print(f"تDistribution of classes:\n{pd.Series(train_df['shares']).value_counts()}")

# Display information about the test dataset
print("\nاطلاعات دیتاست آزمایش (20%):")
print(f"تعداد نمونه‌ها: {test_df.shape[0]}")
print(f"تDistribution of classes:\n{pd.Series(test_df['shares']).value_counts()}")

# Display bin edges for each feature
for i, feature in enumerate(feature_columns):
    print(f"\nویژگی {feature}:")
    print(f"نقاط برش: {discretizer.bin_edges_[i]}")
    print(f"تعداد بازه‌ها: {len(discretizer.bin_edges_[i]) - 1}")

# Calculate and display entropy, Gini index, and information gain
print("\nمعیارهای ارزیابی ویژگی‌ها در دیتاست آموزش:")
train_X = train_df[feature_columns].values
train_y = train_df['shares'].values
print(f"train_X dtype: {train_X.dtype}")
print(f"train_y dtype: {train_y.dtype}")
train_X = train_X.astype(np.int64, casting='unsafe')
train_y = train_y.astype(np.int64, casting='unsafe')
overall_entropy = entropy(train_y)
overall_gini = gini_index(train_y)
print(f"\nانتروپی کلی دیتاست آموزش: {overall_entropy:.4f}")
print(f"شاخص جینی کلی دیتاست آموزش: {overall_gini:.4f}")
print("\nارزیابی ویژگی‌ها:")
print(f"{'ویژگی':<30} {'انتروپی':<15} {'شاخص جینی':<15} {'بهره اطلاعات':<15}")
print("-" * 75)
for i, feature in enumerate(feature_columns):
    feature_entropy = entropy(train_X[:, i])
    feature_gini = gini_index(train_X[:, i])
    feature_ig = information_gain(train_X, train_y, i)
    print(f"{feature:<30} {feature_entropy:.4f} {'':<8} {feature_gini:.4f} {'':<6} {feature_ig:.4f}")