import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import os

class NewsPopularityProcessor:
    """Class to process news popularity data and calculate feature metrics."""
    
    @staticmethod
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

    @staticmethod
    def gini_index(y):
        """Calculate Gini index of a target variable."""
        if not np.all(np.isclose(y, np.round(y))):
            print(f"Error in gini_index: y contains non-integer values: {np.unique(y)}")
            raise ValueError("Non-integer values in y for Gini index calculation")
        y = y.astype(np.int64, casting='unsafe')
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    @staticmethod
    def information_gain(X, y, feature_idx):
        """Calculate information gain for a feature."""
        if not np.all(np.isclose(X[:, feature_idx], np.round(X[:, feature_idx]))):
            print(f"Error in information_gain: feature {feature_idx} contains non-integer values: {np.unique(X[:, feature_idx])}")
            raise ValueError(f"Non-integer values in feature {feature_idx}")
        X_col = X[:, feature_idx].astype(np.int64, casting='unsafe')
        parent_entropy = NewsPopularityProcessor.entropy(y)
        n = len(y)
        values = np.unique(X_col)
        weighted_child_entropy = 0
        
        for value in values:
            subset_indices = X_col == value
            subset_y = y[subset_indices]
            if len(subset_y) > 0:
                weight = len(subset_y) / n
                weighted_child_entropy += weight * NewsPopularityProcessor.entropy(subset_y)
        
        return parent_entropy - weighted_child_entropy

    def __init__(self, dataset_path='OnlineNewsPopularity.csv', n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=42):
        """Initialize with dataset path and synthetic data parameters."""
        self.dataset_path = dataset_path
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.random_state = random_state
        self.feature_columns = None
        self.target_column = None
        self.df = None
        self.X_original = None
        self.y_original = None
        self.y_binary = None
        self.n_classes = None
        self.X_new = None
        self.y_new = None
        self.X_discretized = None
        self.df_discretized = None
        self.train_df = None
        self.test_df = None
        self.discretizer = None

    def load_data(self):
        """Load and preprocess the dataset."""
        try:
            self.df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print(f"فایل '{self.dataset_path}' یافت نشد. لطفاً مطمئن شوید که فایل در پوشه درست آپلود شده است.")
            exit()

        print("ستون‌های دیتاست:")
        print(self.df.columns)

        if 'url' in self.df.columns:
            self.df = self.df.drop(columns=['url'])
        else:
            print("ستون 'url' یافت نشد.")

        if 'shares' not in self.df.columns:
            possible_columns = [col for col in self.df.columns if 'shares' in col.lower()]
            if possible_columns:
                print(f"ستون 'shares' یافت نشد، اما ستون‌های مشابه پیدا شدند: {possible_columns}")
                self.target_column = possible_columns[0]
                print(f"استفاده از ستون '{self.target_column}' به عنوان ستون هدف.")
            else:
                print("هیچ ستونی مرتبط با 'shares' یافت نشد. لطفاً فایل CSV را بررسی کنید.")
                exit()
        else:
            self.target_column = 'shares'

        self.X_original = self.df.drop(columns=[self.target_column])
        self.y_original = self.df[self.target_column]

    def preprocess_data(self):
        """Preprocess data and binarize target."""
        self.feature_columns = self.X_original.columns.tolist()
        if len(self.feature_columns) < self.n_features:
            print(f"خطا: دیتاست اصلی تنها {len(self.feature_columns)} ستون عددی دارد، اما {self.n_features} ستون مورد نیاز است.")
            exit()

        self.feature_columns = self.feature_columns[:self.n_features]
        self.y_binary = (self.y_original >= 1400).astype(np.int64)

        self.n_classes = len(np.unique(self.y_binary))
        print(f"تعداد کلاس‌ها: {self.n_classes}")
        print(f"تDistribution of classes in original dataset:\n{pd.Series(self.y_binary).value_counts()}")

    def generate_synthetic_data(self):
        """Generate synthetic dataset."""
        self.X_new, self.y_new = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_classes=self.n_classes,
            random_state=self.random_state,
            weights=np.bincount(self.y_binary) / len(self.y_binary)
        )

        print(f"y_new dtype before casting: {self.y_new.dtype}")
        print(f"Unique values in y_new: {np.unique(self.y_new)}")
        if not np.all(np.isclose(self.y_new, np.round(self.y_new))):
            print("خطا: مقادیر غیرصحیح (غیرصفر یا یک) در y_new یافت شد.")
            exit()
        self.y_new = self.y_new.astype(np.int64, casting='unsafe')

    def standardize_and_discretize(self):
        """Standardize and discretize features."""
        scaler = StandardScaler()
        self.X_new = scaler.fit_transform(self.X_new)

        self.discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
        self.X_discretized = self.discretizer.fit_transform(self.X_new)
        self.X_discretized = self.X_discretized.astype(np.int64, casting='unsafe')
        print(f"X_discretized dtype after casting: {self.X_discretized.dtype}")

        self.df_discretized = pd.DataFrame(self.X_discretized, columns=self.feature_columns)
        self.df_discretized['shares'] = self.y_new

    def split_data(self):
        """Split dataset into train and test sets."""
        self.train_df, self.test_df = train_test_split(
            self.df_discretized, test_size=0.2, random_state=self.random_state, stratify=self.df_discretized['shares']
        )

    def save_datasets(self):
        """Save train and test datasets with error handling."""
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

        save_with_fallback(self.train_df, 'Train_Discretized_SyntheticNewsPopularity.csv')
        save_with_fallback(self.test_df, 'Test_Discretized_SyntheticNewsPopularity.csv')

    def display_dataset_info(self):
        """Display information about datasets."""
        print("\nاطلاعات دیتاست گسسته‌شده (کل):")
        print(f"تعداد نمونه‌ها: {self.df_discretized.shape[0]}")
        print(f"تعداد ویژگی‌ها: {self.df_discretized.shape[1] - 1}")
        print(f"تDistribution of classes:\n{pd.Series(self.df_discretized['shares']).value_counts()}")

        print("\nاطلاعات دیتاست آموزش (80%):")
        print(f"تعداد نمونه‌ها: {self.train_df.shape[0]}")
        print(f"تDistribution of classes:\n{pd.Series(self.train_df['shares']).value_counts()}")

        print("\nاطلاعات دیتاست آزمایش (20%):")
        print(f"تعداد نمونه‌ها: {self.test_df.shape[0]}")
        print(f"تDistribution of classes:\n{pd.Series(self.test_df['shares']).value_counts()}")

    def display_bin_edges(self):
        """Display bin edges for each feature."""
        for i, feature in enumerate(self.feature_columns):
            print(f"\nویژگی {feature}:")
            print(f"نقاط برش: {self.discretizer.bin_edges_[i]}")
            print(f"تعداد بازه‌ها: {len(self.discretizer.bin_edges_[i]) - 1}")

    def calculate_metrics(self):
        """Calculate and display entropy, Gini index, and information gain."""
        print("\nمعیارهای ارزیابی ویژگی‌ها در دیتاست آموزش:")
        train_X = self.train_df[self.feature_columns].values
        train_y = self.train_df['shares'].values
        print(f"train_X dtype: {train_X.dtype}")
        print(f"train_y dtype: {train_y.dtype}")
        train_X = train_X.astype(np.int64, casting='unsafe')
        train_y = train_y.astype(np.int64, casting='unsafe')
        overall_entropy = self.entropy(train_y)
        overall_gini = self.gini_index(train_y)
        print(f"\nانتروپی کلی دیتاست آموزش: {overall_entropy:.4f}")
        print(f"شاخص جینی کلی دیتاست آموزش: {overall_gini:.4f}")
        print("\nارزیابی ویژگی‌ها:")
        print(f"{'ویژگی':<30} {'انتروپی':<15} {'شاخص جینی':<15} {'بهره اطلاعات':<15}")
        print("-" * 75)
        for i, feature in enumerate(self.feature_columns):
            feature_entropy = self.entropy(train_X[:, i])
            feature_gini = self.gini_index(train_X[:, i])
            feature_ig = self.information_gain(train_X, train_y, i)
            print(f"{feature:<30} {feature_entropy:.4f} {'':<8} {feature_gini:.4f} {'':<6} {feature_ig:.4f}")

    def run(self):
        """Execute all processing steps."""
        self.load_data()
        self.preprocess_data()
        self.generate_synthetic_data()
        self.standardize_and_discretize()
        self.split_data()
        self.save_datasets()
        self.display_dataset_info()
        self.display_bin_edges()
        self.calculate_metrics()

if __name__ == "__main__":
    processor = NewsPopularityProcessor()
    processor.run()


    ######## Samples shuffle in dataset doesn't affect the results ########