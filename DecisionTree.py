import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import os

class NewsPopularityProcessor:
    """Class to process news popularity data, calculate feature metrics, and build/visualize a decision tree."""
    
    class Node:
        """Class to represent a node in the decision tree."""
        def __init__(self, feature_idx=None, split_value=None, left=None, right=None, label=None, gini=None, info_gain=None, n_samples=None):
            self.feature_idx = feature_idx  # Index of feature to split on (None for leaf)
            self.split_value = split_value  # Value to split on (None for leaf)
            self.left = left  # Left child node
            self.right = right  # Right child node
            self.label = label  # Class label (for leaf nodes)
            self.gini = gini  # Gini index of the node
            self.info_gain = info_gain  # Information gain of the split (None for leaf)
            self.n_samples = n_samples  # Number of samples at this node

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

    def __init__(self, dataset_path='OnlineNewsPopularity.csv', n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=42, max_depth=5, min_samples_split=2):
        """Initialize with dataset path, synthetic data parameters, and decision tree parameters."""
        self.dataset_path = dataset_path
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
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
        self.tree = None

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

        self.discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
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

    def perform_holdout_validation(self):
        """Perform holdout validation for hyper-parameter tuning of n_bins."""
        print("\nانجام اعتبارسنجی Holdout برای تنظیم تعداد بازه‌ها (n_bins):")
        train_sub_df, val_df = train_test_split(
            self.train_df, test_size=0.2, random_state=self.random_state, stratify=self.train_df['shares']
        )
        
        n_bins_options = [3, 4, 5, 6]
        results = []
        
        for n_bins in n_bins_options:
            scaler = StandardScaler()
            train_X_sub = scaler.fit_transform(train_sub_df[self.feature_columns])
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
            train_X_sub_discretized = discretizer.fit_transform(train_X_sub).astype(np.int64, casting='unsafe')
            
            val_X = scaler.transform(val_df[self.feature_columns])
            val_X_discretized = discretizer.transform(val_X).astype(np.int64, casting='unsafe')
            val_y = val_df['shares'].values.astype(np.int64, casting='unsafe')
            
            info_gains = []
            for i in range(len(self.feature_columns)):
                ig = self.information_gain(val_X_discretized, val_y, i)
                info_gains.append(ig)
            
            avg_info_gain = np.mean(info_gains)
            results.append((n_bins, avg_info_gain))
            print(f"n_bins={n_bins}, میانگین بهره اطلاعات روی مجموعه اعتبارسنجی: {avg_info_gain:.4f}")
        
        best_n_bins, best_avg_ig = max(results, key=lambda x: x[1])
        print(f"\nبهترین تعداد بازه‌ها: {best_n_bins} با میانگین بهره اطلاعات: {best_avg_ig:.4f}")
        return best_n_bins

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

    def find_best_split(self, X, y):
        """Find the best feature and split value based on information gain."""
        best_gain = -1
        best_feature_idx = None
        best_split_value = None
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            for value in values:
                gain = self.information_gain(X, y, feature_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_split_value = value
        
        return best_feature_idx, best_split_value, best_gain

    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        gini = self.gini_index(y)  # Calculate Gini index for the node
        
        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Create leaf node with majority class
            label = np.bincount(y.astype(np.int64)).argmax()
            return self.Node(label=label, gini=gini, n_samples=n_samples)
        
        # Find best split
        feature_idx, split_value, gain = self.find_best_split(X, y)
        if feature_idx is None or gain <= 0:
            # No valid split; create leaf node
            label = np.bincount(y.astype(np.int64)).argmax()
            return self.Node(label=label, gini=gini, n_samples=n_samples)
        
        # Split data
        left_indices = X[:, feature_idx] <= split_value
        right_indices = ~left_indices
        
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            # Invalid split; create leaf node
            label = np.bincount(y.astype(np.int64)).argmax()
            return self.Node(label=label, gini=gini, n_samples=n_samples)
        
        # Recursively build left and right subtrees
        left_child = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Create decision node
        return self.Node(feature_idx=feature_idx, split_value=split_value, left=left_child, right=right_child, gini=gini, info_gain=gain, n_samples=n_samples)

    def predict_single(self, x, node):
        """Predict class for a single sample by traversing the tree."""
        if node.label is not None:
            return node.label
        
        if x[node.feature_idx] <= node.split_value:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        """Predict classes for all samples."""
        return np.array([self.predict_single(x, self.tree) for x in X])

    def train_decision_tree(self):
        """Train the decision tree on the training data."""
        print("\nآموزش درخت تصمیم:")
        train_X = self.train_df[self.feature_columns].values.astype(np.int64, casting='unsafe')
        train_y = self.train_df['shares'].values.astype(np.int64, casting='unsafe')
        self.tree = self.build_tree(train_X, train_y)
        print("درخت تصمیم با موفقیت آموزش دید.")

    def evaluate_tree(self):
        """Evaluate the decision tree on the test data."""
        print("\nارزیابی درخت تصمیم:")
        test_X = self.test_df[self.feature_columns].values.astype(np.int64, casting='unsafe')
        test_y = self.test_df['shares'].values.astype(np.int64, casting='unsafe')
        predictions = self.predict(test_X)
        accuracy = np.mean(predictions == test_y)
        print(f"دقت درخت تصمیم روی مجموعه آزمایش: {accuracy:.4f}")

    def visualize_tree(self):
        """Print an enhanced text-based representation of the decision tree."""
        def print_node(node, depth=0, prefix="", is_last=True):
            """Recursively print the tree with enhanced formatting."""
            indent = "|  " * (depth - 1) + ("|-- " if depth > 0 else "")
            if is_last and depth > 0:
                indent = "|  " * (depth - 1) + "`-- "
            
            if node.label is not None:
                # Leaf node
                node_info = f"[Class: {node.label}, Gini: {node.gini:.4f}, Samples: {node.n_samples}]"
            else:
                # Decision node
                feature_name = self.feature_columns[node.feature_idx]
                node_info = f"[Feature: {feature_name}, IG: {node.info_gain:.4f}, Gini: {node.gini:.4f}, Samples: {node.n_samples}]"
            
            print(f"{indent}{node_info}")
            
            # Print children
            if node.left is not None:
                print_node(node.left, depth + 1, f"<= {node.split_value}", False)
            if node.right is not None:
                print_node(node.right, depth + 1, f"> {node.split_value}", True)

        print("\nنمایش متنی درخت تصمیم:")
        print_node(self.tree)

    def run(self):
        """Execute all processing steps, train/evaluate decision tree, and visualize it."""
        self.load_data()
        self.preprocess_data()
        self.generate_synthetic_data()
        self.standardize_and_discretize()
        self.split_data()
        self.perform_holdout_validation()
        self.save_datasets()
        self.display_dataset_info()
        self.display_bin_edges()
        self.calculate_metrics()
        self.train_decision_tree()
        self.evaluate_tree()
        self.visualize_tree()

if __name__ == "__main__":
    processor = NewsPopularityProcessor()
    processor.run()