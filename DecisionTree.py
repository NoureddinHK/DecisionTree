import pandas as pd  # وارد کردن کتابخانه پانداس برای کار با داده‌های جدولی
import numpy as np  # وارد کردن کتابخانه نامپای برای عملیات ریاضی و آرایه‌ها
from sklearn.datasets import make_classification  # وارد کردن تابع برای تولید داده‌های مصنوعی طبقه‌بندی
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer  # وارد کردن ابزارهای استانداردسازی و گسسته‌سازی داده‌ها
from sklearn.model_selection import train_test_split  # وارد کردن تابع برای تقسیم داده‌ها به آموزش و آزمایش
import os  # وارد کردن کتابخانه برای عملیات سیستم فایل مانند ذخیره فایل
import warnings  # وارد کردن کتابخانه برای مدیریت هشدارهای برنامه
import pydot  # وارد کردن کتابخانه برای ایجاد گراف‌های گرافویز

# Suppress all warnings
warnings.filterwarnings("ignore")  # نادیده گرفتن تمام هشدارها برای جلوگیری از خروجی اضافی

class NewsPopularityProcessor:  # تعریف کلاس اصلی برای پردازش داده‌های محبوبیت اخبار
    """Class to process news popularity data, calculate feature metrics, and build/visualize a decision tree."""  # توضیح مستند برای کلاس
    
    class Node:  # تعریف کلاس داخلی برای گره‌های درخت تصمیم
        """Class to represent a node in the decision tree."""  # توضیح مستند برای کلاس گره
        def __init__(self, feature_idx=None, split_value=None, left=None, right=None, label=None, gini=None, info_gain=None, n_samples=None):  # سازنده کلاس گره با پارامترهای پیش‌فرض
            self.feature_idx = feature_idx  # شاخص ویژگی برای تقسیم (None برای گره‌های برگ)
            self.split_value = split_value  # مقدار آستانه برای تقسیم (None برای گره‌های برگ)
            self.left = left  # اشاره‌گر به گره فرزند چپ
            self.right = right  # اشاره‌گر به گره فرزند راست
            self.label = label  # برچسب کلاس برای گره‌های برگ
            self.gini = gini  # شاخص جینی برای گره
            self.info_gain = info_gain  # بهره اطلاعات برای تقسیم (None برای گره‌های برگ)
            self.n_samples = n_samples  # تعداد نمونه‌های موجود در این گره

    @staticmethod
    def entropy(y):  # تعریف متد استاتیک برای محاسبه آنتروپی
        """Calculate entropy of a target variable."""  # توضیح مستند برای متد
        if not np.all(np.isclose(y, np.round(y))):  # بررسی اینکه آیا تمام مقادیر y اعداد صحیح هستند
            raise ValueError(f"Error in entropy: y contains non-integer values: {np.unique(y)}")  # پرتاب خطا در صورت وجود مقادیر غیرصحیح
        y = y.astype(np.int64, casting='unsafe')  # تبدیل مقادیر y به نوع integer 64-bit
        counts = np.bincount(y)  # شمارش تعداد نمونه‌ها در هر کلاس
        probs = counts / len(y)  # محاسبه احتمال هر کلاس
        probs = probs[probs > 0]  # حذف احتمال‌های صفر برای جلوگیری از خطا در لگاریتم
        return -np.sum(probs * np.log2(probs))  # محاسبه و بازگشت مقدار آنتروپی

    @staticmethod
    def gini_index(y):  # تعریف متد استاتیک برای محاسبه شاخص جینی
        """Calculate Gini index of a target variable."""  # توضیح مستند برای متد
        if not np.all(np.isclose(y, np.round(y))):  # بررسی مقادیر صحیح در y
            raise ValueError(f"Error in gini_index: y contains non-integer values: {np.unique(y)}")  # پرتاب خطا در صورت مقادیر غیرصحیح
        y = y.astype(np.int64, casting='unsafe')  # تبدیل y به نوع integer 64-bit
        counts = np.bincount(y)  # شمارش تعداد نمونه‌ها در هر کلاس
        probs = counts / len(y)  # محاسبه احتمال هر کلاس
        return 1 - np.sum(probs ** 2)  # محاسبه و بازگشت شاخص جینی

    @staticmethod
    def information_gain(X, y, feature_idx):  # تعریف متد استاتیک برای محاسبه بهره اطلاعات
        """Calculate information gain for a feature."""  # توضیح مستند برای متد
        if not np.all(np.isclose(X[:, feature_idx], np.round(X[:, feature_idx]))):  # بررسی مقادیر صحیح در ستون ویژگی
            raise ValueError(f"Error in information_gain: feature {feature_idx} contains non-integer values: {np.unique(X[:, feature_idx])}")  # پرتاب خطا در صورت مقادیر غیرصحیح
        X_col = X[:, feature_idx].astype(np.int64, casting='unsafe')  # تبدیل ستون ویژگی به نوع integer
        parent_entropy = NewsPopularityProcessor.entropy(y)  # محاسبه آنتروپی گره والد
        n = len(y)  # تعداد کل نمونه‌ها
        values = np.unique(X_col)  # استخراج مقادیر منحصر به فرد در ستون ویژگی
        weighted_child_entropy = 0  # مقدار اولیه برای آنتروپی وزنی فرزندان
        
        for value in values:  # حلقه روی مقادیر منحصر به فرد ویژگی
            subset_indices = X_col == value  # شاخص‌های نمونه‌هایی که مقدار ویژگی برابر value است
            subset_y = y[subset_indices]  # استخراج y برای زیرمجموعه
            if len(subset_y) > 0:  # اگر زیرمجموعه خالی نباشد
                weight = len(subset_y) / n  # محاسبه وزن زیرمجموعه
                weighted_child_entropy += weight * NewsPopularityProcessor.entropy(subset_y)  # اضافه کردن آنتروپی وزنی فرزندان
        
        return parent_entropy - weighted_child_entropy  # بازگشت بهره اطلاعات (آنتروپی والد منهای آنتروپی وزنی)

    def __init__(self, dataset_path='OnlineNewsPopularity.csv', n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=42, max_depth=5, min_samples_split=2):  # سازنده کلاس اصلی
        """Initialize with dataset path, synthetic data parameters, and decision tree parameters."""  # توضیح مستند برای سازنده
        self.dataset_path = dataset_path  # ذخیره مسیر فایل دیتاست
        self.n_samples = n_samples  # تعداد نمونه‌های مصنوعی برای تولید
        self.n_features = n_features  # تعداد ویژگی‌های مورد نیاز
        self.n_informative = n_informative  # تعداد ویژگی‌های اطلاع‌رسان
        self.n_redundant = n_redundant  # تعداد ویژگی‌های اضافی
        self.random_state = random_state  # بذر تصادفی برای تکرارپذیری
        self.max_depth = max_depth  # حداکثر عمق درخت تصمیم
        self.min_samples_split = min_samples_split  # حداقل تعداد نمونه برای تقسیم گره
        self.feature_columns = None  # لیست ستون‌های ویژگی (بعدا پر می‌شود)
        self.target_column = None  # ستون هدف (بعدا پر می‌شود)
        self.df = None  # دیتافریم اصلی (بعدا پر می‌شود)
        self.X_original = None  # ویژگی‌های اصلی دیتاست
        self.y_original = None  # مقادیر هدف اصلی
        self.y_binary = None  # هدف باینری (0 یا 1)
        self.n_classes = None  # تعداد کلاس‌های هدف
        self.X_new = None  # ویژگی‌های داده‌های مصنوعی
        self.y_new = None  # هدف داده‌های مصنوعی
        self.X_discretized = None  # ویژگی‌های گسسته‌شده
        self.df_discretized = None  # دیتافریم داده‌های گسسته‌شده
        self.train_df = None  # دیتافریم مجموعه آموزش
        self.test_df = None  # دیتافریم مجموعه آزمایش
        self.discretizer = None  # شیء گسسته‌سازی
        self.tree = None  # درخت تصمیم ساخته‌شده

    def load_data(self):  # متد برای بارگذاری و پیش‌پردازش دیتاست
        """Load and preprocess the dataset."""  # توضیح مستند برای متد
        try:  # شروع بلوک مدیریت خطا
            self.df = pd.read_csv(self.dataset_path)  # خواندن فایل CSV به دیتافریم
        except FileNotFoundError:  # در صورت عدم وجود فایل
            raise FileNotFoundError(f"File '{self.dataset_path}' not found. Please ensure the file is uploaded in the correct directory.")  # پرتاب خطا با پیام مناسب

        if 'url' in self.df.columns:  # بررسی وجود ستون 'url'
            self.df = self.df.drop(columns=['url'])  # حذف ستون 'url' اگر وجود داشته باشد

        if 'shares' not in self.df.columns:  # بررسی وجود ستون 'shares'
            possible_columns = [col for col in self.df.columns if 'shares' in col.lower()]  # جستجوی ستون‌های مشابه 'shares'
            if possible_columns:  # اگر ستون مشابه یافت شد
                self.target_column = possible_columns[0]  # انتخاب اولین ستون مشابه به عنوان هدف
            else:  # اگر ستون مشابه یافت نشد
                raise ValueError("No column related to 'shares' found. Please check the CSV file.")  # پرتاب خطا
        else:  # اگر ستون 'shares' وجود داشت
            self.target_column = 'shares'  # تنظیم ستون هدف به 'shares'

        self.X_original = self.df.drop(columns=[self.target_column])  # استخراج ویژگی‌ها با حذف ستون هدف
        self.y_original = self.df[self.target_column]  # استخراج مقادیر هدف

    def preprocess_data(self):  # متد برای پیش‌پردازش داده‌ها
        """Preprocess data and binarize target."""  # توضیح مستند برای متد
        self.feature_columns = self.X_original.columns.tolist()  # ذخیره لیست نام ستون‌های ویژگی
        if len(self.feature_columns) < self.n_features:  # بررسی تعداد کافی ستون‌های ویژگی
            raise ValueError(f"Error: Dataset has only {len(self.feature_columns)} numeric columns, but {self.n_features} are required.")  # پرتاب خطا در صورت کمبود ستون

        self.feature_columns = self.feature_columns[:self.n_features]  # محدود کردن ویژگی‌ها به تعداد مورد نظر
        self.y_binary = (self.y_original >= 1400).astype(np.int64)  # تبدیل مقادیر هدف به باینری (0 یا 1) با آستانه 1400
        self.n_classes = len(np.unique(self.y_binary))  # محاسبه تعداد کلاس‌های منحصر به فرد

    def generate_synthetic_data(self):  # متد برای تولید داده‌های مصنوعی
        """Generate synthetic dataset."""  # توضیح مستند برای متد
        self.X_new, self.y_new = make_classification(  # تولید داده‌های مصنوعی با استفاده از make_classification
            n_samples=self.n_samples,  # تعداد نمونه‌ها
            n_features=self.n_features,  # تعداد ویژگی‌ها
            n_informative=self.n_informative,  # تعداد ویژگی‌های اطلاع‌رسان
            n_redundant=self.n_redundant,  # تعداد ویژگی‌های اضافی
            n_classes=self.n_classes,  # تعداد کلاس‌ها
            random_state=self.random_state,  # بذر تصادفی
            weights=np.bincount(self.y_binary) / len(self.y_binary)  # وزن کلاس‌ها بر اساس توزیع اصلی
        )

        if not np.all(np.isclose(self.y_new, np.round(self.y_new))):  # بررسی مقادیر صحیح در y_new
            raise ValueError("Error: Invalid (non-zero or non-one) values found in y_new.")  # پرتاب خطا در صورت مقادیر نامعتبر
        self.y_new = self.y_new.astype(np.int64, casting='unsafe')  # تبدیل y_new به نوع integer

    def standardize_and_discretize(self):  # متد برای استانداردسازی و گسسته‌سازی ویژگی‌ها
        """Standardize and discretize features."""  # توضیح مستند برای متد
        scaler = StandardScaler()  # ایجاد شیء استانداردساز
        self.X_new = scaler.fit_transform(self.X_new)  # استانداردسازی داده‌های مصنوعی

        self.discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')  # ایجاد شیء گسسته‌سازی با 4 بین
        self.X_discretized = self.discretizer.fit_transform(self.X_new)  # گسسته‌سازی داده‌ها
        self.X_discretized = self.X_discretized.astype(np.int64, casting='unsafe')  # تبدیل داده‌های گسسته‌شده به integer

        self.df_discretized = pd.DataFrame(self.X_discretized, columns=self.feature_columns)  # ایجاد دیتافریم از داده‌های گسسته‌شده
        self.df_discretized['shares'] = self.y_new  # اضافه کردن ستون هدف به دیتافریم

    def split_data(self):  # متد برای تقسیم داده‌ها به آموزش و آزمایش
        """Split dataset into train and test sets."""  # توضیح مستند برای متد
        self.train_df, self.test_df = train_test_split(  # تقسیم داده‌ها به دو مجموعه
            self.df_discretized, test_size=0.2, random_state=self.random_state, stratify=self.df_discretized['shares']  # 20% آزمایش، با حفظ توزیع کلاس‌ها
        )

    def perform_holdout_validation(self):  # متد برای اعتبارسنجی هولدآوت
        """Perform holdout validation for hyper-parameter tuning of n_bins."""  # توضیح مستند برای متد
        train_sub_df, val_df = train_test_split(  # تقسیم مجموعه آموزش به زیرمجموعه‌های آموزش و اعتبارسنجی
            self.train_df, test_size=0.2, random_state=self.random_state, stratify=self.train_df['shares']  # 20% اعتبارسنجی
        )
        
        n_bins_options = [3, 4, 5, 6]  # گزینه‌های تعداد بین‌ها
        results = []  # لیست برای ذخیره نتایج
        
        for n_bins in n_bins_options:  # حلقه روی تعداد بین‌ها
            scaler = StandardScaler()  # ایجاد شیء استانداردساز
            train_X_sub = scaler.fit_transform(train_sub_df[self.feature_columns])  # استانداردسازی ویژگی‌های زیرمجموعه آموزش
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')  # ایجاد گسسته‌ساز
            train_X_sub_discretized = discretizer.fit_transform(train_X_sub).astype(np.int64, casting='unsafe')  # گسسته‌سازی داده‌های آموزش
            
            val_X = scaler.transform(val_df[self.feature_columns])  # استانداردسازی ویژگی‌های اعتبارسنجی
            val_X_discretized = discretizer.transform(val_X).astype(np.int64, casting='unsafe')  # گسسته‌سازی داده‌های اعتبارسنجی
            val_y = val_df['shares'].values.astype(np.int64, casting='unsafe')  # استخراج هدف اعتبارسنجی
            
            info_gains = []  # لیست برای ذخیره بهره‌های اطلاعات
            for i in range(len(self.feature_columns)):  # حلقه روی ویژگی‌ها
                ig = self.information_gain(val_X_discretized, val_y, i)  # محاسبه بهره اطلاعات برای هر ویژگی
                info_gains.append(ig)  # اضافه کردن به لیست
            
            avg_info_gain = np.mean(info_gains)  # محاسبه میانگین بهره اطلاعات
            results.append((n_bins, avg_info_gain))  # ذخیره نتیجه برای این تعداد بین
        
        best_n_bins, best_avg_ig = max(results, key=lambda x: x[1])  # انتخاب بهترین تعداد بین بر اساس میانگین بهره
        return best_n_bins  # بازگشت بهترین تعداد بین

    def save_datasets(self):  # متد برای ذخیره مجموعه‌های داده
        """Save train and test datasets with error handling."""  # توضیح مستند برای متد
        def save_with_fallback(df, filename, fallback_dir='~'):  # تابع داخلی برای ذخیره با مسیر جایگزین
            try:  # شروع بلوک مدیریت خطا
                df.to_csv(filename, index=False)  # ذخیره دیتافریم به CSV
            except PermissionError:  # در صورت خطای دسترسی
                fallback_path = os.path.join(os.path.expanduser(fallback_dir), filename)  # ایجاد مسیر جایگزین
                try:  # تلاش برای ذخیره در مسیر جایگزین
                    df.to_csv(fallback_path, index=False)  # ذخیره در مسیر جایگزین
                except Exception as e:  # در صورت خطای دیگر
                    raise RuntimeError(f"Error saving file in fallback path: {e}")  # پرتاب خطا

        save_with_fallback(self.train_df, 'Train_Discretized_SyntheticNewsPopularity.csv')  # ذخیره مجموعه آموزش
        save_with_fallback(self.test_df, 'Test_Discretized_SyntheticNewsPopularity.csv')  # ذخیره مجموعه آزمایش

    def display_dataset_info(self):  # متد برای نمایش اطلاعات مجموعه داده
        """Display information about datasets."""  # توضیح مستند برای متد
        pass  # خالی، چون خروجی غیرضروری حذف شده است

    def display_bin_edges(self):  # متد برای نمایش حدود بین‌ها
        """Display bin edges for each feature."""  # توضیح مستند برای متد
        pass  # خالی، چون خروجی غیرضروری حذف شده است

    def calculate_metrics(self):  # متد برای محاسبه معیارها
        """Calculate and display entropy, Gini index, and information gain."""  # توضیح مستند برای متد
        pass  # خالی، چون خروجی غیرضروری حذف شده است

    def find_best_split(self, X, y):  # متد برای یافتن بهترین تقسیم
        """Find the best feature and split value based on information gain."""  # توضیح مستند برای متد
        best_gain = -1  # مقدار اولیه برای بهترین بهره اطلاعات
        best_feature_idx = None  # شاخص بهترین ویژگی
        best_split_value = None  # بهترین مقدار تقسیم
        n_features = X.shape[1]  # تعداد ویژگی‌ها
        
        for feature_idx in range(n_features):  # حلقه روی تمام ویژگی‌ها
            values = np.unique(X[:, feature_idx])  # استخراج مقادیر منحصر به فرد ویژگی
            for value in values:  # حلقه روی مقادیر منحصر به فرد
                gain = self.information_gain(X, y, feature_idx)  # محاسبه بهره اطلاعات
                if gain > best_gain:  # اگر بهره بهتر از بهترین قبلی است
                    best_gain = gain  # به‌روزرسانی بهترین بهره
                    best_feature_idx = feature_idx  # به‌روزرسانی شاخص ویژگی
                    best_split_value = value  # به‌روزرسانی مقدار تقسیم
        
        return best_feature_idx, best_split_value, best_gain  # بازگشت بهترین مقادیر

    def build_tree(self, X, y, depth=0):  # متد برای ساخت درخت تصمیم به صورت بازگشتی
        """Recursively build the decision tree."""  # توضیح مستند برای متد
        n_samples = len(y)  # تعداد نمونه‌ها
        n_classes = len(np.unique(y))  # تعداد کلاس‌های منحصر به فرد
        gini = self.gini_index(y)  # محاسبه شاخص جینی برای گره
        
        # Stopping criteria
        if (depth >= self.max_depth or  # اگر عمق از حداکثر بیشتر شد
            n_samples < self.min_samples_split or  # یا تعداد نمونه‌ها کمتر از حداقل است
            n_classes == 1):  # یا فقط یک کلاس وجود دارد
            # Create leaf node with majority class
            label = np.bincount(y.astype(np.int64)).argmax()  # انتخاب کلاس اکثریت
            return self.Node(label=label, gini=gini, n_samples=n_samples)  # بازگشت گره برگ
        
        # Find best split
        feature_idx, split_value, gain = self.find_best_split(X, y)  # یافتن بهترین تقسیم
        if feature_idx is None or gain <= 0:  # اگر تقسیم معتبر نبود
            # No valid split; create leaf node
            label = np.bincount(y.astype(np.int64)).argmax()  # انتخاب کلاس اکثریت
            return self.Node(label=label, gini=gini, n_samples=n_samples)  # بازگشت گره برگ
        
        # Split data
        left_indices = X[:, feature_idx] <= split_value  # شاخص‌های نمونه‌های سمت چپ
        right_indices = ~left_indices  # شاخص‌های نمونه‌های سمت راست
        
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:  # اگر یکی از زیرمجموعه‌ها خالی باشد
            # Invalid split; create leaf node
            label = np.bincount(y.astype(np.int64)).argmax()  # انتخاب کلاس اکثریت
            return self.Node(label=label, gini=gini, n_samples=n_samples)  # بازگشت گره برگ
        
        # Recursively build left and right subtrees
        left_child = self.build_tree(X[left_indices], y[left_indices], depth + 1)  # ساخت زیرشاخه چپ
        right_child = self.build_tree(X[right_indices], y[right_indices], depth + 1)  # ساخت زیرشاخه راست
        
        # Create decision node
        return self.Node(feature_idx=feature_idx, split_value=split_value, left=left_child, right=right_child, gini=gini, info_gain=gain, n_samples=n_samples)  # بازگشت گره تصمیم

    def prune_tree(self, node):  # متد برای هرس کردن درخت پس از ساخت
        """Post-pruning: recursively prune the tree if both children are leaves with the same label."""  # توضیح مستند برای متد
        if node.label is not None:  # اگر گره برگ است
            return node  # بازگشت گره بدون تغییر
        
        # Recursively prune children
        node.left = self.prune_tree(node.left)  # هرس بازگشتی فرزند چپ
        node.right = self.prune_tree(node.right)  # هرس بازگشتی فرزند راست
        
        # If both children are leaves with the same label, replace with a leaf
        if node.left.label is not None and node.right.label is not None and node.left.label == node.right.label:  # اگر هر دو فرزند برگ با برچسب یکسان باشند
            node.feature_idx = None  # حذف شاخص ویژگی
            node.split_value = None  # حذف مقدار تقسیم
            node.label = node.left.label  # تنظیم برچسب به برچسب فرزند
            node.gini = 0.0  # تنظیم جینی به 0 (گره خالص)
            node.info_gain = None  # حذف بهره اطلاعات
            node.n_samples = node.n_samples  # حفظ تعداد نمونه‌ها
            node.left = None  # حذف فرزند چپ
            node.right = None  # حذف فرزند راست
        
        return node  # بازگشت گره هرس‌شده

    def predict_single(self, x, node):  # متد برای پیش‌بینی یک نمونه
        """Predict class for a single sample by traversing the tree."""  # توضیح مستند برای متد
        if node.label is not None:  # اگر گره برگ است
            return node.label  # بازگشت برچسب کلاس
        
        if x[node.feature_idx] <= node.split_value:  # اگر مقدار ویژگی کمتر یا برابر آستانه باشد
            return self.predict_single(x, node.left)  # پیش‌بینی با فرزند چپ
        else:  # در غیر این صورت
            return self.predict_single(x, node.right)  # پیش‌بینی با فرزند راست

    def predict(self, X):  # متد برای پیش‌بینی مجموعه نمونه‌ها
        """Predict classes for all samples."""  # توضیح مستند برای متد
        return np.array([self.predict_single(x, self.tree) for x in X])  # پیش‌بینی برای هر نمونه و تبدیل به آرایه

    def train_decision_tree(self):  # متد برای آموزش درخت تصمیم
        """Train the decision tree on the test data."""  # توضیح مستند برای متد
        test_X = self.test_df[self.feature_columns].values.astype(np.int64, casting='unsafe')  # استخراج ویژگی‌های مجموعه آزمایش
        test_y = self.test_df['shares'].values.astype(np.int64, casting='unsafe')  # استخراج هدف مجموعه آزمایش
        self.tree = self.build_tree(test_X, test_y)  # ساخت درخت تصمیم
        self.tree = self.prune_tree(self.tree)  # هرس کردن درخت

    def evaluate_tree(self):  # متد برای ارزیابی درخت تصمیم
        """Evaluate the decision tree on the test data."""  # توضیح مستند برای متد
        test_X = self.test_df[self.feature_columns].values.astype(np.int64, casting='unsafe')  # استخراج ویژگی‌های آزمایش
        test_y = self.test_df['shares'].values.astype(np.int64, casting='unsafe')  # استخراج هدف آزمایش
        predictions = self.predict(test_X)  # پیش‌بینی برای مجموعه آزمایش
        accuracy = np.mean(predictions == test_y)  # محاسبه دقت پیش‌بینی‌ها

    def visualize_tree(self):  # متد برای نمایش متنی درخت تصمیم
        """Print an enhanced text-based representation of the decision tree."""  # توضیح مستند برای متد
        def print_node(node, depth=0, prefix="", is_last=True):  # تابع داخلی برای چاپ بازگشتی گره‌ها
            """Recursively print the tree with enhanced formatting."""  # توضیح مستند برای تابع
            indent = "|  " * (depth - 1) + ("|-- " if depth > 0 else "")  # ایجاد فرمت فاصله برای چاپ
            if is_last and depth > 0:  # اگر گره آخرین فرزند در عمق باشد
                indent = "|  " * (depth - 1) + "`-- "  # استفاده از فرمت متفاوت برای آخرین گره
            
            if node.label is not None:  # اگر گره برگ است
                # Leaf node
                node_info = f"[Class: {node.label}, Gini: {node.gini:.4f}, Samples: {node.n_samples}]"  # اطلاعات گره برگ
            else:  # اگر گره تصمیم است
                # Decision node
                feature_name = self.feature_columns[node.feature_idx]  # نام ویژگی
                node_info = f"[Feature: {feature_name}, IG: {node.info_gain:.4f}, Gini: {node.gini:.4f}, Samples: {node.n_samples}]"  # اطلاعات گره تصمیم
            
            print(f"{indent}{node_info}")  # چاپ اطلاعات گره
            
            # Print children
            if node.left is not None:  # اگر فرزند چپ وجود دارد
                print_node(node.left, depth + 1, f"<= {node.split_value}", False)  # چاپ فرزند چپ
            if node.right is not None:  # اگر فرزند راست وجود دارد
                print_node(node.right, depth + 1, f"> {node.split_value}", True)  # چاپ فرزند راست

        print_node(self.tree)  # شروع چاپ درخت از گره ریشه

    def visualize_tree_graphviz(self):  # متد برای نمایش گرافیکی درخت با گرافویز
        """Generate a Graphviz-based visualization of the decision tree and save as PNG."""  # توضیح مستند برای متد
        def add_node_to_graph(graph, node, node_id, parent_id=None, edge_label=None):  # تابع داخلی برای اضافه کردن گره‌ها به گراف
            """Recursively add nodes and edges to the Graphviz graph."""  # توضیح مستند برای تابع
            if node.label is not None:  # اگر گره برگ است
                # Leaf node
                label = f"Class: {node.label}\\nGini: {node.gini:.4f}\\nSamples: {node.n_samples}"  # برچسب گره برگ
                node_color = "lightgreen" if node.label == 0 else "lightcoral"  # رنگ سبز برای کلاس 0، مرجانی برای کلاس 1
            else:  # اگر گره تصمیم است
                # Decision node
                feature_name = self.feature_columns[node.feature_idx]  # نام ویژگی
                label = f"Feature: {feature_name}\\nIG: {node.info_gain:.4f}\\nGini: {node.gini:.4f}\\nSamples: {node.n_samples}"  # برچسب گره تصمیم
                node_color = "lightblue"  # رنگ آبی روشن برای گره‌های تصمیم

            # Create node
            dot_node = pydot.Node(node_id, label=label, shape="box", style="filled", fillcolor=node_color)  # ایجاد گره گرافویز
            graph.add_node(dot_node)  # اضافه کردن گره به گراف

            # Add edge from parent if applicable
            if parent_id is not None:  # اگر گره والد وجود دارد
                edge = pydot.Edge(parent_id, node_id, label=edge_label)  # ایجاد یال با برچسب
                graph.add_edge(edge)  # اضافه کردن یال به گراف

            # Recursively add children
            if node.left is not None:  # اگر فرزند چپ وجود دارد
                left_id = f"{node_id}_left"  # شناسه منحصر به فرد برای فرزند چپ
                add_node_to_graph(graph, node.left, left_id, node_id, f"<= {node.split_value}")  # اضافه کردن فرزند چپ
            if node.right is not None:  # اگر فرزند راست وجود دارد
                right_id = f"{node_id}_right"  # شناسه منحصر به فرد برای فرزند راست
                add_node_to_graph(graph, node.right, right_id, node_id, f"> {node.split_value}")  # اضافه کردن فرزند راست

        # Create Graphviz graph
        graph = pydot.Dot(graph_type='digraph', rankdir='TB')  # ایجاد گراف گرافویز با جهت بالا به پایین
        add_node_to_graph(graph, self.tree, "root")  # اضافه کردن گره‌های درخت از ریشه
        
        # Save as PNG
        try:  # شروع بلوک مدیریت خطا
            graph.write_png("decision_tree.png")  # ذخیره گراف به صورت فایل PNG
        except Exception as e:  # در صورت خطا
            raise RuntimeError(f"Error saving Graphviz tree as PNG: {e}")  # پرتاب خطا با پیام

    def run(self):  # متد برای اجرای کل فرآیند
        """Execute all processing steps, train/evaluate decision tree, and visualize it."""  # توضیح مستند برای متد
        self.load_data()  # بارگذاری داده‌ها
        self.preprocess_data()  # پیش‌پردازش داده‌ها
        self.generate_synthetic_data()  # تولید داده‌های مصنوعی
        self.standardize_and_discretize()  # استانداردسازی و گسسته‌سازی
        self.split_data()  # تقسیم داده‌ها به آموزش و آزمایش
        self.perform_holdout_validation()  # انجام اعتبارسنجی هولدآوت
        self.save_datasets()  # ذخیره مجموعه‌های داده
        self.train_decision_tree()  # آموزش درخت تصمیم
        self.evaluate_tree()  # ارزیابی درخت تصمیم
        self.visualize_tree()  # نمایش متنی درخت
        self.visualize_tree_graphviz()  # نمایش گرافیکی درخت با گرافویز

if __name__ == "__main__":  # شرط اجرای مستقیم اسکریپت
    processor = NewsPopularityProcessor()  # ایجاد نمونه از کلاس پردازشگر
    processor.run()  # اجرای فرآیند اصلی


    ################ data samples shuffle doesnt affect if noticed