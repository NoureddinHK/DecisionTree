import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split

# خواندن دیتاست اصلی
try:
    df = pd.read_csv('OnlineNewsPopularity.csv')
except FileNotFoundError:
    print("فایل 'OnlineNewsPopularity.csv' یافت نشد. لطفاً مطمئن شوید که فایل در پوشه درست آپلود شده است.")
    exit()

# چاپ نام ستون‌ها برای بررسی
print("ستون‌های دیتاست:")
print(df.columns)

# حذف ستون غیرعددی 'url' اگر وجود داشته باشد
if 'url' in df.columns:
    df = df.drop(columns=['url'])
else:
    print("ستون 'url' یافت نشد.")

# بررسی وجود ستون 'shares'
if 'shares' not in df.columns:
    # تلاش برای یافتن ستونی با نام مشابه (مثلاً با فاصله)
    possible_columns = [col for col in df.columns if 'shares' in col.lower()]
    if possible_columns:
        print(f"ستون 'shares' یافت نشد، اما ستون‌های مشابه پیدا شدند: {possible_columns}")
        # فرض می‌کنیم اولین ستون مشابه همان ستون هدف است
        target_column = possible_columns[0]
        print(f"استفاده از ستون '{target_column}' به عنوان ستون هدف.")
    else:
        print("هیچ ستونی مرتبط با 'shares' یافت نشد. لطفاً فایل CSV را بررسی کنید.")
        exit()
else:
    target_column = 'shares'

# انتخاب ستون هدف و ویژگی‌ها
X_original = df.drop(columns=[target_column])
y_original = df[target_column]

# استخراج نام ستون‌های عددی از دیتاست اصلی
feature_columns = X_original.columns.tolist()

# بررسی تعداد ستون‌های عددی
if len(feature_columns) < 20:
    print(f"خطا: دیتاست اصلی تنها {len(feature_columns)} ستون عددی دارد، اما 20 ستون مورد نیاز است.")
    exit()

# انتخاب 20 ستون اول برای تطبیق با دیتاست مصنوعی
feature_columns = feature_columns[:20]

# تبدیل 'shares' به یک مسئله طبقه‌بندی باینری (آستانه 1400)
y_binary = (y_original >= 1400).astype(int)

# بررسی تعداد کلاس‌ها در دیتاست اصلی
n_classes = len(np.unique(y_binary))
print(f"تعداد کلاس‌ها: {n_classes}")
print(f"توزیع کلاس‌ها در دیتاست اصلی:\n{pd.Series(y_binary).value_counts()}")

# تولید دیتاست جدید با make_classification
X_new, y_new = make_classification(
    n_samples=10000,           # تعداد نمونه‌ها
    n_features=20,             # تعداد ویژگی‌ها
    n_informative=15,          # تعداد ویژگی‌های اطلاعاتی
    n_redundant=5,             # تعداد ویژگی‌های اضافی
    n_classes=n_classes,       # تعداد کلاس‌ها
    random_state=42,           # برای تکرارپذیری
    weights=np.bincount(y_binary) / len(y_binary)  # توزیع کلاس‌ها مشابه دیتاست اصلی
)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

# گسسته‌سازی ویژگی‌ها با روش چارکی (Quantile)
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
X_discretized = discretizer.fit_transform(X_new)

# ایجاد DataFrame برای دیتاست گسسته‌شده با نام ستون‌های واقعی
df_discretized = pd.DataFrame(X_discretized, columns=feature_columns)
df_discretized['shares'] = y_new

# تقسیم دیتاست به آموزش (80%) و آزمایش (20%)
train_df, test_df = train_test_split(df_discretized, test_size=0.2, random_state=42, stratify=df_discretized['shares'])

# ذخیره دیتاست‌های آموزش و آزمایش
train_df.to_csv('Train_Discretized_SyntheticNewsPopularity.csv', index=False)
test_df.to_csv('Test_Discretized_SyntheticNewsPopularity.csv', index=False)
print("دیتاست آموزش در فایل 'Train_Discretized_SyntheticNewsPopularity.csv' ذخیره شد.")
print("دیتاست آزمایش در فایل 'Test_Discretized_SyntheticNewsPopularity.csv' ذخیره شد.")

# نمایش اطلاعات دیتاست گسسته‌شده
print("\nاطلاعات دیتاست گسسته‌شده (کل):")
print(f"تعداد نمونه‌ها: {df_discretized.shape[0]}")
print(f"تعداد ویژگی‌ها: {df_discretized.shape[1] - 1}")
print(f"توزیع کلاس‌ها:\n{pd.Series(df_discretized['shares']).value_counts()}")

# نمایش اطلاعات دیتاست آموزش
print("\nاطلاعات دیتاست آموزش (80%):")
print(f"تعداد نمونه‌ها: {train_df.shape[0]}")
print(f"توزیع کلاس‌ها:\n{pd.Series(train_df['shares']).value_counts()}")

# نمایش اطلاعات دیتاست آزمایش
print("\nاطلاعات دیتاست آزمایش (20%):")
print(f"تعداد نمونه‌ها: {test_df.shape[0]}")
print(f"توزیع کلاس‌ها:\n{pd.Series(test_df['shares']).value_counts()}")

# نمایش نقاط برش برای هر ویژگی
for i, feature in enumerate(feature_columns):
    print(f"\nویژگی {feature}:")
    print(f"نقاط برش: {discretizer.bin_edges_[i]}")
    print(f"تعداد بازه‌ها: {len(discretizer.bin_edges_[i]) - 1}")