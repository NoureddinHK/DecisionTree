import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

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

# ایجاد DataFrame برای دیتاست جدید
columns = [f'feature_{i}' for i in range(20)]
df_new = pd.DataFrame(X_new, columns=columns)
df_new['shares'] = y_new

# ذخیره دیتاست جدید
df_new.to_csv('SyntheticNewsPopularity.csv', index=False)
print("دیتاست جدید با 10000 نمونه و 20 ویژگی در فایل 'SyntheticNewsPopularity.csv' ذخیره شد.")

# نمایش اطلاعات دیتاست جدید
print("\nاطلاعات دیتاست جدید:")
print(f"تعداد نمونه‌ها: {df_new.shape[0]}")
print(f"تعداد ویژگی‌ها: {df_new.shape[1] - 1}")
print(f"توزیع کلاس‌ها:\n{pd.Series(y_new).value_counts()}")