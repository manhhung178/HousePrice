# MODULE: DATA PROCESSING PIPELINE
# ------------------------------------------------------------------------------------------
# Description: Tự động làm sạch, xử lý thiếu và mã hóa dữ liệu.
#              Đảm bảo không bị Data Leakage (Rò rỉ dữ liệu) giữa Train và Test.
# Author:      [Tên Của Bạn]
# Created:     2025-11-28

import pandas as pd
import numpy as np

# Cấu hình hiển thị Pandas / Pandas Display Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. LOAD DATA / TẢI DỮ LIỆU
print(">> [Init] Loading raw datasets...")
# Thay đổi đường dẫn phù hợp với môi trường của bạn
# Change the path according to your environment
TRAIN_PATH = r"D:\HousePrices\train.csv"
TEST_PATH  = r"D:\HousePrices\test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"   Original Train shape: {train.shape}")
print(f"   Original Test shape:  {test.shape}")

# 2. CORE PROCESSING FUNCTION / HÀM XỬ LÝ TRUNG TÂM

def process_data_pipeline(df_train, df_test):
    """
    Thực hiện quy trình xử lý dữ liệu đầu cuối (End-to-End Data Processing Pipeline).
    
    Các bước thực hiện (Steps):
    1. Log Transform biến mục tiêu (SalePrice).
    2. Làm sạch kiểu dữ liệu (Data Type Cleaning).
    3. Điền giá trị thiếu (Imputation) - Sử dụng thống kê từ tập TRAIN để tránh Leakage.
    4. Mã hóa biến phân loại (One-Hot Encoding).
    
    Args:
        df_train (pd.DataFrame): Dữ liệu huấn luyện gốc.
        df_test (pd.DataFrame): Dữ liệu kiểm tra gốc.
        
    Returns:
        tuple: (train_processed, test_processed) - Hai DataFrame đã sẵn sàng để train model.
    """

    # Copy để không ảnh hưởng dữ liệu gốc
    train_df = df_train.copy()
    test_df = df_test.copy()
    
    # --------------------------------------------------------------------------
    # STEP 1: TARGET VARIABLE TRANSFORMATION (Biến đổi biến mục tiêu)
    # --------------------------------------------------------------------------
    # Lý do: Giá nhà thường bị lệch phải (Right-skewed). Log transform giúp phân phối chuẩn hơn.
    # Reason: House prices are often right-skewed. Log transform normalizes the distribution.
    print("\n>> [Step 1] Applying Log Transformation to Target...")
    if 'SalePrice' in train_df.columns:
        train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
        target = train_df['SalePrice'] # Lưu riêng biến mục tiêu / Save target separately
        train_df = train_df.drop('SalePrice', axis=1) # Tạm bỏ để xử lý features
    else:
        raise ValueError("Critical Error: 'SalePrice' column not found in training data!")

    # Gộp Train và Test để xử lý đồng nhất các Features
    # Combine Train and Test for consistent feature engineering
    ntrain = train_df.shape[0]
    all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    
    print(f"\n>> [Step 2] Cleaning Data Types & Anomalies (Total rows: {all_data.shape[0]})...")

    # --------------------------------------------------------------------------
    # STEP 2: DATA TYPE CLEANING (Làm sạch kiểu dữ liệu)
    # --------------------------------------------------------------------------
    # Lý do: Một số cột số bị lẫn ký tự lạ (VD: 'NA', 'Null') nên Pandas hiểu nhầm là Object.
    # Reason: Some numeric columns contain mixed types, causing Pandas to treat them as Objects.
    
    for col in all_data.columns:
        # Bỏ qua nếu cột đã là số
        if pd.api.types.is_numeric_dtype(all_data[col]):
            continue
            
        # Cố gắng chuyển sang số, lỗi biến thành NaN
        # Try converting to numeric, coerce errors to NaN
        numeric_converted = pd.to_numeric(all_data[col], errors='coerce')
        
        # Nếu sau khi ép kiểu, số lượng NaN không quá nhiều (<50%) -> Đây thực sự là cột số
        # If NaN ratio < 50% after conversion -> It's likely a numeric column
        if numeric_converted.isnull().sum() < len(all_data) * 0.5:
            print(f"   -> Fixed data type for column '{col}': Object -> Float")
            all_data[col] = numeric_converted

    # STEP 3: IMPUTATION (Điền dữ liệu thiếu) - CRITICAL STEP
    # NGUYÊN TẮC VÀNG: Chỉ được tính toán thống kê (Median/Mode) trên tập TRAIN.
    # GOLDEN RULE: Compute statistics (Median/Mode) ONLY on TRAIN set to prevent Data Leakage.
    
    print("\n>> [Step 3] Handling Missing Values (Leakage-Free Imputation)...")
    
    # Tách phần dữ liệu Train ra để tính toán thống kê
    X_train_temp = all_data.iloc[:ntrain]
    
    # 3.1: Xử lý cột số (Numerical Columns) -> Điền Median
    num_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        # Tính Median chỉ trên tập Train / Calculate Median on Train only
        median_val = X_train_temp[col].median()
        
        missing_count = all_data[col].isnull().sum()
        if missing_count > 0:
            print(f"   Column '{col}': Filled {missing_count} missing with Median ({median_val})")
            all_data[col] = all_data[col].fillna(median_val)
            
    # 3.2: Xử lý cột phân loại (Categorical Columns) -> Điền Mode
    cat_cols = all_data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Tính Mode chỉ trên tập Train / Calculate Mode on Train only
        # dropna=True để đảm bảo không lấy NaN làm mode
        mode_val = X_train_temp[col].mode(dropna=True)[0]
        
        missing_count = all_data[col].isnull().sum()
        if missing_count > 0:
            # print(f"   Column '{col}': Filled {missing_count} missing with Mode ('{mode_val}')")
            all_data[col] = all_data[col].fillna(mode_val)

    # STEP 4: FEATURE ENCODING (Mã hóa One-Hot)
    # Lý do: Chuyển đổi các biến phân loại thành dạng số (0/1) để mô hình hiểu được.
    # Reason: Convert categorical variables into numeric format (0/1) for the model.
    print("\n>> [Step 4] Applying One-Hot Encoding...")
    all_data = pd.get_dummies(all_data) # drop_first=False để giữ lại thông tin đầy đủ
    print(f"   -> New shape after encoding: {all_data.shape}")
    
    # STEP 5: FINALIZE & EXPORT (Hoàn tất)
    # Tách lại Train và Test như ban đầu
    train_processed = all_data.iloc[:ntrain].copy()
    test_processed = all_data.iloc[ntrain:].copy()
    
    # Gắn lại biến mục tiêu SalePrice vào Train
    train_processed['SalePrice'] = target.values
    
    return train_processed, test_processed

# 3. EXECUTION / THỰC THI

if __name__ == "__main__":
    # Gọi hàm xử lý
    train_processed, test_processed = process_data_pipeline(train, test)

    print("\n" + "="*50)
    print("PROCESSING COMPLETED / XỬ LÝ HOÀN TẤT")
    print("="*50)
    print(f"Final Train shape: {train_processed.shape}")
    print(f"Final Test shape:  {test_processed.shape}")

    # Lưu file kết quả (Sử dụng CSV để dễ kiểm tra)
    # Save processed files
    SAVE_TRAIN_PATH = r"D:\HousePrices\train_processed.csv"
    SAVE_TEST_PATH  = r"D:\HousePrices\test_processed.csv"
    
    train_processed.to_csv(SAVE_TRAIN_PATH, index=False)
    test_processed.to_csv(SAVE_TEST_PATH, index=False)
    
    print(f"\n>> Data saved successfully to:")
    print(f"   - {SAVE_TRAIN_PATH}")
    print(f"   - {SAVE_TEST_PATH}")