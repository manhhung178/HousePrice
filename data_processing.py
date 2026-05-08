# MODULE: DATA PROCESSING PIPELINE
# ------------------------------------------------------------------------------------------
# Description: Tự động làm sạch, xử lý thiếu và mã hóa dữ liệu.
#              Đảm bảo không bị Data Leakage (Rò rỉ dữ liệu) giữa Train và Test.
# Author:      [Tên Của Bạn]
# Created:     2025-11-28

import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Cấu hình hiển thị Pandas / Pandas Display Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / 'train.csv'
TEST_PATH = BASE_DIR / 'test.csv'
SAVE_TRAIN_PATH = BASE_DIR / 'train_processed.csv'
SAVE_TEST_PATH = BASE_DIR / 'test_processed.csv'


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

    logger.info("[Step 1] Applying Log Transformation to Target...")
    if 'SalePrice' in train_df.columns:
        train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
        target = train_df['SalePrice']
        train_df = train_df.drop('SalePrice', axis=1)
    else:
        raise ValueError("Critical Error: 'SalePrice' column not found in training data!")

    ntrain = train_df.shape[0]
    all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

    logger.info("[Step 2] Cleaning Data Types & Anomalies (Total rows: %s)...", all_data.shape[0])

    for col in all_data.columns:
        if pd.api.types.is_numeric_dtype(all_data[col]):
            continue

        numeric_converted = pd.to_numeric(all_data[col], errors='coerce')

        if numeric_converted.isnull().sum() < len(all_data) * 0.5:
            logger.info("   -> Fixed data type for column '%s': Object -> Float", col)
            all_data[col] = numeric_converted

    logger.info("[Step 3] Handling Missing Values (Leakage-Free Imputation)...")

    X_train_temp = all_data.iloc[:ntrain]

    num_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        median_val = X_train_temp[col].median()

        missing_count = all_data[col].isnull().sum()
        if missing_count > 0:
            if pd.isna(median_val):
                logger.warning(
                    "   Column '%s': Median is NaN. Applying forward/backward fill for %s missing values.",
                    col,
                    missing_count,
                )
                all_data[col] = all_data[col].ffill().bfill()
                if all_data[col].isnull().sum() > 0:
                    all_data[col] = all_data[col].fillna(0)
                    logger.warning("   Column '%s': Remaining NaN filled with 0.", col)
            else:
                logger.info("   Column '%s': Filled %s missing with Median (%s)", col, missing_count, median_val)
                all_data[col] = all_data[col].fillna(median_val)

    cat_cols = all_data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        mode_series = X_train_temp[col].mode(dropna=True)
        missing_count = all_data[col].isnull().sum()
        if missing_count > 0:
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
                all_data[col] = all_data[col].fillna(mode_val)
            else:
                logger.warning(
                    "   Column '%s': Mode is empty. Applying forward/backward fill for %s missing values.",
                    col,
                    missing_count,
                )
                all_data[col] = all_data[col].ffill().bfill()
                if all_data[col].isnull().sum() > 0:
                    all_data[col] = all_data[col].fillna('Unknown')
                    logger.warning("   Column '%s': Remaining NaN filled with 'Unknown'.", col)

    logger.info("[Step 4] Applying One-Hot Encoding...")
    all_data = pd.get_dummies(all_data)
    logger.info("   -> New shape after encoding: %s", all_data.shape)

    train_processed = all_data.iloc[:ntrain].copy()
    test_processed = all_data.iloc[ntrain:].copy()

    train_processed['SalePrice'] = target.values

    return train_processed, test_processed


def main():
    try:
        logger.info("[Init] Loading raw datasets...")
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)

        logger.info("   Original Train shape: %s", train.shape)
        logger.info("   Original Test shape:  %s", test.shape)

        train_processed, test_processed = process_data_pipeline(train, test)

        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETED / XỬ LÝ HOÀN TẤT")
        logger.info("=" * 50)
        logger.info("Final Train shape: %s", train_processed.shape)
        logger.info("Final Test shape:  %s", test_processed.shape)

        train_processed.to_csv(SAVE_TRAIN_PATH, index=False)
        test_processed.to_csv(SAVE_TEST_PATH, index=False)

        logger.info("Data saved successfully to:")
        logger.info(" - %s", SAVE_TRAIN_PATH)
        logger.info(" - %s", SAVE_TEST_PATH)

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        raise
    except Exception as exc:
        logger.exception("Data processing failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
