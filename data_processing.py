import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Cấu hình hiển thị Pandas / Pandas Display Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = BASE_DIR / "train.csv"
DEFAULT_TEST_PATH = BASE_DIR / "test.csv"
DEFAULT_SAVE_TRAIN_PATH = BASE_DIR / "train_processed.csv"
DEFAULT_SAVE_TEST_PATH = BASE_DIR / "test_processed.csv"
CAT_FALLBACK_VALUE = "Missing"


def setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_raw_data(train_path=DEFAULT_TRAIN_PATH, test_path=DEFAULT_TEST_PATH):
    LOGGER.info(">> [Init] Loading raw datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    LOGGER.info("   Original Train shape: %s", train.shape)
    LOGGER.info("   Original Test shape:  %s", test.shape)
    return train, test


def process_data_pipeline(df_train, df_test):
    """
    Thực hiện quy trình xử lý dữ liệu đầu cuối (End-to-End Data Processing Pipeline).
    """
    train_df = df_train.copy()
    test_df = df_test.copy()

    LOGGER.info("\n>> [Step 1] Applying Log Transformation to Target...")
    if 'SalePrice' in train_df.columns:
        train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
        target = train_df['SalePrice']
        train_df = train_df.drop('SalePrice', axis=1)
    else:
        raise ValueError("Critical Error: 'SalePrice' column not found in training data!")

    ntrain = train_df.shape[0]
    all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    LOGGER.info("\n>> [Step 2] Cleaning Data Types & Anomalies (Total rows: %s)...", all_data.shape[0])

    for col in all_data.columns:
        if pd.api.types.is_numeric_dtype(all_data[col]):
            continue

        numeric_converted = pd.to_numeric(all_data[col], errors='coerce')
        if numeric_converted.isnull().sum() < len(all_data) * 0.5:
            LOGGER.info("   -> Fixed data type for column '%s': Object -> Float", col)
            all_data[col] = numeric_converted

    LOGGER.info("\n>> [Step 3] Handling Missing Values (Leakage-Free Imputation)...")
    X_train_temp = all_data.iloc[:ntrain]

    num_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        median_val = X_train_temp[col].median()
        missing_count = all_data[col].isnull().sum()
        if missing_count > 0:
            LOGGER.info("   Column '%s': Filled %s missing with Median (%s)", col, missing_count, median_val)
            all_data[col] = all_data[col].fillna(median_val)

    cat_cols = all_data.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        mode_series = X_train_temp[col].mode(dropna=True)
        mode_val = mode_series.iloc[0] if not mode_series.empty else CAT_FALLBACK_VALUE

        missing_count = all_data[col].isnull().sum()
        if missing_count > 0:
            if mode_series.empty:
                LOGGER.warning(
                    "   Column '%s': Mode is empty (all NaN in train). Filled %s missing with fallback '%s'",
                    col, missing_count, CAT_FALLBACK_VALUE
                )
            all_data[col] = all_data[col].fillna(mode_val)

    LOGGER.info("\n>> [Step 4] Applying One-Hot Encoding...")
    all_data = pd.get_dummies(all_data)
    LOGGER.info("   -> New shape after encoding: %s", all_data.shape)

    train_processed = all_data.iloc[:ntrain].copy()
    test_processed = all_data.iloc[ntrain:].copy()
    train_processed['SalePrice'] = target.values

    return train_processed, test_processed


def save_processed_data(
    train_processed,
    test_processed,
    train_output_path=DEFAULT_SAVE_TRAIN_PATH,
    test_output_path=DEFAULT_SAVE_TEST_PATH,
):
    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)
    LOGGER.info("\n>> Data saved successfully to:")
    LOGGER.info("   - %s", train_output_path)
    LOGGER.info("   - %s", test_output_path)


def main():
    setup_logging()
    try:
        train, test = load_raw_data()
        train_processed, test_processed = process_data_pipeline(train, test)

        LOGGER.info("\n%s", "=" * 50)
        LOGGER.info("PROCESSING COMPLETED / XỬ LÝ HOÀN TẤT")
        LOGGER.info("%s", "=" * 50)
        LOGGER.info("Final Train shape: %s", train_processed.shape)
        LOGGER.info("Final Test shape:  %s", test_processed.shape)

        save_processed_data(train_processed, test_processed)
    except FileNotFoundError as error:
        LOGGER.exception("Input file not found: %s", error)
        raise
    except Exception:
        LOGGER.exception("Unexpected error while running data processing pipeline.")
        raise


if __name__ == "__main__":
    main()
