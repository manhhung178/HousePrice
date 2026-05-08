# ĐỒ ÁN CUỐI KỲ: LẬP TRÌNH KHOA HỌC DỮ LIỆU
# FINAL PROJECT: DATA SCIENCE PROGRAMMING
# Đề tài (Topic): Dự đoán giá nhà (House Price Prediction)
# Mô hình (Models): Ensemble Learning (XGBoost, LightGBM, CatBoost)

import logging
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from matplotlib.widgets import Button
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import data_processing

pd.set_option('display.max_columns', None)

import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PROCESSED_PATH = BASE_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = BASE_DIR / "test_processed.csv"
SUBMISSION_PATH = BASE_DIR / "submission.csv"


def setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def ensure_processed_data_exists():
    if TRAIN_PROCESSED_PATH.exists() and TEST_PROCESSED_PATH.exists():
        return

    LOGGER.info("Processed files not found. Running data_processing.py automatically...")
    data_processing.main()

    if not TRAIN_PROCESSED_PATH.exists() or not TEST_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed files were not created: {TRAIN_PROCESSED_PATH}, {TEST_PROCESSED_PATH}"
        )


def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_perfect_fit_dashboard(y_true_orig, y_true_log, models_data, winner_name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        f"BẢNG ĐIỀU KHIỂN HIỆU NĂNG MÔ HÌNH (MODEL PERFORMANCE DASHBOARD)\nChampion Model: {winner_name}",
        fontsize=20,
        fontweight='bold',
        color='#333333',
    )

    all_preds_orig = np.concatenate([m['pred_orig'] for m in models_data])
    max_orig = max(y_true_orig.max(), all_preds_orig.max())
    min_orig = min(y_true_orig.min(), all_preds_orig.min())

    all_preds_log = np.concatenate([m['pred_log'] for m in models_data])
    max_log = max(y_true_log.max(), all_preds_log.max())
    min_log = min(y_true_log.min(), all_preds_log.min())

    for i, data in enumerate(models_data):
        name = data['name']
        is_champion = name == winner_name

        color_point = '#ff7f0e' if is_champion else '#1f77b4'
        alpha_point = 0.6 if is_champion else 0.3

        ax1 = axes[0, i]
        ax1.scatter(
            y_true_orig,
            data['pred_orig'],
            alpha=alpha_point,
            color=color_point,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )
        ax1.plot([min_orig, max_orig], [min_orig, max_orig], '--k', lw=2, label='Perfect Fit')

        rmse_title = f"RMSE: ${data['rmse_orig']:,.0f}"
        if is_champion:
            ax1.set_title(f"🏆 {name}\n{rmse_title}", fontsize=14, fontweight='bold', color='#d62728')
            ax1.set_facecolor('#fff5f0')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#d62728')
                spine.set_linewidth(2.5)
        else:
            ax1.set_title(f"{name}\n{rmse_title}", fontsize=13)

        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax1.set_xlabel("Thực tế (USD)", fontsize=11)
        if i == 0:
            ax1.set_ylabel("GÓC NHÌN KINH DOANH\nDự đoán (USD)", fontsize=12, fontweight='bold', color='darkblue')

        ax2 = axes[1, i]
        ax2.scatter(
            y_true_log,
            data['pred_log'],
            alpha=alpha_point,
            color=color_point,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )
        ax2.plot([min_log, max_log], [min_log, max_log], '--k', lw=2)

        rmse_title_log = f"RMSE: {data['rmse_log']:.4f}"
        if is_champion:
            ax2.set_title(f"🏆 {name} (Log)\n{rmse_title_log}", fontsize=14, fontweight='bold', color='#d62728')
            ax2.set_facecolor('#fff5f0')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#d62728')
                spine.set_linewidth(2.5)
        else:
            ax2.set_title(f"{name} (Log)\n{rmse_title_log}", fontsize=13)

        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_xlabel("Thực tế (Log Scale)", fontsize=11)
        if i == 0:
            ax2.set_ylabel(
                "GÓC NHÌN KỸ THUẬT\nDự đoán (Log Scale)",
                fontsize=12,
                fontweight='bold',
                color='darkgreen',
            )

    return fig


def plot_histogram(y_orig, y_log):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(y_orig, kde=True, ax=axes[0], color='#1f77b4')
    axes[0].set_title("Phân phối giá GỐC (Original USD)", fontsize=12, fontweight='bold')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    sns.histplot(y_log, kde=True, ax=axes[1], color='#2ca02c')
    axes[1].set_title("Phân phối giá LOG (Log-Transformed)", fontsize=12, fontweight='bold')

    fig.suptitle("Kiểm tra phân phối dữ liệu (Data Distribution Check)", fontsize=14)
    plt.tight_layout()
    return fig


def main():
    setup_logging()
    try:
        ensure_processed_data_exists()

        print(">> [1/8] Đang tải dữ liệu (Loading data)...")
        train = pd.read_csv(TRAIN_PROCESSED_PATH)
        test = pd.read_csv(TEST_PROCESSED_PATH)

        sample_mean = train['SalePrice'].mean()

        if sample_mean > 100:
            print(f"   -> Phát hiện giá GỐC (Mean=${sample_mean:,.0f}). Đang chuyển đổi sang Log...")
            y_orig = train['SalePrice']
            y_log = np.log1p(train['SalePrice'])
        else:
            print(f"   -> Phát hiện giá LOG (Mean={sample_mean:.2f}). Đang chuyển đổi sang Thực...")
            y_log = train['SalePrice']
            y_orig = np.expm1(y_log)

        train['SalePrice_Log'] = y_log
        train['SalePrice_Real'] = y_orig

        mask = (
            (train["GrLivArea"] < 4500)
            & (train["SalePrice_Real"] > 30000)
            & (train["SalePrice_Real"] < 700000)
        )
        train = train[mask]

        y_log_clean = train['SalePrice_Log']
        y_orig_clean = train['SalePrice_Real']
        print(f"   -> Số lượng mẫu sạch để huấn luyện: {train.shape[0]}")

        print(">> [2/8] Chia tập dữ liệu (Splitting data)...")
        X = train.drop(['SalePrice', 'SalePrice_Log', 'SalePrice_Real'], axis=1)

        X_train, X_valid, y_train_log, y_valid_log = train_test_split(
            X, y_log_clean, test_size=0.2, random_state=42
        )
        y_valid_orig = y_orig_clean.loc[y_valid_log.index]

        print(">> [3/8] Đang huấn luyện 3 mô hình mạnh nhất (Training top 3 models)...")
        execution_times = {}

        print("   -> Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        start_time = time.time()
        xgb.fit(X_train, y_train_log)
        xgb_time = time.time() - start_time
        execution_times['XGBoost'] = xgb_time
        print(f"      [Done] XGBoost Time: {xgb_time:.2f} seconds")
        pred_xgb_log = xgb.predict(X_valid)
        pred_xgb_orig = np.expm1(pred_xgb_log)

        print("   -> Training LightGBM...")
        lgb = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=32,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        start_time = time.time()
        lgb.fit(X_train, y_train_log)
        lgb_time = time.time() - start_time
        execution_times['LightGBM'] = lgb_time
        print(f"      [Done] LightGBM Time: {lgb_time:.2f} seconds")
        pred_lgb_log = lgb.predict(X_valid)
        pred_lgb_orig = np.expm1(pred_lgb_log)

        print("   -> Training CatBoost...")
        cat = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=42,
            allow_writing_files=False,
        )
        start_time = time.time()
        cat.fit(X_train, y_train_log)
        cat_time = time.time() - start_time
        execution_times['CatBoost'] = cat_time
        print(f"      [Done] CatBoost Time: {cat_time:.2f} seconds")
        pred_cat_log = cat.predict(X_valid)
        pred_cat_orig = np.expm1(pred_cat_log)

        print(">> [4/8] Đánh giá kết quả (Evaluating)...")
        rmse_xgb_real = get_rmse(y_valid_orig, pred_xgb_orig)
        rmse_xgb_log = get_rmse(y_valid_log, pred_xgb_log)
        rmse_lgb_real = get_rmse(y_valid_orig, pred_lgb_orig)
        rmse_lgb_log = get_rmse(y_valid_log, pred_lgb_log)
        rmse_cat_real = get_rmse(y_valid_orig, pred_cat_orig)
        rmse_cat_log = get_rmse(y_valid_log, pred_cat_log)

        models_data_full = [
            {
                'name': 'XGBoost',
                'model': xgb,
                'pred_orig': pred_xgb_orig,
                'rmse_orig': rmse_xgb_real,
                'pred_log': pred_xgb_log,
                'rmse_log': rmse_xgb_log,
                'time': execution_times['XGBoost'],
            },
            {
                'name': 'LightGBM',
                'model': lgb,
                'pred_orig': pred_lgb_orig,
                'rmse_orig': rmse_lgb_real,
                'pred_log': pred_lgb_log,
                'rmse_log': rmse_lgb_log,
                'time': execution_times['LightGBM'],
            },
            {
                'name': 'CatBoost',
                'model': cat,
                'pred_orig': pred_cat_orig,
                'rmse_orig': rmse_cat_real,
                'pred_log': pred_cat_log,
                'rmse_log': rmse_cat_log,
                'time': execution_times['CatBoost'],
            },
        ]

        best_result = min(models_data_full, key=lambda x: x['rmse_orig'])
        best_name = best_result['name']
        best_model = best_result['model']

        print(f"\nCHAMPION MODEL: {best_name} | RMSE: ${best_result['rmse_orig']:,.0f}")
        print("Thời gian huấn luyện từng mô hình:")
        for model_data in models_data_full:
            print(f" - {model_data['name']}: {model_data['time']:.2f} giây")

        print(">> [5/8] Đang tạo biểu đồ (Plotting)...")
        fig1 = plot_histogram(y_valid_orig, y_valid_log)
        button_ax = fig1.add_axes([0.4, 0.01, 0.2, 0.06])
        button = Button(button_ax, 'Show Final Dashboard →', color='#ffd700', hovercolor='#ffeb3b')

        def on_click(event):
            plt.close(fig1)
            fig2 = plot_perfect_fit_dashboard(y_valid_orig, y_valid_log, models_data_full, best_name)
            plt.show()

        button.on_clicked(on_click)
        plt.show()

        print(">> [6/8] Đang dự đoán tập Test (Predicting Test set)...")
        X_test_submit = test.drop('SalePrice', axis=1) if 'SalePrice' in test.columns else test
        final_pred_log = best_model.predict(X_test_submit)
        final_pred_real = np.expm1(final_pred_log)

        submission = pd.DataFrame({
            "Id": range(1461, 1461 + len(final_pred_real)),
            "SalePrice": final_pred_real,
        })

        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f" File kết quả đã được lưu tại: {SUBMISSION_PATH}")
        print("   Chương trình kết thúc thành công!")
    except FileNotFoundError as error:
        LOGGER.exception("Missing file while running training pipeline: %s", error)
        raise
    except Exception:
        LOGGER.exception("Unexpected error while running HousePrice pipeline.")
        raise


if __name__ == "__main__":
    main()
