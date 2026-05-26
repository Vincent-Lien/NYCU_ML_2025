import os
import argparse
import pandas as pd

def calculate_accuracy():
    # ==================== 設定參數解析器 (Args) ====================
    parser = argparse.ArgumentParser(description="📊 模型準確度 (Accuracy) 評估工具 - 文字標籤版")
    
    # 加入必填參數或帶有預設值的參數
    parser.add_argument(
        '--pred', '-p', 
        type=str, 
        required=True, 
        help="模型預測結果 CSV 檔案的路徑 (例如: submission.csv 或 test_result.csv)"
    )
    parser.add_argument(
        '--gt', '-g', 
        type=str, 
        default="self_gt.csv", 
        help="標準答案 CSV 檔案的路徑 (預設為: self_gt.csv)"
    )
    
    args = parser.parse_args()
    
    GT_PATH = args.gt
    PRED_PATH = args.pred
    # ==============================================================

    # 1. 檢查檔案是否存在
    if not os.path.exists(GT_PATH):
        print(f"❌ 錯誤：找不到標準答案檔案 '{GT_PATH}'")
        return
    if not os.path.exists(PRED_PATH):
        print(f"❌ 錯誤：找不到預測結果檔案 '{PRED_PATH}'")
        return

    # 2. 讀取 CSV 檔案
    df_gt = pd.read_csv(GT_PATH, encoding='utf-8-sig')
    df_pred = pd.read_csv(PRED_PATH, encoding='utf-8-sig')

    # 3. 確保欄位名稱正確且不包含前後空白
    df_gt.columns = df_gt.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    if 'filename' not in df_gt.columns or 'label' not in df_gt.columns:
        print(f"❌ 錯誤：'{GT_PATH}' 的欄位必須包含 'filename' 和 'label'")
        return

    # 欄位自動相容與更名
    if 'filename' in df_pred.columns:
        pass
    elif 'img_path' in df_pred.columns:
        df_pred = df_pred.rename(columns={'img_path': 'filename'})
    else:
        print(f"❌ 錯誤：'{PRED_PATH}' 找不到 'filename' 或 'img_path' 欄位")
        return

    if 'label' in df_pred.columns:
        df_pred = df_pred.rename(columns={'label': 'pred_label'})
    elif 'predicted_label' in df_pred.columns:
        df_pred = df_pred.rename(columns={'predicted_label': 'pred_label'})
    else:
        print(f"❌ 錯誤：'{PRED_PATH}' 找不到 'label' 或 'predicted_label' 欄位")
        return

    # 4. 以 'filename' 為基準進行對齊與合併 (Inner Join)
    df_merged = pd.merge(df_gt, df_pred, on='filename', how='inner')

    total_gt = len(df_gt)
    total_pred = len(df_pred)
    matched_count = len(df_merged)

    print("==============================================")
    print(f"📊 資料對齊統計：")
    print(f"   - 標準答案 ({GT_PATH}) 總筆數: {total_gt}")
    print(f"   - 模型預測 ({PRED_PATH}) 總筆數: {total_pred}")
    print(f"   - 成功配對的圖片張數: {matched_count}")
    
    if matched_count == 0:
        print("❌ 錯誤：配對張數為 0！請檢查兩檔案的檔名格式是否完全一致。")
        return

    # 5. 確保兩邊皆轉為字串型態並清除前後空格，避免因空格導致文字比對失敗
    df_merged['label'] = df_merged['label'].astype(str).str.strip()
    df_merged['pred_label'] = df_merged['pred_label'].astype(str).str.strip()
    
    # 計算正確預測的數量
    correct_count = (df_merged['label'] == df_merged['pred_label']).sum()

    # 6. 計算最終準確率
    accuracy = (correct_count / matched_count) * 100

    print("----------------------------------------------")
    print(f"🎯 評估結果：")
    print(f"   - 預測正確張數: {correct_count} / {matched_count}")
    print(f"   - 模型準確度 (Accuracy): {accuracy:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    calculate_accuracy()
