import os
import pandas as pd

# ==================== 設定區 ====================
ALL_RESULTS_PATH = "all_inference_results.csv"  # 原始完整預測結果
REVIEWED_PATH = "needs_review.csv"             # 你用 UI 檢查完的檔案
LABELS_CSV = "dataset/class_labels.csv"         # 類別與 ID 的對照表
OUTPUT_PATH = "submission.csv"                 # 最終精簡產出的繳交檔案
# ================================================

def merge_to_clean_format():
    if not os.path.exists(ALL_RESULTS_PATH) or not os.path.exists(REVIEWED_PATH):
        print("❌ 錯誤：找不到輸入檔案，請確認 all_inference_results.csv 與 needs_review.csv 是否在當前目錄下。")
        return
    if not os.path.exists(LABELS_CSV):
        print(f"❌ 錯誤：找不到對照表檔案 '{LABELS_CSV}'，無法將 ID 轉回類別名稱。")
        return

    # 讀取 CSV
    df_all = pd.read_csv(ALL_RESULTS_PATH, encoding='utf-8-sig')
    df_review = pd.read_csv(REVIEWED_PATH, encoding='utf-8-sig')
    df_labels = pd.read_csv(LABELS_CSV, encoding='utf-8-sig')

    # 建立 ID 對應到英文類別名稱的字典 (用於將人工修正的 ID 轉回英文名稱)
    id_to_name = dict(zip(df_labels['label_id'].astype(int), df_labels['class_name']))

    # 1. 篩選出真正有人工標記的資料 (排除未檢查的 -1 與空值)
    if 'manual_label_id' in df_review.columns:
        df_fixed = df_review[(df_review['manual_label_id'] != -1) & (df_review['manual_label_id'].notna())]
        # 建立 圖片名稱 -> 人工正確 ID 的對照表
        fixed_dict = dict(zip(df_fixed['img_path'], df_fixed['manual_label_id'].astype(int)))
    else:
        fixed_dict = {}

    # 2. 建立一個全新的 DataFrame 結構，用來儲存最終結果
    final_rows = []
    overwrite_count = 0

    for _, row in df_all.iterrows():
        img_name = row['img_path']
        ai_id = int(row['predicted_label_id'])
        ai_name = row['predicted_class_name']
        
        # 判斷邏輯：如果這張圖片有被人工修改，就用人工的 ID 查出對應的類別名稱；否則維持 AI 預測名稱
        if img_name in fixed_dict:
            final_id = fixed_dict[img_name]
            final_name = id_to_name.get(final_id, ai_name) # 查不到則用舊名保底
            
            if final_id != ai_id:
                overwrite_count += 1
        else:
            final_name = ai_name
            
        # 收集這兩欄：欄位名稱調整為你前面定義的格式
        final_rows.append({
            'img_path': img_name,
            'predicted_label': final_name  # ⚠️ 這裡已經成功替換為類別名稱囉！
        })

    # 3. 轉成 DataFrame 並輸出
    df_output = pd.DataFrame(final_rows)
    
    # 儲存檔案 (不保留 pandas 預設的 0,1,2 索引欄位)
    df_output.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("==============================================")
    print(f"🎉 精簡版資料整合完成（類別名稱版）！")
    print(f"✍️ 總共偵測並以人工標籤成功覆蓋了 {overwrite_count} 筆資料。")
    print(f"💾 已儲存成僅含兩欄的檔案：'{OUTPUT_PATH}'")
    print("==============================================")

if __name__ == "__main__":
    merge_to_clean_format()
