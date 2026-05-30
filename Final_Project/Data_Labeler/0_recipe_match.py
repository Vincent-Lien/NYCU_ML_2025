import os
import shutil
import zipfile
import pandas as pd

# ==========================================
# 1. 讀取並過濾出「找不到類別」的資料
# ==========================================
df_result = pd.read_csv('evaluate/self_gt_new.csv')

# 找出 label 為空（NaN、空字串或只包含空格）的條件
is_empty = df_result['label'].isna() | (df_result['label'].astype(str).str.strip() == '')
df_empty = df_result[is_empty]

# 取得這 447 筆圖片的檔名清單
empty_filenames = df_empty['filename'].tolist()


# ==========================================
# 2. 設定路徑並開始打包
# ==========================================
image_source_dir = 'test'                # 您的圖片來源資料夾
output_zip_name = 'unlabeled_images.zip' # 預計輸出的壓縮檔檔名

print(f"📦 開始將 {len(empty_filenames)} 筆無類別圖片壓縮至 {output_zip_name}...")

missing_files_count = 0

# 建立壓縮檔
with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for filename in empty_filenames:
        # 組合圖片在 test 資料夾中的實際完整路徑
        file_path = os.path.join(image_source_dir, filename)
        
        # 檢查該圖片檔案是否存在，存在才進行壓縮防出錯
        if os.path.exists(file_path):
            # arcname=filename 代表解壓縮後，檔案不會帶有 test/ 這個前綴路徑
            zipf.write(file_path, arcname=filename)
        else:
            missing_files_count += 1

# ==========================================
# 3. 輸出打包結果與防錯提示
# ==========================================
print("✨ 打包完成！")
if missing_files_count > 0:
    print(f"⚠️ 提示：有 {missing_files_count} 筆圖片在 '{image_source_dir}' 資料夾中找不到實體檔案，已自動跳過。")
else:
    print(f"🎉 成功！447 筆圖片已完美封裝進 {output_zip_name}。")
