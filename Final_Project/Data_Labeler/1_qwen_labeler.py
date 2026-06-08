import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import ollama

# ==========================================
# 1. 設定檔案路徑與開源模型名稱
# ==========================================
desc_csv_path = '../taiwanese_food_descriptions.csv'     # 原始食物敘述檔（用來算交集候選名單）
recipe_csv_path = '../evaluate/self_gt_new.csv'           # 您的原始 CSV 檔案
target_folder = 'unlabeled_images'                     # 📝 您指定只處理這個資料夾裡的圖片
output_csv_path = '../evaluate/ai_labeled_output.csv'     # 最終輸出的全新 CSV 檔案

MODEL_NAME = 'qwen2.5vl:7b' # 適合您 12GB 顯卡的 7B 視覺模型


# ==========================================
# 2. 建立「食材 -> 所有可能菜色」的對照字典（交集核心）
# ==========================================
df_desc = pd.read_csv(desc_csv_path)
ingredient_to_all_classes = {}

for _, row in df_desc.iterrows():
    current_class = row['class_name']
    ingredients_list = [i.strip().lower() for i in str(row['ingredients']).split(',')]
    for ingredient in ingredients_list:
        if ingredient:
            if ingredient not in ingredient_to_all_classes:
                ingredient_to_all_classes[ingredient] = set()
            ingredient_to_all_classes[ingredient].add(current_class)

# 全局所有合法的類別清單
all_valid_labels = df_desc['class_name'].unique().tolist()


# ==========================================
# 3. 讀取原始資料，並複製一份結構準備輸出
# ==========================================
if not os.path.exists(recipe_csv_path):
    raise FileNotFoundError(f"找不到您的 CSV 檔案: {recipe_csv_path}")

df_test = pd.read_csv(recipe_csv_path)

print(f"📋 載入原始資料，共 {len(df_test)} 筆。")
print(f"🔍 檢查目標資料夾 '{target_folder}' ...")


# ==========================================
# 4. 逐筆判斷：是否在指定資料夾內？
# ==========================================
patched_count = 0
none_count = 0

# 使用 tqdm 顯示處理進度
for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
    filename = row['filename']
    recipe_str = str(row['label'])
    
    # 組合在 target_folder 裡面的預期路徑
    folder_image_path = os.path.join(target_folder, filename)
    
    # 核心條件：如果圖片「不存在」於 unlabeled_images 資料夾中
    if not os.path.exists(folder_image_path):
        # 🟢 什麼都不做，直接保留 self_gt_new.csv 原本的資料填到新檔案
        continue
        
    # 🔴 如果圖片「存在」於資料夾中，啟動交集比對與開源 AI 辨識
    
    # --- 步驟 A: 動態計算這筆資料的食材交集 ---
    test_ingredients = [i.strip().lower() for i in recipe_str.split(',')]
    valid_sets = []
    for ing in test_ingredients:
        if ing in ingredient_to_all_classes:
            valid_sets.append(ingredient_to_all_classes[ing])
            
    candidate_labels = []
    if valid_sets:
        intersection_result = valid_sets
        for s in valid_sets[1:]:
            intersection_result = intersection_result.intersection(s)
        candidate_labels = list(intersection_result)
    
    # --- 步驟 B: 設定動態 Prompt 選項 ---
    if len(candidate_labels) > 0:
        current_candidates_str = ", ".join(candidate_labels)
        allowed_options = candidate_labels
        focus_msg = "請注意：根據食材分析，這張圖片【只有可能】是以下候選類別之一。"
    else:
        current_candidates_str = ", ".join(all_valid_labels)
        allowed_options = all_valid_labels
        focus_msg = "提示：食材交集無解，請完全依賴您的視覺常識從以下全局清單選擇。"

    prompt = f"""
你是一位精通台灣美食與小吃的專家。
請根據提供的這張菜餚圖片，以及它的部分食材清單，從下方的【候選類別】中選擇一個最符合、最精準的菜色名稱。

【部分食材清單】: {recipe_str}
【候選類別】: [{current_candidates_str}]

{focus_msg}

限制規則：
1. 請直接輸出候選類別中的其中一個英文名稱（例如: beef_noodles），不要有任何額外的解釋、句號、冒號或引號。
2. 如果你看了圖片與食材後，認為【完全不屬於】候選類別中的任何一項，或者你真的不知道答案，請直接輸出 None。
"""

    # --- 步驟 C: 呼叫本地 Qwen2.5-VL ---
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [folder_image_path] # 讀取該資料夾內的圖片
            }],
            options={
                'temperature': 0.1,
                'num_predict': 32,
                'num_thread': 8
            }
        )
        
        ai_response = response['message']['content'].strip().replace("'", "").replace('"', "")
        
        # 嚴格驗證
        if ai_response in allowed_options:
            df_test.at[index, 'label'] = ai_response
            patched_count += 1
        else:
            # 包含吐出 'None' 或不合規定的胡言亂語，皆安全標記為 None
            df_test.at[index, 'label'] = 'None'
            none_count += 1
            
    except Exception as e:
        print(f"\n⚠️ 處理 {filename} 時發生錯誤: {e}")
        df_test.at[index, 'label'] = 'None'
        none_count += 1
        continue

# ==========================================
# 5. 儲存結果與列印最終報告
# ==========================================
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df_test.to_csv(output_csv_path, index=False)

print(f"\n✨ 資料夾基準標記完成！新檔案已儲存至：{output_csv_path}")
print(f"📊 執行統計報告：")
print(f"  - 🟢 複製原始資料（未調整）的筆數: {len(df_test) - (patched_count + none_count)} 筆")
print(f"  - 🤖 資料夾內成功辨識並補齊的筆數: {patched_count} 筆")
# 修正邏輯：明確計算標記為 'None' 的筆數
print(f"  - ❌ 最終因無法判定被標記為 'None' 的筆數: {none_count} 筆")
