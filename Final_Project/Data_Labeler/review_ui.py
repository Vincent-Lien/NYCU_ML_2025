import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from PIL import Image, ImageTk

# ==================== 解決高解析度(DPI)螢幕字體模糊問題 ====================
try:
    import ctypes
    # 使用 2 啟用每台螢幕獨立 DPI 縮放，若失敗則使用 1 啟用系統級抗鋸齒
    ctypes.windll.shcore.SetProcessDpiCapability(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ==================== 設定區 ====================
CSV_PATH = "needs_review.csv"             # 低信心度的 CSV 檔案路徑
IMAGE_FOLDER = "../test"                  # 盲測圖片所在的資料夾
LABELS_CSV = "dataset/class_labels.csv"    # 類別對照表
# ================================================

translate = {
    "beef_noodles": "牛肉maien",
    "braised_pork_over_rice": "滷肉飯",
    "bubble_tea": "珍珠nai茶",
    "coffin_toast": "棺材板",
    "deep-fried_chicken_cutlets": "炸G排",
    "egg_pancake_roll": "蛋餅",
    "grilled_taiwanese_sausage": "kao香腸",
    "iron_egg": "鐵蛋",
    "mango_shaved_ice": "芒果雪花冰",
    "mochi": "麻shu",
    "oyster_omelet": "e仔煎",
    "papaya_milk": "木瓜牛nai",
    "pineapple_cake": "鳳梨酥",
    "potsticker": "鍋貼",
    "rice_dumpling": "肉粽",
    "scallion_pancake": "cong油餅",
    "stinky_tofu": "臭豆腐",
    "sweet_potato_ball": "地瓜球",
    "tanghulu": "糖葫蘆",
    "taro_ball": "芋圓",
    "three-cup_chicken": "三杯G",
    "turkey_rice": "火G肉飯",
    "turnip_cake": "蘿蔔gao",
    "xiaolongbao": "小籠包",
    "yolk_pastry": "蛋huang酥"
}

inv_translate = {v: k for k, v in translate.items()}

class ImageReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 台灣美食圖片人工稽核系統 (高畫質放大版)")
        self.root.geometry("1024x800") # 視窗加大
        
        # 設定全域字體樣式
        self.font_title = ("Microsoft JhengHei", 16, "bold")
        self.font_body = ("Microsoft JhengHei", 14)
        self.font_btn = ("Microsoft JhengHei", 14, "bold")
        
        # 調整 ttk 下拉選單的字體大小
        style = ttk.Style()
        style.theme_use('clam') # 使用 clam 風格在 Windows 下對高 DPI 支援較好
        style.configure("TCombobox", font=self.font_body)
        self.root.option_add("*TCombobox*Listbox.font", self.font_body)
        self.root.option_add("*Font", self.font_body) # 強制全域元件字型渲染

        # 1. 讀取資料 (指定 utf-8-sig 防止中文亂碼)
        if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
            messagebox.showinfo("提示", "找不到 needs_review.csv 或檔案為空，無需檢查！")
            self.root.destroy()
            return
            
        self.df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        
        if 'manual_label_id' not in self.df.columns:
            self.df['manual_label_id'] = -1
            self.df['manual_class_name'] = "未檢查"
            
        labels_df = pd.read_csv(LABELS_CSV, encoding='utf-8-sig')
        self.name_to_id = dict(zip(labels_df['class_name'], labels_df['label_id']))
        self.class_list_zh = sorted([translate.get(name, name) for name in labels_df['class_name'].tolist()])
        
        self.pending_indices = self.df[self.df['manual_label_id'] == -1].index.tolist()
        
        if not self.pending_indices:
            messagebox.showinfo("完成", "所有圖片都已經人工檢查完畢！")
            self.root.destroy()
            return
            
        self.current_pointer = 0
        
        # 2. 建立 UI 元件
        self.setup_ui()
        self.load_current_item()

    def setup_ui(self):
        # 上方資訊區
        self.info_frame = tk.Frame(self.root, pady=15)
        self.info_frame.pack()
        
        self.lbl_progress = tk.Label(self.info_frame, text="", font=self.font_title, fg="#0056b3")
        self.lbl_progress.pack(pady=2)
        
        self.lbl_filename = tk.Label(self.info_frame, text="", font=self.font_body, fg="#333333")
        self.lbl_filename.pack(pady=2)
        
        self.lbl_ai_pred = tk.Label(self.info_frame, text="", font=self.font_title, fg="#d9534f")
        self.lbl_ai_pred.pack(pady=2)

        # 中間圖片顯示區 (加大成 450x450)
        self.img_label = tk.Label(self.root, text="圖片載入中...", bg="#e9ecef", width=450, height=450)
        self.img_label.pack(pady=15)

        # 下方控制與下拉選單區
        self.control_frame = tk.Frame(self.root, pady=15)
        self.control_frame.pack()
        
        tk.Label(self.control_frame, text="請選擇正確類別：", font=self.font_body).grid(row=0, column=0, padx=10)
        
        self.combo_classes = ttk.Combobox(self.control_frame, values=self.class_list_zh, width=22, state="readonly")
        self.combo_classes.grid(row=0, column=1, padx=10)
        
        # 放大按鈕尺寸與字體
        self.btn_submit = tk.Button(self.control_frame, text="確認修改並下一張 (Enter)", bg="#28a745", fg="white", 
                                    font=self.font_btn, command=self.save_and_next, padx=15, pady=5)
        self.btn_submit.grid(row=0, column=2, padx=15)
        
        self.btn_skip = tk.Button(self.control_frame, text="暫時跳過", bg="#ffc107", fg="black", 
                                  font=self.font_btn, command=self.skip_to_next, padx=15, pady=5)
        self.btn_skip.grid(row=0, column=3, padx=10)

        self.root.bind('<Return>', lambda event: self.save_and_next())

    def load_current_item(self):
        if self.current_pointer >= len(self.pending_indices):
            messagebox.showinfo("完成", "剩餘的低信心度圖片已全部檢查完畢！")
            self.root.destroy()
            return
            
        self.real_idx = self.pending_indices[self.current_pointer]
        row = self.df.loc[self.real_idx]
        
        ai_pred_en = row['predicted_class_name']
        ai_pred_zh = translate.get(ai_pred_en, ai_pred_en)
        
        self.lbl_progress.config(text=f"審查進度：{self.current_pointer + 1} / {len(self.pending_indices)}")
        self.lbl_filename.config(text=f"檔案名稱：{row['img_path']}")
        self.lbl_ai_pred.config(text=f"AI 預測結果：{ai_pred_zh} (信心度: {row['confidence']*100:.1f}%)")
        
        if ai_pred_zh in self.class_list_zh:
            self.combo_classes.set(ai_pred_zh)
            
        img_path = os.path.join(IMAGE_FOLDER, row['img_path'])
        try:
            img = Image.open(img_path)
            img.thumbnail((450, 450)) # 圖片顯示同步放大
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo, text="")
        except Exception as e:
            self.img_label.config(image="", text=f"❌ 無法讀取圖片\n{row['img_path']}", font=self.font_body)

    def save_and_next(self):
        selected_zh = self.combo_classes.get()
        if not selected_zh:
            messagebox.showwarning("警告", "請選擇一個類別！")
            return
            
        selected_en = inv_translate.get(selected_zh, selected_zh)
        selected_id = self.name_to_id[selected_en]
        
        self.df.at[self.real_idx, 'manual_label_id'] = int(selected_id)
        self.df.at[self.real_idx, 'manual_class_name'] = selected_en
        
        # ⚠️ 關鍵儲存：加入 utf-8-sig 編碼，徹底根除 \u1234 亂碼
        self.df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        
        self.current_pointer += 1
        self.load_current_item()

    def skip_to_next(self):
        self.current_pointer += 1
        self.load_current_item()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewApp(root)
    root.mainloop()
