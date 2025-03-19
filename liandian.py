import pyautogui
import time

# 定义点击区域的坐标
click_x = 900  # 替换为你的点击区域的 x 坐标
click_y = 700  # 替换为你的点击区域的 y 坐标
interval = 5  # 时间间隔(秒)

# 定期点击指定区域
while True:
    pyautogui.click(click_x, click_y)
    print(f"Clicked at ({click_x},{click_y})")
    time.sleep(interval)
