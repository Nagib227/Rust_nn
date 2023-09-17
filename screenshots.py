import pyautogui
import time
import os


num_screenshots = 1
num_screenshots_in_minute = 16
first_name = 2001
time_between_screenshots = 60 / num_screenshots_in_minute


time.sleep(30)

for i in range(num_screenshots):
    screenshot = pyautogui.screenshot()
    screenshot.save(f"datasets/rust/new_dataset/{first_name}.jpg") # ("1.png")
    
    first_name += 1
    time.sleep(time_between_screenshots)
