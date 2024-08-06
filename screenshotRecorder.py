import pyautogui
import time

def capture_screenshots(interval, duration):
    total_time = 0
    screenshot_count = 1
    
    while total_time < duration:
        screenshot = pyautogui.screenshot()
        screenshot.save(f'screenshot_{screenshot_count}.png')
        print(f'Screenshot {screenshot_count} taken')
        
        time.sleep(interval)
        total_time += interval
        screenshot_count += 1



# Capture screenshots every 5 seconds for 60 seconds
time.sleep(5)
capture_screenshots(5, 60)
