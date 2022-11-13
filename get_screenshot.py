import numpy as np
import pyautogui
import win32gui

scale = 1.0

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

pairs = [(hwnd, title) for hwnd, title in winlist if '语雀' in title.lower()]
hwnd = pairs[0][0]

win32gui.SetForegroundWindow(hwnd)
bbox_ = win32gui.GetWindowRect(hwnd)
bbox = list((np.array(bbox_) * scale).astype(np.int32))
img = pyautogui.screenshot(region=bbox)
# img.show()

pyautogui.keyDown('ctrl')
pyautogui.press('a')
pyautogui.keyUp('ctl')