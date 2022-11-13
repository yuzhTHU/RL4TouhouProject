import cv2
import numpy as np
import pyautogui
import PIL

image = pyautogui.screenshot(region=[0, 0, 1920, 1080])
# image.show()
image.save('test.jpg')