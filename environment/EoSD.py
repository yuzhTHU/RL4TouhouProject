import numpy as np
import time
import pyautogui
import win32gui, win32api, win32con
import pytesseract
from PIL import ImageOps, Image
pytesseract.pytesseract.tesseract_cmd = r"D:\ProgramFiles\Tesseract-OCR\tesseract.exe"

class EoSD:
    """
    東方紅魔郷: the Embodiment of Scarlet Devil
    """
    name = "東方紅魔郷     the Embodiment of Scarlet Devil"
    def __init__(self):
        super(EoSD, self).__init__()
        self.hwnd = win32gui.FindWindow(None, self.name)
        assert self.hwnd > 0, f"未找到窗口 {self.name}！"
        self.wparam = dict(
            left=win32con.VK_LEFT, 
            right=win32con.VK_RIGHT, 
            up=win32con.VK_UP,
            down=win32con.VK_DOWN,
            shift=win32con.VK_LSHIFT,
            z=0x5A,
            x=0x58,            
        )
        self.scancode = dict([(key, win32api.MapVirtualKey(wparam, 0)) for key, wparam in self.wparam.items()])
        self.current_reward = 0
    
    def init(self):
        screenshot = self._get_screenshot()
        state = self._get_state(screenshot)
        return state

    def step(self, key_dict):
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(.1)
        for key, value in key_dict.items():
            if value:
                win32api.keybd_event(self.wparam[key], self.scancode[key], 0, 0)  # press
            else:
                win32api.keybd_event(self.wparam[key], self.scancode[key], win32con.KEYEVENTF_KEYUP, 0)  # release
        screenshot = self._get_screenshot()
        info = self._get_info(screenshot)
        state = (info['canvas'], [info['player'], info['bomb']])
        reward = info['reward'] + 1.0 * info['power'] + 1.0 * info['graze'] + 1.0 * info['point'] - self.current_reward
        self.current_reward = reward
        return state, reward

    def _get_screenshot(self):
        win32gui.SetForegroundWindow(self.hwnd)
        bbox_ = win32gui.GetWindowRect(self.hwnd)
        bbox = (bbox_[0] + 5, bbox_[1] + 39, 960, 720)
        img = pyautogui.screenshot(region=bbox)
        # img.save('./playground/screenshot.jpg')  # 用于调试
        return img

    def _get_info(self, screenshot):
        screenshot_array = np.array(screenshot)
        # 画面
        canvas = screenshot_array[25:695, 50:624]
        # 得分
        reward_img = ImageOps.invert(screenshot.crop((740, 120, 940, 150)).convert('L'))  # 转成白底黑字，下同
        reward_text = pytesseract.image_to_string(reward_img, lang="eng", config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        reward = int(reward_text)
        # 残机
        player_list = screenshot_array[196, 755::24]
        player_similarity = np.linalg.norm(player_list - player_list[0], axis=-1)
        player = np.argmax(player_similarity > 20)
        # 符
        bomb_list = screenshot_array[233, 755::24]
        bomb_similarity = np.linalg.norm(bomb_list - bomb_list[0], axis=-1)
        bomb = np.argmax(bomb_similarity > 20)
        # 火力
        power_img = ImageOps.invert(screenshot.crop((740, 277, 807, 307)).convert('L'))
        power_text = pytesseract.image_to_string(power_img, lang="eng", config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        power = int(power_text or 0)
        # 擦弹
        graze_img = ImageOps.invert(screenshot.crop((740, 307, 807, 337)).convert('L'))
        graze_text = pytesseract.image_to_string(graze_img, lang="eng", config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        graze = int(graze_text or 0)
        # 点
        point_img = ImageOps.invert(screenshot.crop((740, 338, 807, 368)).convert('L'))
        point_text = pytesseract.image_to_string(point_img, lang="eng", config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        point = int(point_text or 0)
        info = dict(canvas=canvas, reward=reward, player=player, bomb=bomb, power=power, graze=graze, point=point)
        return info

if __name__ == '__main__':
    env = EoSD()
    state, reward = env.step(dict(z=True, left=True, shift=True, up=True))
    print(state, reward)
    time.sleep(2.0)
    state, reward = env.step(dict(z=False, left=False, shift=False, up=False))
    print(state, reward)
    