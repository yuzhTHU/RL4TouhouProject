import numpy as np
import torch
import os
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
        self.wparam = dict(
            left=win32con.VK_LEFT, 
            right=win32con.VK_RIGHT, 
            up=win32con.VK_UP,
            down=win32con.VK_DOWN,
            shift=win32con.VK_LSHIFT,
            esc=win32con.VK_ESCAPE,
            enter=win32con.VK_RETURN,
            z=0x5A,
            x=0x58,            
        )
        self.scancode = dict([(key, win32api.MapVirtualKey(wparam, 0)) for key, wparam in self.wparam.items()])
        self.current_reward = 0
    
    def init(self):
        """打开游戏并进入标题画面"""
        if self.hwnd == 0:
            os.system('start ./playground/东方红魔乡.lnk')
            time.sleep(2)
            self.hwnd = win32gui.FindWindow(None, self.name)
    
    def start(self):
        """从标题画面开始游戏，并返回最初状态"""
        win32gui.SetForegroundWindow(self.hwnd)
        self._press('z', wait_after=2.)
        self._press('z', wait_after=2.)
        self._press('z', wait_after=2.)
        self._press('z', wait_after=2.)
        self._press('z', wait_after=2.)
        self._press('z', wait_after=2.)
        screenshot = self._get_screenshot()
        info = self._get_info(screenshot)
        state = self._get_state(info)
        self.current_reward = 0
        return state

    def exit(self):
        """从游戏状态回到初始界面"""
        win32gui.SetForegroundWindow(self.hwnd)
        self._release_all()
        self._press('esc', wait_after=.5)
        self._press('down', wait_after=.5)
        self._press('z', wait_after=.5)
        self._press('up', wait_after=.5)
        self._press('z', wait_after=1.)

    def step(self, action):
        assert self.hwnd > 0, "请先执行 init"
        win32gui.SetForegroundWindow(self.hwnd)
        key_dict = self._get_key_dict(action)
        for key, value in key_dict.items():
            if value:
                win32api.keybd_event(self.wparam[key], self.scancode[key], 0, 0)  # press
            else:
                win32api.keybd_event(self.wparam[key], self.scancode[key], win32con.KEYEVENTF_KEYUP, 0)  # release
        screenshot = self._get_screenshot()
        info = self._get_info(screenshot)
        state = self._get_state(info)
        reward = info['reward'] + 1.0 * info['power'] + 1.0 * info['graze'] + 1.0 * info['point'] - self.current_reward
        done = (info['player'] == 0)  # 其实此时还没结束，但为了方便起见认为此时即结束
        self.current_reward = info['reward']
        return state, reward, done
    
    def _get_state(self, info):
        state = torch.cat([
            torch.FloatTensor(info['canvas']).view(-1), 
            torch.FloatTensor([info['player']]), 
            torch.FloatTensor([info['bomb']])
        ]).view(1, -1)
        return state

    def _get_key_dict(self, action):
        key_dict = dict(up=action[0, 0], down=action[0, 1], left=action[0, 2], right=action[0, 3], z=action[0, 4], x=action[0, 5], shift=action[0, 6])
        return key_dict

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
        canvas_img = screenshot.crop((50, 25, 624, 695)).resize((128, 128), Image.ANTIALIAS)
        canvas = np.array(canvas_img)
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

    def _press(self, key, wait_before=0, duration=.1, wait_after=0):
        if wait_before: 
            time.sleep(wait_before)
        win32api.keybd_event(self.wparam[key], self.scancode[key], 0, 0)  # press
        if duration: 
            time.sleep(duration)
        win32api.keybd_event(self.wparam[key], self.scancode[key], win32con.KEYEVENTF_KEYUP, 0)  # release
        if wait_after: 
            time.sleep(wait_after)

    def _release_all(self):
        for key in self.wparam:
            win32api.keybd_event(self.wparam[key], self.scancode[key], win32con.KEYEVENTF_KEYUP, 0)

if __name__ == '__main__':
    env = EoSD()
    state, reward = env.step(dict(z=True, left=True, shift=True, up=True))
    print(state, reward)
    time.sleep(2.0)
    state, reward = env.step(dict(z=False, left=False, shift=False, up=False))
    print(state, reward)
    