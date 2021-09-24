import json

from airtest.aircv import *


class ConfigLoader:
    def __init__(self):
        self.config_path = "config/"
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(self.config_path)

        with open(os.path.join(self.config_path, "專案檔.json"), 'r', encoding="utf-8") as f:
            self.main_config = json.load(f)


if __name__ == "__main__":
    # cl = ConfigLoader()
    # print(cl.main_config)

    a = {'設定檔總表': {'狀態': {'設定檔路徑': '狀態\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   '局數': {'設定檔路徑': '局數\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   '局數判斷': {'設定檔路徑': '局數判斷\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   '未進bouns': {'設定檔路徑': '未進bouns\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   '財產': {'設定檔路徑': '財產\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   'bet': {'設定檔路徑': 'bet\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   '贏分': {'設定檔路徑': '贏分\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}},
                   '盤面': {'設定檔路徑': '盤面\\設定檔.json', '辨識條件': {'判斷設定檔': '', '判斷邏輯': '等於', '判斷值': ''}}},
         '輸出順序': ['狀態', '局數', '局數判斷', '未進bouns', '財產', 'bet', '贏分', '盤面']}

    print(a["設定檔總表"]["bet"])

    with open(os.path.join("config", a["設定檔總表"]["盤面"]["設定檔路徑"]), 'r', encoding="utf-8") as f:
        sub = f.read()

    print(sub)
