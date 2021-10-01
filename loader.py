import json
import os

from utils import symbol_trim
from airtest.aircv import imread


class ConfigLoader:
    def __init__(self):
        self.config_root = "config/"
        self.sub_config = []

        self.MT_cfg = []
        self.YOLO_cfg = {}
        self.ResNet_cfg = []

        if not os.path.exists(self.config_root):
            raise FileNotFoundError(self.config_root)

        with open(os.path.join(self.config_root, "專案檔.json"), 'r', encoding="utf-8") as mc:
            self.__config = json.load(mc)

        for output in self.__config["輸出順序"]:
            self.sub_config.append({output: self.__config["設定檔總表"][output]["設定檔路徑"]})

        self.MT_cfg = self.__load_match_template_cfg()
        self.ResNet_cfg = self.__load_resnet_cfg()
        self.YOLO_cfg = self.__load_yolo_cfg()

    @staticmethod
    def handle_str_crop(str_crop: str) -> tuple:
        split_rect = str_crop.split(",")

        return (int(split_rect[0]), int(split_rect[1]), int(split_rect[0]) + int(split_rect[2]),
                int(split_rect[1]) + int(split_rect[3]))

    @staticmethod
    def handle_read_image(tmpl_path) -> list:
        cv_templates = [{symbol_trim.single_trim(_img): imread(os.path.join(tmpl_path, _img))} for
                        _img in os.listdir(tmpl_path) if _img.endswith(".png")]

        return cv_templates

    def __load_match_template_cfg(self) -> list:
        collection = []

        for output in self.sub_config:
            for k, json_path in output.items():
                if k == "darknet" or k == "resnet":
                    continue

                with open(os.path.join(self.config_root, json_path), 'r', encoding="utf-8") as sub_config:
                    js_sub = json.load(sub_config)

                _crop = [self.handle_str_crop(str_rect) for str_rect in js_sub["辨識範圍"]]
                _tmpl = self.handle_read_image(os.path.join(self.config_root, "".join([k, "/", "樣版圖"])))

                collection.append({k: [_tmpl, _crop, js_sub["精準度"]]})

        return collection

    def __load_resnet_cfg(self) -> list:
        collection = []
        for _config in self.sub_config:
            for k, json_path in _config.items():
                if k == "resnet":
                    with open(os.path.join(self.config_root, json_path), 'r', encoding="utf-8") as resnet_cfg:
                        rc = json.load(resnet_cfg)

                    for symbol_type, items in rc["symbol類型"].items():
                        _crop = [self.handle_str_crop(str_rect) for str_rect in items["辨識範圍"]]
                        collection.append({symbol_type: [items["pt"], items["類別"], _crop]})

        return collection

    def __load_yolo_cfg(self) -> dict:
        for _config in self.sub_config:
            for k, json_path in _config.items():
                if k == "darknet":
                    with open(os.path.join(self.config_root, json_path), 'r', encoding="utf-8") as darknet_cfg:
                        dc = json.load(darknet_cfg)

                    dc.pop('設定檔名稱')
                    return dc


if __name__ == "__main__":
    cl = ConfigLoader()
