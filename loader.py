import json
import os

from utils import symbol_trim
from airtest.aircv import imread


class ConfigLoader:
    def __init__(self):
        self.config_path = "config/"
        self.each_config = []
        self.crop_rectangles = []
        self.symbols_classes = []
        self.cv_templates = []

        self.yolo_cfg = {}
        self.resnet_cfg = []

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(self.config_path)

        with open(os.path.join(self.config_path, "專案檔.json"), 'r', encoding="utf-8") as mc:
            _config = json.load(mc)

        for output in _config["輸出順序"]:
            with open(os.path.join(self.config_path, _config["設定檔總表"][output]["設定檔路徑"]), 'r',
                      encoding="utf-8") as status:
                self.each_config.append(json.load(status))

        self.load_match_template_cfg()
        self.yolo_cfg = self.__yolo_cfg()
        self.resnet_cfg = self.__load_resnet_cfg()

    def __yolo_cfg(self) -> dict:
        for _config in self.each_config:
            if _config["設定檔名稱"] == 'yolo':
                del _config["設定檔名稱"]
                return _config

    def __load_resnet_cfg(self) -> list:
        cfg = []
        try:
            for _config in self.each_config:
                if _config["設定檔名稱"] == 'resnet':
                    for k, v in _config["symbol類型"].items():
                        cfg.append({k: [v["pt"], v["類別"], v["辨識範圍"]]})
        except KeyError:
            pass

        return cfg

    def load_match_template_cfg(self):
        self.symbols_classes = self.__load_symbol_classes()
        self.crop_rectangles = self.load_crop_range()
        self.cv_templates = self.__load_cv_template()

    def __load_symbol_classes(self) -> list:
        symbols_classes = []

        for _config in self.each_config:
            try:
                symbols_path = os.path.join(self.config_path, "".join([_config["設定檔名稱"], "/", "樣版圖"]))
                symbols_classes.append(symbol_trim.trim(symbols_path))
            except (KeyError, FileNotFoundError):
                pass

        return symbols_classes

    def load_crop_range(self, str_rect=None) -> list:
        crop_rectangles = []

        if str_rect is not None:
            for rect in str_rect:
                split_rect = rect.split(",")

                crop_rectangles.append(
                    [int(split_rect[0]), int(split_rect[1]), int(split_rect[0]) + int(split_rect[2]),
                     int(split_rect[1]) + int(split_rect[3])])

            return crop_rectangles

        for _config in self.each_config:
            try:
                if _config["設定檔名稱"] == "盤面":
                    for elem in _config["辨識範圍"]:
                        elem = elem.split(",")

                        crop_rectangles.append([int(elem[0]), int(elem[1]), int(elem[0]) + int(elem[2]),
                                                int(elem[1]) + int(elem[3])])
                else:
                    for str_crop in _config["辨識範圍"]:
                        sc_elem = str_crop.split(",")
                        crop_rectangles.append(
                            [int(sc_elem[0]), int(sc_elem[1]), int(sc_elem[0]) + int(sc_elem[2]),
                             int(sc_elem[1]) + int(sc_elem[3])])
            except KeyError:
                pass

        return crop_rectangles

    def __load_cv_template(self) -> list:
        cv_templates = []

        for _config in self.each_config:
            try:
                cv_templates += [{symbol_trim.single_trim(tmpl_path.split("\\")[1]): imread(
                    os.path.join(self.config_path, "".join([_config["設定檔名稱"], "/", tmpl_path]))),
                } for tmpl_path in _config["各樣版路徑"]]

            except KeyError:
                pass

        return cv_templates


if __name__ == "__main__":
    cl = ConfigLoader()
    print(cl.cv_templates)
