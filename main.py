import os
import re
import cv2
import loader
import json
import threading
import datetime

from tqdm import tqdm
from airtest.aircv import show_origin_size, crop_image
from PIL import Image

from utils import symbol_trim
from identify import MLResnet, CVMatchTemplate, MLYoloV4

from concurrent.futures import ProcessPoolExecutor


class IdentifyWorker(loader.ConfigLoader):
    def __init__(self, yolo_config={}, resnet50_weight=""):
        super(IdentifyWorker, self).__init__()
        symbols_path = os.path.join(self.config_path, "".join(['盤面', '/', '樣版圖']))
        symbol_classes = symbol_trim.trim(symbols_path)

        self.MT = CVMatchTemplate(template_path=symbols_path)

        if resnet50_weight != "":
            self.ML_resnet = MLResnet(symbol_classes, weight_path=resnet50_weight)

        if len(yolo_config) == 3:
            self.ML_yolo = MLYoloV4(symbol_classes, yolo_config["config_path"], yolo_config["meta_path"],
                                    yolo_config["weight"])

    @staticmethod
    def __output_file(output: str, data: str):
        with open("{}.txt".format(output), 'a', encoding='utf-8') as out_file:
            out_file.write(data)

    """match_template/resnet/yolo"""

    def predict_process(self, frame, crop_rects) -> list:
        results = []
        for rect in crop_rects:
            crop_img = crop_image(frame, rect)
            # cvt_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # pil_img = Image.fromarray(cvt_img)
            results.append(self.MT.match_template(crop_img))

        return results

    def start(self, output, video_file, pbar_index: int):
        cap = cv2.VideoCapture(video_file)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        i = 1
        progress_bar = tqdm(total=total_frame, desc=output, leave=True, position=pbar_index)

        while cap.isOpened():
            progress_bar.update(i)

            ret, frame = cap.read()
            if frame is None:
                break

            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            dt = datetime.datetime.utcfromtimestamp(ms / 1000.0).strftime('%H:%M:%S.%f')[:-3]

            with open(os.path.join(self.config_path, self.main_config["設定檔總表"]["盤面"]["設定檔路徑"]), 'r',
                      encoding="utf-8") as sub_config:
                sc = json.load(sub_config)

            crop_rects = []
            for str_crop in sc["辨識範圍"]:
                sc_elem = str_crop.split(",")
                crop_rects.append([int(sc_elem[0]), int(sc_elem[1]), int(sc_elem[0]) + int(sc_elem[2]),
                                   int(sc_elem[1]) + int(sc_elem[3])])

            result = self.predict_process(frame, crop_rects)
            self.__output_file(output=output, data="{}\t{}\n".format(dt, result))

        cap.release()
        print("finish!!")


if __name__ == '__main__':
    # vid = "video/星城_海神_2.mkv"
    # fn = vid.split(".")[0]
    # IW = IdentifyWorker()
    # IW.start(output="{}".format(fn), video_file=vid)

    fp = "video/"
    threads = []

    IW = IdentifyWorker()
    for i, l in enumerate(os.listdir(fp)):
        vf = l.split(".")[0]
        threads.append(threading.Thread(target=IW.start("{}".format(vf), os.path.join(fp, l), pbar_index=i)))
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()
