import os
import cv2
import loader
import json
import datetime

from tqdm import tqdm
from airtest.aircv import crop_image
from PIL import Image

from utils import symbol_trim
from identify import MLResnet, CVMatchTemplate, MLYoloV4

from concurrent.futures import ThreadPoolExecutor


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

    def predict_process(self, frame, crop_rects):
        results = []

        if hasattr(self, 'ML_yolo'):
            return self.ML_yolo.inference(frame)
        else:
            for rect in crop_rects:
                crop_img = crop_image(frame, rect)

                if hasattr(self, 'ML_resnet'):
                    cvt_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cvt_img)
                    results.append(self.ML_resnet.inference(pil_img))
                else:
                    results.append(self.MT.match_template(crop_img))

            return "\t".join(results)

    def start(self, output, video_file, pbar_index: int):
        cap = cv2.VideoCapture(video_file)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        i = 1
        progress_bar = tqdm(total=total_frame, desc=output, leave=True, position=pbar_index)

        with open(os.path.join(self.config_path, self.main_config["設定檔總表"]["盤面"]["設定檔路徑"]), 'r',
                  encoding="utf-8") as sub_config:
            sc = json.load(sub_config)

        """"""
        crop_rects = []
        for str_crop in sc["辨識範圍"]:
            sc_elem = str_crop.split(",")
            crop_rects.append([int(sc_elem[0]), int(sc_elem[1]), int(sc_elem[0]) + int(sc_elem[2]),
                               int(sc_elem[1]) + int(sc_elem[3])])

        while cap.isOpened():
            progress_bar.update(i)

            ret, frame = cap.read()
            if frame is None:
                break

            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            dt = datetime.datetime.utcfromtimestamp(ms / 1000.0).strftime('%H:%M:%S.%f')[:-3]

            with ThreadPoolExecutor(max_workers=10) as executor:
                fut = executor.submit(self.predict_process, frame=frame, crop_rects=crop_rects)

            self.__output_file(output=output, data="{}\t{}\n".format(dt, fut.result()))
        """"""

        cap.release()
        print("finish!!")


"""
TODO:
2. Symbols include spin/stop/score...
3. Resnet read multiple weights
4. Specified resnet weight for inference specified symbols
"""

if __name__ == '__main__':
    vid = "video/星城_海神_4.mkv"
    fn = vid.split(".")[0]

    IW = IdentifyWorker(yolo_config=dict({"meta_path": r"./darknet/Sea_detection/cfg/symbols.data",
                                          "config_path": r"./darknet/Sea_detection/cfg/yolov4-tiny-obj.cfg",
                                          "weight": r"./darknet/Sea_detection/cfg/weights/yolov4-tiny-obj_10000.weights"}),
                        resnet50_weight="weights/star_city_symbols.pt")

    IW.start(fn, vid, 0)
