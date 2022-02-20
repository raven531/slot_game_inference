import time
import cv2
import loader
import datetime

from airtest.aircv import crop_image
from PIL import Image

from identify import MLResnet, CVMatchTemplate, MLYoloV4

from concurrent.futures import ThreadPoolExecutor


class IdentifyWorker(loader.ConfigLoader):
    def __init__(self):
        super(IdentifyWorker, self).__init__()

        self.MT = CVMatchTemplate()

        if len(self.ResNet_cfg) != 0:
            for k, v in self.ResNet_cfg[0].items():
                self.ML_resnet = MLResnet(v[1], weight_path=v[0])

        if self.YOLO_cfg is not None:
            self.ML_yolo = MLYoloV4(self.YOLO_cfg["yolo組態"], self.YOLO_cfg["symbol_meta"], self.YOLO_cfg["權重"])

    @staticmethod
    def __output_file(output: str, data: str):
        with open("{}.txt".format(output), 'a', encoding='utf-8') as out_file:
            out_file.write(data)

    """
    Identify Order: MT -> Resnet50 -> YOLO
    """

    def predict_process(self, dt, frame):
        results = []
        resnet_result = []
        yolo_result = []

        for _cfg in self.MT_cfg:
            for k, items in _cfg.items():
                for crop_rect in items[1]:
                    crop_img = crop_image(frame, crop_rect)
                    results.append(self.MT.match_template(crop_img, items[0], threshold=items[2]))

            if hasattr(self, 'ML_resnet'):
                for _, items in self.ResNet_cfg[0].items():
                    for rect in items[2]:
                        crop_img = crop_image(frame, rect)
                        cvt_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(cvt_img)
                        resnet_result.append(self.ML_resnet.inference(pil_img))

            if hasattr(self, 'ML_yolo'):
                yolo_result = self.ML_yolo.inference(frame)

            results += resnet_result
            print(results)

            return dt + "\t".join(results) + str(yolo_result)

    def start(self, video: str, output, pbar_index: int):

        # progress_bar = tqdm(total=total_frame, desc=output, leave=True, position=pbar_index)

        cap = cv2.VideoCapture(video)

        with ThreadPoolExecutor(max_workers=3) as executor:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                dt = datetime.datetime.utcfromtimestamp(ms / 1000.0).strftime('%H:%M:%S.%f')[:-3]

                fut = executor.submit(self.predict_process, dt, frame)

        """預測結果輸出成文件"""
        self.__output_file(output=output, data="{}\n".format(fut.result()))
        # progress_bar.update(1)

        print("finish!!")


if __name__ == '__main__':
    video = "video/影片.mkv"

    fn = video.split(".")[0]

    IW = IdentifyWorker()

    IW.start(video, fn, 0)
