import cv2
import loader
import datetime

from tqdm import tqdm
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

        if len(self.YOLO_cfg) == 3:
            self.ML_yolo = MLYoloV4(self.YOLO_cfg["yolo組態"], self.YOLO_cfg["symbol_meta"], self.YOLO_cfg["權重"])

    @staticmethod
    def __output_file(output: str, data: str):
        with open("{}.txt".format(output), 'a', encoding='utf-8') as out_file:
            out_file.write(data)

    """
    Identify Order: MT -> Resnet50 -> YOLO
    """

    def predict_process(self, frame):
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

        return "\t".join(results) + str(yolo_result)

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

            with ThreadPoolExecutor(max_workers=10) as executor:
                fut = executor.submit(self.predict_process, frame=frame)

            self.__output_file(output=output, data="{}\t{}\n".format(dt, fut.result()))
        """"""

        cap.release()
        print("finish!!")


"""
TODO:
~~ 1. match template separate each part~~
~~ 2. match template with precise from config ~~
{
    3. Resnet read multiple weights
    4. Specified resnet weight for inference specified symbols, dynamic remove current weight
}
{
    ~~7. output string with `\t`~~
    8. output translate Eng->Ch
}
"""

if __name__ == '__main__':
    video = "video/星城_海神_4.mkv"
    fn = video.split(".")[0]

    IW = IdentifyWorker()

    IW.start(fn, video, 0)
