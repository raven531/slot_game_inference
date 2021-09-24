import os
import cv2
import torch
import torch.nn as nn
import random
import numpy as np

from airtest.aircv import imread

from torchvision import models, transforms
from darknet.x64 import darknet as dn

from utils import symbol_trim


class CVMatchTemplate:
    def __init__(self, template_path, threshold: float = 0.8):
        self.threshold = threshold

        self.cv_templates = [{tmpl.split("_")[0]: imread(os.path.join(template_path, tmpl))} for tmpl in
                             os.listdir(template_path)]

    def match_template(self, image) -> str:
        collection = []

        for templates in self.cv_templates:
            for symbol, cv_tmpl in templates.items():
                ret = cv2.matchTemplate(image, cv_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_result, _, _ = cv2.minMaxLoc(ret)

                if max_result >= self.threshold:
                    collection.append({"symbol": symbol_trim.single_trim(symbol), "conf": max_result})

        collection = sorted(collection, key=lambda x: float(x["conf"]))

        try:
            if float(collection[0]["conf"]) >= self.threshold:
                return collection[0]["symbol"]
            else:
                return "無法辨識"
        except IndexError:
            return "無法辨識"


class MLInterface:
    def __init__(self, symbols: list):
        self.symbols = symbols

    def inference(self, image):
        pass


class MLResnet(MLInterface):
    def __init__(self, symbols: list, weight_path):
        super().__init__(symbols=symbols)

        # TODO: Check .weight/.pt file exist

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = models.resnet50(pretrained=False).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(num_features, len(self.symbols)).to(self.device)
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

    def inference(self, image) -> str:
        tensor_img = self.transform(image).cuda().unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            outputs = torch.softmax(outputs, dim=-1).cuda()

            predictions = outputs.to('cpu').numpy()[0]
            get_max = max(predictions)
            index = list(predictions).index(get_max)

            return self.symbols[index]


class MLYoloV4(MLInterface):
    def __init__(self, symbols, config_path, meta_path, weight_path, gpu_id=0, ):
        super().__init__(symbols=symbols)
        dn.set_gpu(gpu_id)

        self.network, \
        self.class_names, \
        self.class_colors = dn.load_network(config_path, meta_path, weight_path, batch_size=1)
        self.colors = self.__color()
        # TODO: Check .weight/.pt file exist

    def __color(self) -> list:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.symbols))]
        return colors

    @staticmethod
    def __plot_one_box(xyxy, img_rgb, color, label):
        img = img_rgb[..., ::-1]
        pt1 = (int(xyxy[0]), int(xyxy[0]))
        pt2 = (int(xyxy[2]), int(xyxy[3]))

        thickness = round(0.001 * max(img.shape[0:2])) + 1
        cv2.rectangle(img, pt1, pt2, color, thickness)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=thickness / 3, thickness=thickness)[0]

        c1 = (pt1[0], pt1[1] - int(t_size[1] * 1.5)) if pt1[1] - int(t_size[1] * 1.5) >= 0 else (pt1[0], pt1[1])
        c2 = (pt1[0] + t_size[0], pt1[1]) if pt1[1] - int(t_size[1] * 1.5) >= 0 else (
            pt1[0] + t_size[0], pt1[1] + int(t_size[1] * 1.5))

        if c1[0] < 0 or c1[1] < 0:
            x_t = c1[0] if c1[0] >= 0 else 0
            y_t = c1[1] if c1[1] >= 0 else 0
            c1 = (x_t, y_t)

        cv2.rectangle(img, c1, c2, color, -1)
        text_pos = (c1[0], c1[1] + t_size[1])
        cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, [255, 255, 255], thickness=thickness,
                    lineType=cv2.LINE_AA)

    def inference(self, image, threshold=0.25, is_show=True, save_path=''):
        rgb_img = image[..., ::-1]
        height, width = rgb_img[:2]
        network_width = dn.network_width(self.network)
        network_height = dn.network_height(self.network)

        resize_img = cv2.resize(rgb_img, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
        darknet_img = dn.array_to_image(resize_img)

        detections = dn.detect_image(self.network, self.class_names, darknet_img, threshold)

        if is_show:
            for detection in detections:
                x, y, w, h = detection[2][0], \
                             detection[2][1], \
                             detection[2][2], \
                             detection[2][3]

                conf = detection[1]
                label = detection[0]

                x *= width / network_width
                w *= width / network_width
                y *= height / network_height
                h *= height / network_height

                xyxy = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                index = self.class_names.index(label)
                label_conf = '{} {}'.format(label, conf)
                self.__plot_one_box(xyxy, rgb_img, self.colors[index], label_conf)
            bgr_img = rgb_img[..., ::-1]

            if save_path:
                cv2.imwrite(save_path, bgr_img)
            return bgr_img

        return [i[0] for i in sorted(detections, key=lambda k: (k[2][0], k[2][1] * 10))]
