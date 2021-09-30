import cv2
import torch
import torch.nn as nn
import numpy as np

from torchvision import models, transforms
from darknet.x64 import darknet as dn

from utils import symbol_trim


class CVMatchTemplate:
    def __init__(self, cv_templates: list, threshold: float = 0.85):
        self.threshold = threshold
        self.cv_templates = cv_templates

    def match_template(self, image) -> str:
        collection = []
        for templates in self.cv_templates:
            for symbol, cv_tmpl in templates.items():
                try:
                    ret = cv2.matchTemplate(image, cv_tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_result, _, _ = cv2.minMaxLoc(ret)
                    if max_result >= self.threshold:
                        collection.append({"symbol": symbol_trim.single_trim(symbol), "conf": max_result})
                except Exception as e:
                    # TODO prevent template size larger than input image
                    # print(str(e))
                    pass

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
    def __init__(self, config_path, meta_path, weight_path, gpu_id=0):
        super().__init__(symbols=[])
        dn.set_gpu(gpu_id)

        self.network, \
        self.class_names, \
        self.class_colors = dn.load_network(config_path, meta_path, weight_path, batch_size=1)
        # TODO: Check .weight/.pt file exist

    def inference(self, image, threshold=0.25):
        rgb_img = image[..., ::-1]
        height, width = rgb_img.shape[:2]
        network_width = dn.network_width(self.network)
        network_height = dn.network_height(self.network)

        resize_img = cv2.resize(rgb_img, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
        darknet_img = dn.array_to_image(resize_img)

        detections = dn.detect_image(self.network, self.class_names, darknet_img, threshold)

        collection = []
        for detection in detections:
            x, y, w, h = detection[2][0], \
                         detection[2][1], \
                         detection[2][2], \
                         detection[2][3]

            # conf = detection[1]
            label = detection[0]

            x *= width / network_width
            w *= width / network_width
            y *= height / network_height
            h *= height / network_height

            xyxy = np.array([round(x - w / 2), round(y - h / 2), round(x + w / 2), round(y + h / 2)])
            collection.append({label: [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]})

        return collection
