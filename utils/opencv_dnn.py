import cv2
import numpy as np
import random

from darknet.x64 import darknet as dn

"""deprecated"""


class OpenCVDNN:
    def __init__(self, weights, cfg_path, names_path):
        self.names_path = names_path

        self.conf_threshold = 0.2
        self.nms_threshold = 0.4

        self.colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        self.net = cv2.dnn.readNet(weights, cfg_path)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

        with open(self.names_path) as f:
            self.classes_name = [cname.strip("\n") for cname in f.readlines()]

    def detection(self, img):
        classes, scores, boxes = self.model.detect(img, self.conf_threshold, self.nms_threshold)

        classes = classes.tolist()
        scores = scores.tolist()
        composed_list = [
            [b[0], b[1], b[2], b[3], self.classes_name[classes[i][0]] if scores[i][0] > 0.9 else '無法辨識'] for
            i, b in enumerate(boxes)
        ]

        # TODO: Sorted bounding box
        # return sorted(composed_list, key=lambda a: (a[0] * 10 + a[1]))

        """fetch symbols"""
        # row = 3
        # col = 5
        # sorted_list = {}
        # for r in range(1, row + 1):
        #     for c in range(1, col + 1):
        #         sorted_list["r{}c{}".format(r, c)] = composed_list.pop(c)
        #
        # return sorted_list

        sorted_list = [i[4] for i in composed_list]
        return sorted_list


class Detect:
    def __init__(self, meta_path, config_path, weight_path, gpu_id=0):
        dn.set_gpu(gpu_id)
        self.network, self.class_names, self.class_colors = dn.load_network(config_path, meta_path, weight_path,
                                                                            batch_size=1)
        self.colors = self.color()

    def color(self):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
        return colors

    def _plot_one_box(self, xyxy, img_rgb, color, label):
        img = img_rgb[..., ::-1]
        pt1 = (int(xyxy[0]), int(xyxy[1]))
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

    def predict_image(self, image, thresh=0.25, is_show=True, save_path=''):
        rgb_img = image[..., ::-1]
        height, width = rgb_img.shape[:2]
        network_width = dn.network_width(self.network)
        network_height = dn.network_height(self.network)

        resize_img = cv2.resize(rgb_img, (network_width, network_height), interpolation=cv2.INTER_LINEAR)

        darknet_img = dn.array_to_image(resize_img)
        detections = dn.detect_image(self.network, self.class_names, darknet_img, thresh=thresh)

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
                self._plot_one_box(xyxy, rgb_img, self.colors[index], label_conf)
            bgr_img = rgb_img[..., ::-1]

            if save_path:
                cv2.imwrite(save_path, bgr_img)

            return bgr_img
        # return detections
        return [i[0] for i in sorted(detections, key=lambda k: (k[2][0] + k[2][1] * 10))]
