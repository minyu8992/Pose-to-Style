from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from chainer import cuda, Variable, serializers
from PIL import Image, ImageFilter
from net import *
import torchvision.transforms as transforms
import cv2
import time
from torch2trt import TRTModule
import torch2trt
import torch
import trt_pose.models
import json
import trt_pose.coco
import os
import sys

class StyleTransfer:
    def __init__(self):
        self.model = FastStyleNet()
        self.mpath = None
        self.WIDTH = 320
        self.HEIGHT = 240
        self.PADDING = 25
        self.MEDIAN_FILTER = 1
        self.loaded = False
        self.RUN_ON_GPU = True

    def _transform(self, in_image, style):
        if style == 'normal':
            return in_image
        self.mpath = f'models/{style}.model'
        image = cv2.resize(in_image, (960, 720))
        if not self.loaded:
            serializers.load_npz(self.mpath, self.model)
            if self.RUN_ON_GPU:
                cuda.get_device(0).use()  
                self.model.to_gpu()
            print("model loaded !")

        xp = np if not self.RUN_ON_GPU else cuda.cupy
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        if self.PADDING > 0:
            image = np.pad(image, [[0, 0], [0, 0], [self.PADDING, self.PADDING], [self.PADDING, self.PADDING]], 'symmetric')
        image = xp.asarray(image)
        x = Variable(image)
        y = self.model(x)
        result = cuda.to_cpu(y.data)
        if self.PADDING > 0:
            result = result[:, :, self.PADDING:-self.PADDING, self.PADDING:-self.PADDING]
        result = np.uint8(result[0].transpose((1, 2, 0)))
        med = Image.fromarray(result)
        if self.MEDIAN_FILTER > 0:
            med = med.filter(ImageFilter.MedianFilter(self.MEDIAN_FILTER))
        return cv2.resize(np.asarray(med), (1280, 960))

if __name__ == '__main__':
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    model_st = StyleTransfer()

    def preprocess(image):
        global device
        device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.Resize((224, 224))(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def normalize(objects):
        # objects : [18, 2]
        n_objects = torch.zeros(objects.shape)
        x_min = torch.min(objects[:, 0])
        x_range = torch.max(objects[:, 0]) - x_min
        y_min = torch.min(objects[:, 1])
        y_range = torch.max(objects[:, 1]) - y_min
        for i in range(objects.shape[0]):
            n_objects[i, 0] = (objects[i ,0] - x_min) / x_range
            n_objects[i, 1] = (objects[i ,1] - y_min) / y_range 
        return n_objects

    def style(objects):
        style = ['scream', 'prismas', 'fur', 'mermaid', 'wukon', 'pop', 'sketch'] # style name list
        threshold = [0.03, 0.03, 0.02, 0.03, 0.008, 0.03, 0.01]
        loss = torch.zeros(len(style))
        for i, s in enumerate(style):
            with open(f'models/{s}.json', 'r') as f:
                r_data = json.load(f)
            reference = torch.tensor(r_data['keypoints'])
            r_mask = torch.tensor(r_data['mask'])
            loss[i] = (torch.square(objects - reference).sum(dim=1) * r_mask).sum() / (torch.sum(r_mask).item())
        min_num = torch.argmin(loss)
        if min(loss) <= threshold[min_num]:
            return style[torch.argmin(loss)]
        else:
            return 'normal'

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    #cap = cv2.VideoCapture('rtmp://140.116.56.6:1935/live')
    cap = cv2.VideoCapture(0)
    while(True):
        ret, image = cap.read()
        if ret:
            image = cv2.resize(image, (1280, 960))
            data = preprocess(image)
            cmap, paf = model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)
            # draw_objects(image, counts, objects, peaks)
            n_peaks = normalize(peaks[0, :, 0, :])
            style_name = style(n_peaks)
            image_t = model_st._transform(image, style_name)
            image = cv2.addWeighted(image, 0.3, image_t, 0.7, 0)
            image = image[:, ::-1, :]
            cv2.imshow('pose to style', image)
            cv2.waitKey(1)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
