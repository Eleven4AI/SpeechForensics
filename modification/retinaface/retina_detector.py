import torch
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_retina import RetinaFace
from .bbox import nms,decode,decode_landm,py_cpu_nms
from .detect import detect, batch_detect
from .config import cfg_re50
from .net import PriorBox
from collections import OrderedDict
import cv2
import numpy as np

models_urls = {
    'retinaface': 'https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view?usp=drive_link',
}


class RetinaFaceDetector(FaceDetector):
    '''SF3D Detector.
    '''

    def __init__(self, device, path_to_detector=None, verbose=False, filter_threshold=0.5):
        super(RetinaFaceDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls['retinaface'])
        else:
            model_weights = torch.load(path_to_detector,map_location='cpu')
        new_weight_dict=OrderedDict()
        for key,value in model_weights.items():
            new_weight_dict[key.replace('module.','')]=value
        model_weights=new_weight_dict

        self.fiter_threshold = filter_threshold
        self.face_detector = RetinaFace(cfg=cfg_re50,phase='test')
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()
        self.resize=1

    def _filter_bboxes(self, bboxlist):
        if len(bboxlist) > 0:
            keep = nms(bboxlist, 0.3)
            bboxlist = bboxlist[keep, :]
            bboxlist = [x for x in bboxlist if x[-1] > self.fiter_threshold]

        return bboxlist

    def detect_from_image(self, img):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img=np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104,117,123)
        img = img.transpose(2, 0, 1)


        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.face_detector(img)  # forward pass

        priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        #landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        #landms = landms * scale1 / self.resize
        #landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        #landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        #landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        #landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:750, :]
        #landms = landms[:750, :]


        return dets

    def detect_from_batch(self, tensor):
        bboxlists = batch_detect(self.face_detector, tensor, device=self.device)

        new_bboxlists = []
        for i in range(bboxlists.shape[0]):
            bboxlist = bboxlists[i]
            bboxlist = self._filter_bboxes(bboxlist)
            new_bboxlists.append(bboxlist)

        return new_bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
