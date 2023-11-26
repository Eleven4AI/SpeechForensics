import cv2, os
import numpy as np
import torch
from tqdm import tqdm
import os
import os.path as osp
import sys
from skimage import io
import face_alignment
from glob import glob
import json
import argparse


def detect_save_landmark_68(args):
    video_root=args.video_root
    file_list=args.file_list
    out_dir=args.out_dir

    with open(file_list,'r') as f:
        video_list=f.read().split('\n')

    for video_item in tqdm(video_list):
        video_path=osp.join(video_root,video_item.split(' ')[0])

        out_path=video_path.replace(video_root,out_dir).replace('.mp4','.json')
        os.makedirs(osp.dirname(out_path),exist_ok = True)

        if osp.exists(out_path):
            continue

        frames=[]
        cap=cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()

        frames=np.asarray(frames)

        landmarks = {}
        for i in range(len(frames)):
            frame=frames[i]
            landmark = fa.get_landmarks(frame)
            if (landmark!=None) and (len(landmark)!=0):
                landmark=landmark[0]
                landmark=landmark.tolist()
            img_name='%04d.jpg'%i
            landmarks[img_name]=landmark

        open_mode='w' if os.path.exists(out_path) else 'x'
        with open(out_path,open_mode) as f:
            json.dump(landmarks,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extracting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_root', type=str,required=True,help='video root dir')
    parser.add_argument('--file_list',type=str,required=True,help='file list')
    parser.add_argument('--out_dir', type=str,required=True,help='landmark dir')
    parser.add_argument('--face_detector', type=str,default='checkpoints/Resnet50_Final.pth',
                        help='path to face detector')
    parser.add_argument('--face_predictor', type=str,default='checkpoints/2DFAN4-cd938726ad.zip',
                        help='path to landmark predictor')
    parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg',
                        help='ffmpeg path')
    args = parser.parse_args()

    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,face_detector='retinaface',device='cuda',
                                    face_detector_kwargs={'path_to_detector':args.face_detector})

    detect_save_landmark_68(args)


