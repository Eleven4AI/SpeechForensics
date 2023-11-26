import cv2
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
from utils import landmarks_interpolate, crop_patch, write_video_ffmpeg
import argparse
import subprocess
from glob import glob
import shutil
import json


#Extract mouth ROI
STD_SIZE = (256, 256)
mean_face_path = "data/20words_mean_face.npy"
mean_face_landmarks = np.load(mean_face_path)
stablePntsIDs = [33, 36, 39, 42, 45]
def preprocess_video(input_video_path, output_video_path,landmark_path=None,fps=25):
    if not osp.exists(landmark_path):
        print(f'The {landmark_path} is not found!')
        return

    landmarks=[]
    with open(landmark_path,'r') as f:
        landmarks_dict=json.load(f)
    for key,value in landmarks_dict.items():
        if not value is None:
            landmarks.append(np.asarray(value))
        else:
            landmarks.append(value)

    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if preprocessed_landmarks is None:
        return
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE,
                      window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, ffmpeg='/usr/bin/ffmpeg',fps=fps)
    return

def align_mouth(args):
    video_root=args.video_root
    file_list=args.file_list
    landmark_dir=args.landmark_dir
    out_dir=args.out_dir

    with open(file_list,'r') as f:
        video_list=f.read().split('\n')

    for video_item in tqdm(video_list):
        video_path=osp.join(video_root,video_item.split(' ')[0])
        landmark_path=video_path.replace(video_root,landmark_dir).replace('.mp4','.json')
        mouth_roi_path=video_path.replace(video_root,out_dir)
        wav_path=mouth_roi_path.replace('.mp4','.wav')
        if (os.path.exists(wav_path)):
            continue

        cap=cv2.VideoCapture(video_path)
        fps=cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        preprocess_video(video_path, mouth_roi_path,landmark_path=landmark_path,fps=fps)
        if not os.path.exists(mouth_roi_path):
            continue
        cmd = "ffmpeg -i " + video_path + " -f wav -vn -ar 16000 -ac 1 -y " + wav_path + ' -loglevel quiet'
        subprocess.call(cmd, shell = True)
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extracting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_root', type=str,required=True,help='video root dir')
    parser.add_argument('--file_list',type=str,required=True,help='file list')
    parser.add_argument('--landmark_dir',type=str,required=True,help='landmark dir')
    parser.add_argument('--out_dir', type=str,required=True,help='out dir')
    parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg',
                        help='ffmpeg path')
    args = parser.parse_args()

    align_mouth(args)
