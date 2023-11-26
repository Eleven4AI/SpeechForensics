import cv2
import numpy as np
import skvideo
skvideo.setFFmpegPath('/usr/bin')
import skvideo.io
from tqdm import tqdm
import os
import os.path as osp
import sys
from av_hubert.avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from base64 import b64encode
import tempfile
from argparse import Namespace
from av_hubert.fairseq import checkpoint_utils, options, tasks, utils
from av_hubert.fairseq.dataclass.configs import GenerationConfig
from glob import glob
from scipy.io import wavfile
import shutil
import av_hubert.avhubert.utils as avhubert_utils
import soundfile as sf
import json
import torch.nn.functional as F
from sklearn import metrics


def calc_cos_dist(feat1,feat2,vshift=15):
    feat1=torch.nn.functional.normalize(feat1,p=2,dim=1)
    feat2 = torch.nn.functional.normalize(feat2, p = 2, dim = 1)
    if len(feat1)!=len(feat2):
        sample=np.linspace(0,len(feat1)-1,len(feat2),dtype = int)
        feat1=feat1[sample.tolist()]
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.cosine_similarity(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]).cpu().numpy())
    dists=np.asarray(dists)
    return dists

def extract_visual_feature(video_path,max_length):
    transform = avhubert_utils.Compose([
      avhubert_utils.Normalize(0.0, 255.0),
      avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
      avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])

    video = cv2.VideoCapture(video_path)
    fps=video.get(cv2.CAP_PROP_FPS)
    video.release()
    if len(frames)>fps*max_length:
        frames=frames[:int(fps*max_length)]

    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda(0)
    with torch.no_grad():
        feature, _, = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
        feature = feature.squeeze(dim=0)
    return feature

def stacker(feats, stack_order):
    """
    Concatenating consecutive audio frames
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis = 0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats

def extract_audio_feature(audio_path):
    sample_rate, wav_data = wavfile.read(audio_path)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    audio_feats = logfbank(wav_data, samplerate = sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, 4)
    audio_feats=torch.FloatTensor(audio_feats).cuda(0)
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    audio_feats=audio_feats.transpose(0,1).unsqueeze(dim=0)

    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer
        feature, _,= model.extract_finetune(source = {'video': None, 'audio': audio_feats}, padding_mask = None,
                                            output_layer = None)
        feature = feature.squeeze(dim = 0)
    return feature

tmp_dir = tempfile.mkdtemp()
def evaluate_audio_visual_feature(mouth_roi_path,wav_path,max_length=50):
    #trim audio
    wav,sr=sf.read(wav_path)
    if len(wav)>sr*max_length:
        wav_path=osp.join(tmp_dir,'audio.wav')
        if osp.exists(wav_path):
            os.remove(wav_path)
        sf.write(wav_path,wav[:sr*max_length],sr)

    visual_feature=extract_visual_feature(mouth_roi_path,max_length)
    audio_feature=extract_audio_feature(wav_path)

    dist=calc_cos_dist(visual_feature.cpu(),audio_feature.cpu()) #cosine
    dist=dist.mean(axis = 0)
    dist=dist.max()

    return float(dist)

def evaluate_auc(args):
    video_root=args.video_root
    file_list=args.file_list
    cropped_mouth_dir=args.mouth_dir
    max_length=args.max_length

    with open(file_list,'r') as f:
        video_list=f.read().split('\n')

    outputs=[]
    labels=[]
    for video_item in tqdm(video_list):
        video_path=osp.join(video_root,video_item.split(' ')[0])
        video_label=video_item.split(' ')[1]
        mouth_roi_path=video_path.replace(video_root,cropped_mouth_dir)
        wav_path=mouth_roi_path.replace('.mp4','.wav')
        if not  ((osp.exists(mouth_roi_path) and osp.exists(wav_path))):
            continue
        #Extract visual and audio speech representations respectively and compute their cosine similarity
        sim=evaluate_audio_visual_feature(mouth_roi_path,wav_path,max_length)

        outputs.append(sim)
        labels.append(int(video_label))

    outputs=np.asarray(outputs)
    labels=np.asarray(labels)
    fpr,tpr,_ = metrics.roc_curve(labels,outputs)
    auc=metrics.auc(fpr, tpr)
    print(len(outputs))
    print('AUC:{}'.format(auc))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extracting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path',type=str,default='./checkpoints/large_vox_iter5.pt',help='checkpoint path')
    parser.add_argument('--video_root', type=str,required=True,help='video root dir')
    parser.add_argument('--file_list',type=str,required=True,help='file list')
    parser.add_argument('--mouth_dir',type=str,required=True,help='cropped mouth dir')
    parser.add_argument('--max_length',type=int, default=50, help='maximum video length consumed by model')
    parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg',
                        help='ffmpeg path')
    args = parser.parse_args()

    ckpt_path = args.checkpoint_path
    user_dir = os.getcwd()
    utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")
    model.cuda(0)
    model.eval()

    evaluate_auc(args)
