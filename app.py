from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from wav2lip_models import Wav2Lip
import platform
from face_parsing import init_parser, swap_regions
from basicsr.apply_sr import init_sr_model, enhance
import uuid

import gradio as gr

import json
import base64
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models

img_size = 96
wav2lip_batch_size = 128
save_as_video = True
no_sr = False
no_segmentation = False
no_smooth = False
rotate = False
resize_factor = 1
crop = [0, -1, 0, -1]
face_det_batch_size = 16
pads = [0,10,0,0]
fps = 25

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not no_smooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(mels, face_file):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    """
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    """

    reader = read_frames(face_file)

    for i, m in enumerate(mels):
        try:
            frame_to_save = next(reader)
        except StopIteration:
            reader = read_frames(face_file)
            frame_to_save = next(reader)

        face, coords = face_detect([frame_to_save])[0]

        face = cv2.resize(face, (img_size, img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def read_frames(face_file):
    if face_file.split('.')[1] in ['jpg', 'png', 'jpeg']:
        face = cv2.imread(face_file)
        while 1:
            yield face

    video_stream = cv2.VideoCapture(face_file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    print('Reading video frames from start...')

    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        yield frame

def tts(scripts, speed, gender):
    try:
        voiceType = {
        "男声":101013,
        "女声":101011
        }
        # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
        # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
        # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
        cred = credential.Credential("AKID7B4LQSJ1PsGQMG2d0ILCONBsWxAC5vY6", "Tp4A5oLjOL4cwGPLF6j7F8pIahbHjndW")
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tts.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = tts_client.TtsClient(cred, "ap-chengdu", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.TextToVoiceRequest()
        params = {
            "Text": scripts,
            "SessionId": str(uuid.uuid4()),
            "Speed": speed/10,
            "VoiceType": voiceType[gender]
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个TextToVoiceResponse的实例，与请求对象对应
        resp = client.TextToVoice(req)
        # 输出json格式的字符串回包
        resp_str = resp.to_json_string()
        resp = json.loads(resp_str)
        base64_str = resp["Audio"]
        audio_path = f"temp/audio/{uuid.uuid4()}.wav"
        with open(audio_path, "wb") as wav_file:
            decode_string = base64.b64decode(base64_str)
            wav_file.write(decode_string)
        return audio_path

    except TencentCloudSDKException as err:
        print(err)

def main(face_file, scripts, speed, gender):

    audio_file = tts(scripts, speed, gender)
    
    outfile = f"results/{uuid.uuid4()}.mp4"
    checkpoint_path = "checkpoints/wav2lip_gan.pth"
    segmentation_path = "checkpoints/face_segmentation.pth"
    sr_path = "checkpoints/esrgan_yunying.pth"


    video_stream = cv2.VideoCapture(face_file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    video_stream.release()


    if not audio_file.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        audio_file = 'temp/temp.wav'

    wav = audio.load_wav(audio_file, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    batch_size = wav2lip_batch_size
    gen = datagen(mel_chunks, face_file)

    abs_idx = 0
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            print("Loading segmentation network...")
            seg_net = init_parser(segmentation_path)

            print("Loading super resolution model...")
            sr_net = init_sr_model(sr_path)

            model = load_model(checkpoint_path)
            print ("Model loaded")

            frame_h, frame_w = next(read_frames(face_file)).shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c

            if no_sr:
                p = enhance(sr_net, p)
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            if no_segmentation:
                p = swap_regions(f[y1:y2, x1:x2], p, seg_net)

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file, 'temp/result.avi', outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

    torch.cuda.empty_cache()
    return outfile


gr.Interface(main,
    inputs=[gr.Video(type="filepath"),
            "text",
            gr.Slider(0.5, 2, value=1, label="语速"),
            gr.Radio(["男声", "女声"], label="性别")],
    # gr.Image(type="pil"),
    outputs = "playable_video", 
    title="数字人像",
    description="基于文字的lipsync").launch(server_name="0.0.0.0", server_port=30223, share=True)
