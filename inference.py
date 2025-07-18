import lighthouse
import subprocess
from lighthouse.models import CGDETRPredictor
import os
import torch
import numpy as np
from video_edit import cut_video_ffmpeg_fast
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


def load_weights(weight_dir: str) -> None:
    if not os.path.exists(os.path.join(weight_dir, 'clip_slowfast_cg_detr_qvhighlight.ckpt')):  
        command = 'wget -P gradio_demo/weights/ https://zenodo.org/records/13960580/files/clip_slowfast_cg_detr_qvhighlight.ckpt'
        subprocess.run(command, shell=True)

    if not os.path.exists('SLOWFAST_8x8_R50.pkl'):
        subprocess.run('wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl', shell=True)

    if not os.path.exists('Cnn14_mAP=0.431.pth'):
        subprocess.run('wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth', shell=True)


def get_video_duration(video_path):
    """영상의 총 길이를 초 단위로 반환. GPT 말대로라면 메타데이터만 읽기 때문에 빠르다고 한다"""
    with VideoFileClip(video_path) as video:
        return video.duration

def cut_and_save(title: str, video_path: str, duration=1000):
    """
    video_path의 영상을 duration 초 단위로 자른 mp4들을 저장

    Args:
        video_path (str): 자를 원본 영상 경로
        duration (int): 자를 구간 길이 (초)
    """
    # 전체 길이 구하기
    total_duration = get_video_duration(video_path)
    print(f"Total duration: {total_duration:.2f} seconds")

    #base_name = os.path.splitext(os.path.basename(video_path))[0]

    save_dir = os.path.dirname(video_path)

    # duration 단위로 나누기
    num_segments = int(total_duration // duration) + 1

    output_paths=[]

    for i in range(num_segments):
        start_time = i * duration
        segment_duration = min(duration, total_duration - start_time)
        if segment_duration <= 0:
            break

        output_path = os.path.join(save_dir, f"{title}_{start_time}_{int(start_time + segment_duration)}.mp4")

        output_paths.append(output_path)

        if os.path.exists(output_path):
            print(f"파일이 이미 존재합니다: {output_path} — 건너뜁니다.")
            continue

        print(f"Saving segment: {output_path}")
        cut_video_ffmpeg_fast(
            input_path=video_path,
            output_path=output_path,
            start_time=start_time,
            duration=segment_duration
        )
        
    return output_paths

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

def second_clipping(video_path, times) :
    save_dir = os.path.dirname(video_path)
    output_paths=[]
    for interval in times :
        start_time=interval[0]
        duration=interval[1]-interval[0]
        output_path = os.path.join(save_dir, f"{title}_{int(start_time)}_{int(interval[1])}.mp4")

        output_paths.append(output_path)
        if os.path.exists(output_path):
            print(f"파일이 이미 존재합니다: {output_path} — 건너뜁니다.")
            continue

        cut_video_ffmpeg_fast(
                input_path=video_path,
                output_path=output_path,
                start_time=start_time,
                duration=duration
            )
        
    return output_paths

def main_inference(video_paths, query, THRESHOLD=0.7):
    
    start_time=0

    result=[]

    for video_path in tqdm(video_paths, desc="Processing videos"):

        print("Processing:", video_path)

        base_name = os.path.splitext(os.path.basename(video_path))[0]      

        print(base_name) 

        start_time = int(base_name.split('_')[-2])          
        npz_path = f"./data/{base_name}.npz"                            


        if os.path.exists(npz_path):                                        #이미 해당 비디오를 해당 간격으로 자른 feature가 존재한다면
            print(f"Loading cached features from: {npz_path}")
            data = np.load(npz_path)
            v_feat = torch.tensor(data["v_feat"]).to(device)  
            v_mask = torch.tensor(data["v_mask"]).to(device)
            
            video=(v_feat, v_mask)
            #print(video)

        else:
            print("Encoding video...")
            video = model._vision_encoder.encode(video_path)
            v_feat = video[0].cpu().numpy()
            v_mask = video[1].cpu().numpy()
            np.savez(npz_path, v_feat=v_feat, v_mask=v_mask)
            print(f"Saved features to: {npz_path}")


        N = video[0].size(0)  # feature 개수
        print(f"dimension :{N}")
        timestamps = torch.arange(N, dtype=torch.float32)

        # 시작과 끝 타임스탬프 계산 (총 구간: [0, 1])
        start_times = timestamps / N          # (k - 1) / N
        end_times = (timestamps + 1) / N      # k / N

        # 두 열을 세로로 쌓아서 (N, 2) shape로 만듦
        time_info = torch.stack([start_times, end_times], dim=1).to(device)

        # 기존 feature vector와 concatenate
        features_with_timestamps = (torch.cat([video[0], time_info], dim=1), video[1])

        v_feat={}
        v_feat["video_feats"]=features_with_timestamps[0].unsqueeze(0)
        v_feat["video_mask"]=video[1]
        v_feat["audio_feats"]=None

        #print(v_feat)

        output=model.predict(query, v_feat)
        print(output['pred_relevant_windows'])
        scores = output['pred_saliency_scores']
        scores = np.array(scores)

        high_score_indices = [w for w in output['pred_relevant_windows'] if w[2]>=THRESHOLD]
        high_score_indices = sorted(high_score_indices, key=lambda w: w[2], reverse=True)
        #high_score_start_times = [(seconds_to_hms(t[0]+start_time), seconds_to_hms(t[1]+start_time)) for t in high_score_indices]

        high_score_start_times = [(t[0]+start_time, t[1]+start_time) for t in high_score_indices]
        result.append(high_score_start_times)

    return result
  


device: str = "cuda" if torch.cuda.is_available() else "cpu"
weight_dir: str = './gradio_demo/weights'
weight_path: str = os.path.join(weight_dir, 'clip_slowfast_cg_detr_qvhighlight.ckpt')
load_weights(weight_dir)
model: CGDETRPredictor = CGDETRPredictor(weight_path, device=device, feature_name='clip_slowfast', 
                                        slowfast_path='SLOWFAST_8x8_R50.pkl', pann_path=None)
print(dir(model._vision_encoder))

title="shuka_"


#video_path = f"/home/lasker06/workspace/junhyeok/video/{title}.mp4"              #여기서 로드
video_path = f"/home/lasker06/workspace/data/contents/test1.mp4"
query = "a man wearing a glasses and a blue cap is rasing hand"
video_paths=cut_and_save("test1", video_path, 500)                       #1000초 간격으로 잘라서 하나씩 저장한 경로들의 리스트를 반환
result=main_inference(video_paths, query)


print("1차 탐색 결과 :")
#print(result)
flattened = [t for sublist in result for t in sublist]
result_hms= [[(seconds_to_hms(r[0]), seconds_to_hms(r[1])) for r in sublist] for sublist in result]
print(result_hms)

flattened = [t for sublist in result for t in sublist]

video_paths=second_clipping(video_path, flattened)
result=main_inference(video_paths, query, THRESHOLD=0.8)


print("2차 탐색 결과 :")
result_hms= [[(seconds_to_hms(r[0]), seconds_to_hms(r[1])) for r in sublist] for sublist in result]
print(result_hms)
