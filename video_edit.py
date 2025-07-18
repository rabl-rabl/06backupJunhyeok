import subprocess

def cut_video_ffmpeg_fast(input_path, output_path, start_time, duration):
    """
    ffmpeg을 사용하여 빠르게 영상 자르기 (재인코딩 없이 스트림 복사)
    단점: 키프레임 단위로만 정확도 보장
    """
    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",  # 스트림 복사
        output_path
    ]
    subprocess.run(command, check=True)



