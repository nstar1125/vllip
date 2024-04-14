import subprocess
import os
import json
import argparse


def segment_scenes(input_dir, output_dir):
    height = 240
    image_name = "shot-$SCENE_NUMBER-img-$IMAGE_NUMBER"
    for video in os.listdir(input_dir):
        if video.endswith(".mkv"):
            os.makedirs(f"{output_dir}/{video[:-4]}")
            cmd = [
                "scenedetect", "-i", f"{input_dir}/{video}", "save-images",
                "-f", image_name,
                "-o", f"{output_dir}/{video[:-4]}", "-H", str(height)
            ]
            subprocess.run(cmd)

def change_filename(output_dir):
    for files_dir in os.listdir(output_dir):
        files_path = os.path.join(output_dir, files_dir)
        for filename in os.listdir(files_path):
            if filename.startswith("shot-") and filename.endswith(".jpg"):
                # 숫자 부분 추출
                shot_number = filename.split("-")[1] #0001
                img_number = filename.split("-img-")[1][:-4] #01
            
                # 새로운 파일 이름 생성
                new_filename = f"shot_{int(shot_number)-1:04d}_img_{int(img_number)-1:d}.jpg" #0001->000, 01->0
                
                # 파일 이동 및 이름 변경
                os.rename(os.path.join(files_path, filename), os.path.join(files_path, new_filename))
    print("=> 이름 형식 변경완료")

def generate_json(output_dir):
    json_data = {"train": []}
    os.makedirs(f"{output_dir}/annotation")
    for files_dir in os.listdir(output_dir):
        files_path = os.path.join(output_dir, files_dir)
        if os.path.isdir(files_path):        
            video_data = {"name": files_dir, "label": []}
            for image_file in sorted(os.listdir(files_path)):
                if image_file.endswith("0.jpg"):
                    shot_number = image_file.split("_")[1]
                    video_data["label"].append([shot_number, "0"]) #label 데이터 필요없으니 전부으로 채움
            json_data["train"].append(video_data)
        if  files_dir!="annotation":
            with open(f"{output_dir}/annotation/{files_dir}.json", "w") as json_file:
                json.dump(json_data, json_file, indent=2)
        print("=> JSON 파일 생성완료")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str, default='./')
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    segment_scenes(cfg.i, cfg.o)
    change_filename(cfg.o)
    generate_json(cfg.o)

if __name__ == '__main__':
    cfg = get_config()
    main(cfg)