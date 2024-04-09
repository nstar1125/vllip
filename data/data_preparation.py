import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# Concate 16 shot images into a single image, 
# the concated images are used for speeding up pre-training. 
# Matrix size of the concated image: [16x3]
def concate_pic(shot_info, img_path, save_path, row=16):
    for imdb, shot_num in tqdm(shot_info.items()): # id와 총 shot 수, 압축율 
        pic_num = shot_num // row # 압축 이미지 수
        compressed_dir = f"{save_path}/{imdb}"
        if os.path.isdir(compressed_dir): #이어하기
            continue
        for item in range(pic_num):
            img_list = []
            for idx in range(row): #16번 반복 = 압축 이미지에 들어갈 샷 선택
                shot_id = item * row + idx # 순서대로 반복
                img_name_0 = f"{img_path}/{imdb}/shot_{str(shot_id).zfill(4)}_img_0.jpg"
                img_name_1 = f"{img_path}/{imdb}/shot_{str(shot_id).zfill(4)}_img_1.jpg"
                img_name_2 = f"{img_path}/{imdb}/shot_{str(shot_id).zfill(4)}_img_2.jpg" # shot 당 이미지 3개
                img_0 = cv2.imread(img_name_0)
                img_1 = cv2.imread(img_name_1)
                img_2 = cv2.imread(img_name_2)
                img = np.concatenate([img_0,img_1,img_2],axis=1) #한 줄로 연결 = 1x3
                img_list.append(img)
            full_img = np.concatenate(img_list,axis=0) # 1x3 row 이미지들을 연결하여 16x3 만듬
            # print(img.shape)
            # print(full_img.shape)
            new_pic_dir = f"{save_path}/{imdb}/"
            if not os.path.isdir(new_pic_dir):
                os.makedirs(new_pic_dir)
            filename = new_pic_dir + str(item).zfill(4) + '.jpg' #새로운 레이블링
            cv2.imwrite(filename, full_img) # 저장

# Number of shot in each movie
def _generate_shot_num(new_shot_info='./data/MovieNet_shot_num.json'): #./MovieNet_shot_num.json 만드는 과정
    shot_info = './data/MovieNet_1.0_shotinfo.json' #train shot num 0부터 레이블링
    shot_split = './data/movie1K.split.v1.json' #train, test, val로 영화 분리
    with open(shot_info, 'rb') as f:
        shot_info_data = json.load(f)
    with open(shot_split, 'rb') as f:
        shot_split_data = json.load(f)
    new_shot_info_data = {}
    _type = ['train','val','test']
    for _t in _type:
        new_shot_info_data[_t] = {} #train,val,test
        _movie_list = shot_split_data[_t] #train,val,test의 영화 IMDB 목록
        for idx, imdb_id in enumerate(_movie_list):
            shot_num = shot_info_data[_t][str(idx)] #train,val,test에서 label -> shotnum
            new_shot_info_data[_t][imdb_id] = shot_num #imdb_id:shotnum 형식으로 train,val,test 만듬 = './MovieNet_shot_num.json'
    with open(new_shot_info, 'w') as f:
        json.dump(new_shot_info_data, f, indent=4) #저장

    
def process_raw_label(_T = 'train', raw_root_dir = './data/'):
    split = 'movie1K.split.v1.json'
    data_dict = json.load(open(os.path.join(raw_root_dir,split))) #./data/movie1K.split.v1.json = IMDB를 train, val, test로 나눔

    # print(data_dict.keys())
    # dict_keys(['train', 'val', 'test', 'full'])
    # print(len(data_dict['train'])) # 660
    # print(len(data_dict['val']))  # 220
    # print(len(data_dict['test'])) # 220
    # print(len(data_dict['full'])) # 1100

    data_list = data_dict[_T] #각자 train, test, val

    # annotation
    annotation_path = 'annotation'
    count = 0
    video_list = []
    # all annotations
    for index,name in enumerate(data_list): #train, test, val의 imdb list
        # print(name)
        annotation_file = os.path.join(raw_root_dir, annotation_path, name+'.json') #./data/annotation/<imdb>.json
        data = json.load(open(annotation_file))
        # only need sence seg labels
        if data['scene'] is not None: #무비넷에서 Scene이 annotation된 데이터만 선택 
            video_list.append({'name':name,'index':index}) #name, id추가
            count += 1
    print(f'scene annotations num: {count}')
    return video_list #name, id의 리스트 = scene annotation이 된 영화들로 필터링



# GT generation
def process_scene_seg_lable(scene_seg_path = '../data/movienet/scene318/label318',
    scene_seg_label_json_name = './data/movie1K.scene_seg_318_name_index_shotnum_label.v1.json',
    raw_root_dir = './data/'):
    def _process(data):
        seg_label = []
        for i in data: #name, idx, shotcount, shot(4자리)에 대한 label
            name = i['name']
            index = i['index']
            label = []
            with open (os.path.join(scene_seg_path,name+'.txt'), 'r') as f:
                shotnum_label = f.readlines()
            for i in shotnum_label:
                if ' ' in i:
                    shot_id = i.split(' ')[0].strip()
                    l = i.split(' ')[1].strip()
                    label.append((shot_id,l))
            shot_count = len(label) + 1
            seg_label.append({"name":name, "index":index, "shot_count":shot_count, "label":label })
        return seg_label

    train_list = process_raw_label('train',raw_root_dir)
    val_list = process_raw_label('val',raw_root_dir)
    test_list = process_raw_label('test',raw_root_dir) #train, val, test의 name, idx list
    data = {'train':train_list, 'val':val_list, 'test':test_list} # = 필터링 된 영화들

    # CVPR20SceneSeg GT
    train = _process(data['train'])
    test = _process(data['test'])
    val = _process(data['val']) #name, idx, shotcount, shot(4자리)에 대한 label 저장된 data
    d_all = {'train':train, 'val':val, 'test':test}
    
    with open(scene_seg_label_json_name,'w') as f:
        f.write(json.dumps(d_all)) #'./movie1K.scene_seg_318_name_index_shotnum_label.v1.json'에 저장 = #name, idx, shotcount, shot(4자리)
   


if __name__ == '__main__':
    # Path of movienet images
    img_path = '../data/movienet/240P_frames'

    # Shot number
    shot_info = './data/MovieNet_shot_num.json' 
    _generate_shot_num(shot_info) #train,val,test의 영화마다 총 shot 수 저장하는 './MovieNet_shot_num.json' 생성

    # GT label
    scene_seg_label_json_name = './data/movie1K.scene_seg_318_name_index_shotnum_label.v1.json'
    ## Download LGSS Annotation from: https://github.com/AnyiRao/SceneSeg/blob/master/docs/INSTALL.md
    ## 'scene_seg_path' is the path of the downloaded annotations
    scene_seg_path = '../data/movienet/scene318/label318' #movie에 대한 샷의 boundary label 있음
    ## Path of raw MovieNet 
    raw_root_dir = './data/'
    process_scene_seg_lable(scene_seg_path ,scene_seg_label_json_name, raw_root_dir)

    # Concate images
    save_path = './compressed_shot_images' # 저장 장소
    with open(shot_info, 'rb') as f:
        shot_info_data = json.load(f) #'./movie1K.scene_seg_318_name_index_shotnum_label.v1.json' = #name, idx, shotcount, shot(4자리)
    concate_pic(shot_info_data['train'], img_path, save_path) # 사진 압축 = train set
            
