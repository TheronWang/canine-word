import bisect
import copy
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from time import sleep

import librosa
import numpy as np
import pandas as pd
import requests
import soundfile
from tqdm import tqdm

def get_youtube_h():
    mpath = '/data/hridayesh/dog_video_downloader/downloaded_videos_metadata.json.csv'
    metadatas = pd.read_csv(mpath)
    new_metadatas = []
    for _, row in metadatas.iterrows():
        channel_name = row['channel_name']
        video_url = row['video_url']
        breed = row['breed']
        new_metadatas.append([video_url, channel_name, breed, row])
    new_metadatas.sort(key=lambda n: n[0])

    video_list_path = '/data/dog_videos/german_shepherd/video_list.txt'
    video_root = '/data/dog_videos/german_shepherd/output'
    b = 'german shepherd'
    output_path = '/data/xingyuan/animal_language/dog_data/german_shepherd/urls.txt'

    video_list = []
    with open(video_list_path, 'r', encoding='utf-8') as file:
        datas = [line.strip() for line in file.readlines() if len(line.strip()) > 0]

    with open(output_path, 'w+', encoding='utf-8') as _: pass
    video_urls = [n[0] for n in new_metadatas]
    for d in tqdm(datas):
        v_id = os.path.basename(d)[:-len('.mp4')]
        c_n = os.path.basename(os.path.dirname(d))
        v_url = f'https://www.youtube.com/watch?v={v_id}'
        nd = copy.deepcopy(d)
        assert nd.find('./output/') == 0
        nd = nd.replace('./output/', '')
        video_path = os.path.join(video_root, nd)
        assert os.path.exists(video_path)

        ind = bisect.bisect_left(video_urls, v_url)
        if not (0 <= ind < len(new_metadatas)): continue
        if not (new_metadatas[ind][0] == v_url): continue
        if not (new_metadatas[ind][2] == b): continue
        sd = {
            'source': 'YouTube',
            'video_path': video_path,
            'url_temp': v_url
        }
        with open(output_path, 'a+', encoding='utf-8') as file:
            file.write(json.dumps(sd))
            file.write('\n')
            file.flush()

    with open(output_path, 'r', encoding='utf-8') as f:
        ds = [json.loads(line.strip()) for line in f.readlines() if len(line.strip()) > 0]
    print(len(ds))


def get_statistics_data(root):
    print(root)
    audio_root = os.path.join(root, 'audios')
    duration_path = os.path.join(root, 'duration.jsonlines')

    duration_data = []
    if os.path.exists(duration_path):
        with open(duration_path, 'r', encoding='utf-8') as file:
            duration_data = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        duration_data.sort(key=lambda d: d['path'])
    done_paths = [d['path'] for d in duration_data]

    durations = []
    for audio in tqdm(os.listdir(audio_root)):
        path = os.path.join(audio_root, audio)
        ind = bisect.bisect_left(done_paths, path)
        if (0 <= ind < len(done_paths)) and (duration_data[ind]['path'] == path):
            durations.append(duration_data[ind]['duration'])
        else:
            wav, sr = librosa.load(path, sr=None, mono=True)
            duration = wav.shape[0] / sr
            durations.append(duration)
            with open(duration_path, 'a+', encoding='utf-8') as file:
                file.write(json.dumps({'path': path, 'duration': duration}, ensure_ascii=False))
                file.write('\n')
                file.flush()

    print(f'total:\t{sum(durations) / 3600.}')
    print(f'mean:\t{np.mean(durations)}')
    print(f'max:\t{max(durations)}')
    print(f'min:\t{min(durations)}')
    print(f'count:\t{len(durations)}')


def add_tiktok_data(tiktok_path, audio_root, meta_path):
    # TODO
    with open(tiktok_path, 'r', encoding='utf-8') as file:
        tiktok_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    keys = ['id', 'channel_id', 'channel', 'uploader', 'uploader_id', 'channel_url', 'uploader_url', 'duration', 'title', 'description', 'upload_date']

    for td in tiktok_datas:
        new_metadata = {}
        for k in keys:
            if k not in td.keys(): continue
            print(k)
            print(td[k])
            new_metadata[k] = td[k]
        new_metadata['source'] = 'TikTok'
        new_metadata['video_path'] = os.path.join(audio_root)
        new_metadata['name'] = ''
        exit()

def add_reddit_data(video_root, meta_path):
    def get_author_fullname(video_id):
        video_url = f"https://v.redd.it/{video_id}"
        response = requests.get(video_url, headers={'User-agent': 'Mozilla/5.0'}, allow_redirects=True)
        final_url = response.url
        json_url = final_url + '.json'
        sleep(1.)
        json_response = requests.get(json_url, headers={'User-agent': 'Mozilla/5.0'})
        if json_response.status_code == 200:
            try:
                data = json_response.json()
                author_fullname = data[0]['data']['children'][0]['data']['author_fullname']
                return author_fullname
            except (IndexError, KeyError) as e:
                print(f"Error extracting author_fullname: {e}")
        else:
            print(f"Failed to retrieve data from the .json URL. Status code: {json_response.status_code}")

    def find_name(names, name):
        names.sort()
        ind = bisect.bisect_left(names, name)
        return (0 <= ind < len(names)) and (names[ind] == name)

    with open(meta_path, 'r', encoding='utf-8') as file:
        meta_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    names = [md['name'] for md in meta_datas]
    names.sort()

    print(f'meta count:\t{len(meta_datas)}')
    for video in tqdm(os.listdir(video_root)):
        if not video.startswith('merged_'): continue
        video_id = video[len('merged_'): video.rfind('_')]
        user_id = get_author_fullname(video_id)
        assert '+' not in user_id
        assert len(video_id) == len('lwwlsys2ffpd1')

        name = f'{user_id}+{video_id}'
        if find_name(names=names, name=name): name = f'{name}+Reddit'
        if find_name(names=names, name=name): continue

        meta_data = {
            'id': video_id,
            'user_id': user_id,
            'name': name,
            'video_path': os.path.join(video_root, video),
            'source': 'Reddit'
        }
        meta_datas.append(meta_data)
        sleep(1.)
    print(f'meta count:\t{len(meta_datas)}')

    save_path = os.path.join(os.path.dirname(meta_path), 'tmp_metadata.jsonlines')
    with open(save_path, 'w+', encoding='utf-8') as file:
        for md in meta_datas:
            file.write(json.dumps(md, ensure_ascii=False))
            file.write('\n')
            file.flush()

def get_statistics_data2(root):
    print(root)
    durations = []
    for audio in tqdm(os.listdir(root)):
        wav, sr = librosa.load(os.path.join(root, audio), sr=None, mono=True)
        durations.append(wav.shape[0] / sr)

    print(f'total:\t{sum(durations) / 3600.}')
    print(f'mean:\t{np.mean(durations)}')
    print(f'max:\t{max(durations)}')
    print(f'min:\t{min(durations)}')
    print(f'count:\t{len(durations)}')



if __name__ == '__main__':
    pass

    # 添加reddit数据
    # audio_root = '/data/hridayesh/dog video downloader reddit/reddit-video-downloader/husky/videos'
    # meta_path = '/data/xingyuan/animal_language/dog_data/husky/metadata.jsonlines'

    # audio_root = '/data/hridayesh/dog video downloader reddit/reddit-video-downloader/HuskyTantrums/videos'
    # meta_path = '/data/xingyuan/animal_language/dog_data/husky/metadata.jsonlines'

    # audio_root = '/data/hridayesh/dog video downloader reddit/reddit-video-downloader/pitbulls/videos'
    # meta_path = '/data/xingyuan/animal_language/dog_data/pitbull/metadata.jsonlines'
    # add_reddit_data(audio_root, meta_path)


    # roots = [
    #     '/data/xingyuan/animal_language/dog_data/chihuahua',
    #     '/data/xingyuan/animal_language/dog_data/german_shepherd',
    #     '/data/xingyuan/animal_language/dog_data/husky',
    #     '/data/xingyuan/animal_language/dog_data/labrador',
    #     '/data/xingyuan/animal_language/dog_data/pitbull',
    # ]
    # output_path = '/home/xingyuan/animal_language2/code/data/tmp_extract_audios.txt'
    # with open(output_path, 'w+') as _: pass
    # for root in roots:
    #     save_root = os.path.join(root, 'tmp_audios')
    #     if os.path.exists(save_root): shutil.rmtree(save_root)
    #     os.makedirs(save_root)
    #
    #     urls_path = os.path.join(root, 'urls.txt')
    #     with open(urls_path, 'r', encoding='utf-8') as file:
    #         datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    #     video_paths = [d['video_path'] for d in datas]
    #     for vp in video_paths:
    #         assert os.path.exists(vp)
    #         fn = os.path.basename(vp)[:-len('.mp4')]
    #         with open(output_path, 'a+', encoding='utf-8') as file:
    #             file.write(f'{vp},{os.path.join(save_root, fn)}.wav\n')
    #             file.flush()

    # path = '/data/hridayesh/dog_video_download_tiktok/chihuahua/tiktok_links_pudgywoke.json'
    # with open(path, 'r', encoding='utf-8') as file:
    #     datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    # print(datas[0].keys())

