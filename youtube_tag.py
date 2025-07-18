from googleapiclient.discovery import build
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import isodate
import re
import spacy
from VL_models.clip import Clip_model
import csv
import random

EXCLUDED_PROPER_NOUNS = {"vlog", "v-log", "youtube", "youtuber", "disney", "netflix"}
API_KEY = "AIzaSyAOHngK8h-LgUkyfwg14og0LW5y1KC-6t8"

def extract_channel_id(channel_url):
    # 채널 ID 직접 포함된 경우
    if "channel/" in channel_url:
        return channel_url.split("channel/")[1].split("/")[0]
    elif "@" in channel_url:
        # 핸들 -> userName 추출
        return channel_url.split("@")[1]
    else:
        raise Exception("URL 형식을 확인해주세요.")

def resolve_channel_id(youtube, handle_or_id):
    if re.match(r"^[A-Za-z0-9_-]{24}$", handle_or_id):
        # 이미 channelId일 가능성
        return handle_or_id
    # 핸들 -> channelId 찾기 (1회만 호출)
    response = youtube.search().list(
        part="snippet",
        q=handle_or_id,
        type="channel",
        maxResults=1
    ).execute()
    return response["items"][0]["snippet"]["channelId"]

def get_uploads_playlist_id(youtube, channel_id):
    response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

def get_videos_from_playlist(youtube, playlist_id, max_videos=50):
    video_ids = []
    next_page_token = None
    while len(video_ids) < max_videos:
        try:
            response = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            for item in response["items"]:
                video_ids.append(item["contentDetails"]["videoId"])
                if len(video_ids) >= max_videos:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
            break  # 다른 예외 발생 시, 해당 반복을 건너뜀
    return video_ids

def get_longform_video_infos(youtube, video_ids, top_n=3):
    longform_infos = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk)
        ).execute()
        for item in response["items"]:
            if "contentDetails" not in item or "duration" not in item["contentDetails"]:
                continue

            duration = isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds()
            if duration >= 60:
                longform_infos.append({
                    "제목": item["snippet"]["title"],
                    "태그": item["snippet"].get("tags", ["(태그 없음)"]),
                    "설명": item["snippet"]["description"],
                    "조회수": item["statistics"].get("viewCount", "N/A"),
                    "길이": f"{int(duration//60)}분 {int(duration%60)}초"
                })
            
        if len(longform_infos) >= top_n:
            break
    return sorted(longform_infos, key=lambda x: int(x["조회수"]), reverse=True)[:top_n]




# spaCy 영어 모델 로드
nlp = spacy.load("en_core_web_sm")

def is_proper_noun_in_english(tag):
    # 영어 텍스트를 spaCy로 분석
    doc = nlp(tag)
    # 고유명사(NNP) 여부 체크
    
    return any(token.pos_ == 'PROPN' and token.text not in EXCLUDED_PROPER_NOUNS for token in doc)

# 예시로 구글 번역 후 고유명사 판별
def filter_proper_nouns(tags):
    english_tags = google_translate(tags)  # 한국어 태그를 영어로 번역
    filtered_tags = [tag for tag in english_tags if not is_proper_noun_in_english(tag)]
    return filtered_tags

def translate_tag(tag):
    try:
        return GoogleTranslator(source='ko', target='en').translate(tag).lower()
    except Exception as e:
        print(f"번역 실패: \"{tag}\" -> {e}")
        return None

def google_translate(korean_tags):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(translate_tag, korean_tags)
        return [r for r in results if r is not None]



def get_category_similarity(tags):
    category_similarity={}
    tag_features=clip_model.get_text_feature(texts=tags) 
    for key, value in category_feature_list.items() :
        for tag_feat in tag_features :
            if key in category_similarity :
                category_similarity[key]=clip_model.get_similarity(value, tag_feat).item()+category_similarity[key]
            else :
                category_similarity[key]=clip_model.get_similarity(value, tag_feat).item()

    

    return {k: v / len(tag_features) for k, v in category_similarity.items()}


def extract_from_url(channel_url) :
    handle_or_id = extract_channel_id(channel_url)
    channel_id = resolve_channel_id(youtube, handle_or_id)
    playlist_id = get_uploads_playlist_id(youtube, channel_id)
    video_ids = get_videos_from_playlist(youtube, playlist_id, max_videos=100)
    longform_videos = get_longform_video_infos(youtube, video_ids, top_n=3)
    return longform_videos

def save_dict_random_sample(style_dict, output_file="channel_tags.csv", min_n=5, max_n=8):
    with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Tags"])  # CSV 헤더

        for category, tag_set in style_dict.items():
            n = min(len(tag_set), random.randint(min_n, max_n))  # tag 개수가 적으면 전체 사용
            sampled_tags = random.sample(list(tag_set), n)
            tag_str = ",".join(sampled_tags)
            writer.writerow([category, tag_str])

    print(f"✅ 무작위 샘플링된 CSV 저장 완료: {output_file}")
    
    


category_feature_list={}
clip_model = Clip_model()
with open("category_keyword0411.csv", "r") as w:
    raw_data=csv.reader(w)
    category_data=[row for row in raw_data]

for row in category_data:                #category의 feature를 모두 뽑아서 저장해둠. id -> text feature 
    #print(value)
    category_feature_list[row[0]] = clip_model.get_text_feature(texts= row[1]) 


# 실행 부분
youtube = build("youtube", "v3", developerKey=API_KEY)

file_path = "/home/lasker06/workspace/junhyeok/channel_matching/channel_list.txt"

# 파일을 열고 한 줄씩 읽기
with open(file_path, "r", encoding="utf-8") as f:
    channel_links = f.readlines()

# 읽어온 링크 출력 (각각의 링크에서 \n을 제거)
channel_links = [link.strip() for link in channel_links]


lv={}

for link in channel_links :
    result=extract_from_url(link)
    lv[link]=result

category_best_channel={}

keys = channel_links
tags = {link: set() for key in keys}


for link in channel_links :
    i=0
    for idx, video in enumerate(lv[link], 1):
        i+=1
        print(f"\n--- 롱폼 영상 {idx} ---")
        print("제목:", video["제목"])
        #print("조회수:", video["조회수"])
        #print("길이:", video["길이"])
        print("태그:", ", ".join(video["태그"]))
        #print("설명:", video["설명"][:300] + "...")
    
        translated = google_translate(video["태그"])
        filtered_tag=filter_proper_nouns(translated)
        tags[link].update(filtered_tag)                                    #세 영상의 필터링된 태그들을 set에다가 전부 집어넣음

        if not filtered_tag :
            continue

        if i==1 :
            cat=get_category_similarity(filtered_tag)
        else :
            cat=Counter(get_category_similarity(filtered_tag))+Counter(cat)
        
        print("태그 eng:", ", ".join(filtered_tag))
        
    output_path = "/home/lasker06/workspace/junhyeok/channel_matching/results/channel_category.txt"

    if not cat :                    #세 영상 모두 tag가 없어졌을 때 
        continue

    sorted_items=sorted(cat.items(), key=lambda x: x[1], reverse=True)
    print(sorted_items)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"{link}\n")
        for key, value in sorted_items:
            f.write(f"{key}: {value}\n")
            if not key in category_best_channel :
                category_best_channel[key]=[link]
            category_best_channel[key].append(link)
            break
    

save_dict_random_sample(tags)

print(category_best_channel)

with open("channel_links.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "urls"])  # 헤더 변경
    for key, urls in category_best_channel.items():
        joined_urls = ", ".join(urls)
        writer.writerow([key, joined_urls])


    
