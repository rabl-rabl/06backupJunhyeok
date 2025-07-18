from googleapiclient.discovery import build
from llava_model import llava_next_video
from video_reader import PyVideoReader
import csv
import random
API_KEY = "AIzaSyAOHngK8h-LgUkyfwg14og0LW5y1KC-6t8"

prompt=[
{
"role": "user",
"content": [
    {"type": "text", "text": 

    """

    Given a advertisement, you should accurately predict purpose of the advertisement. 
    You must only categorize the video into one of these categories: accessories, Book, Cleaning supplies, Clothes, Construction, Cosmetic, Drink, Education, Electronic Device, Event, Finance, Food, furniture, Game, Insurance, Interior, Kitchenware, Leisure, Medicine, Movie, Music, Pets, Shoes, Shopping, Sports, Traffic, 운송/물류, Travel, Vehicle

    
    ###Output Format:
    1. You only provide an answer as one of below answers:

    accessories, Book, Cleaning supplies, Clothes, Construction, Cosmetic, Drink, Education, Electronic Device, Event, Finance, Food, furniture, Game, Insurance, Interior, Kitchenware, Leisure, Medicine, Movie, Music, Pets, Shoes, Shopping, Sports, Traffic, 운송/물류, Travel, Vehicle


    
    2. You must provide only three categories as an answer. Do not provide plural categories as your answer.

    Please only answer the category name and only CHOOSE from the categories listed.
    You must answer neat and concise without trailing plural sentences. 

    If you have to categorize the purpose of the advertisement, what would it be?
    """

},
    {"type": "video"},
        ],
},
]



def print_llava_result(ad_path: str, step_length: int = 16) -> None:

  
    try:
        video = PyVideoReader(ad_path)
    except FileNotFoundError:
        print(f"Error: The file at {ad_path} was not found.")
        return -1
    
    start_frame = 0
    end_frame = int(video.get_info()["frame_count"])

    step = max(int((end_frame - start_frame) / step_length), 1)
    frame_range = range(start_frame, end_frame, step)

    video_frames = video.get_batch(frame_range)

    #print(video_frames)

    


    lnv.prompt_processing(prompt, video_frames)
    result_string = lnv.video_inference()

    print(f"[LLAVA]")

    return result_string


def search_by_category(category) :
    with open("category_keyword0411.csv", "r") as w:
        raw_data=csv.reader(w)
        category_data=[row for row in raw_data]
    
    for row in category_data:
        if category.lower()==row[1].lower() :
            return row[0]
        
    print("Not matched. Try again.")


def extract_channel_id(url):
    # 유튜브 URL에서 채널 ID 추출
    match = re.search(r"(?:channel/|user/|@)([^/?]+)", url)
    return match.group(1) if match else None

def get_channel_details(api_key, url):
    channel_identifier = extract_channel_id(url)
    if not channel_identifier:
        raise ValueError("채널 ID 또는 사용자 이름을 추출할 수 없습니다.")

    youtube = build("youtube", "v3", developerKey=api_key)

    # ID가 채널 ID인지 사용자명인지 확인 후 처리
    try:
        # 사용자명인 경우 채널 ID로 변환
        channel_res = youtube.channels().list(
            forUsername=channel_identifier,
            part="snippet,statistics"
        ).execute()
        if not channel_res["items"]:
            raise Exception("해당 사용자명을 가진 채널을 찾을 수 없습니다.")
    except:
        # 채널 ID로 조회
        channel_res = youtube.channels().list(
            id=channel_identifier,
            part="snippet,statistics"
        ).execute()

    if not channel_res["items"]:
        raise Exception("채널을 찾을 수 없습니다.")

    channel = channel_res["items"][0]
    snippet = channel["snippet"]
    stats = channel["statistics"]

    #with open(channel_links, )

    # 결과 JSON 생성
    result = {
        "name": snippet.get("title"),
        "description": snippet.get("description"),
        "subscribers": format_number(stats.get("subscriberCount")),
        "avgViews": "N/A",  # 개별 영상 분석 필요
        "score": random.randint(70, 100),     #랜덤 점수
        "country": snippet.get("country", "Unknown"),
        "videoCount": stats.get("videoCount"),
        "tags": []  #이거 파일로 만드는 코드는 짜놨는데 돌릴 시간이 없어서 테스트 못해봄. 
    }

    return result

def format_number(number_str):
    try:
        number = int(number_str)
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.0f}K"
        else:
            return str(number)
    except:
        return "N/A"

def save_channel_info_json(api_key, url, output_path):
    data = get_channel_details(api_key, url)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 저장 완료: {output_path}")


groundtruth_dict={}

with open("ad_label.csv", "r") as w:
    raw_data=csv.reader(w)
    category_channels=[row for row in raw_data]  
    
for row in category_channels :
    groundtruth_dict[row[0]]=row[1]                 #정답 비교용 groundtruth 데이터

ad_path=["/media/lasker06/T7 Shield/tvcf/kr/2024/"+row[0] for row in category_channels]
#"/media/lasker06/T7 Shield/tvcf/kr/2024/o2024_10_21_r2024_10_21_n하나투어_20241101_115153.mp4"


lnv=llava_next_video()
result_list=[]
for path in ad_path :
    category_id=print_llava_result(path)
    if category_id==-1 :
        continue
    result_list.append((path, category_id))
    
with open("ad_category_generated.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # CSV 파일에 헤더 추가 (선택 사항)
    writer.writerow(["path", "category_id"])
    
    # result_list의 데이터를 CSV 파일에 기록
    for row in result_list:
        writer.writerow(row)

cnt=0

prefix = "/media/lasker06/T7 Shield/tvcf/kr/2024/"

for row in result_list :
    if groundtruth_dict[row[0].removeprefix(prefix)]==row[1] :
        cnt+=1

print(f"{cnt} out of 650 was correct. {len(result_list)} number of files found.")



#print(category_id)

#with open("channel_links.csv", "r") as w:
#    raw_data=csv.reader(w)
#    category_channels=[row for row in raw_data]             #category-key, channel(링크)- ,로 분리


#if category_channels[category_id] :
#    print(category_channels[category_id][0])

#else :
#    print("Not enough channel data")

#save_channel_info_json(API_KEY, random.choice(category_channels[category_id]), "recommendation_result.json")


