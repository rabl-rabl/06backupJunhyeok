from googleapiclient.discovery import build
import time

API_KEY = "AIzaSyAa4pkiThi3Z5dHhs4Q_5ifFzUMJfbHSsQ"
youtube = build("youtube", "v3", developerKey=API_KEY)

def search_korean_channels(query, max_channels=100):
    channel_ids = set()
    next_page_token = None

    while len(channel_ids) < max_channels:
        request = youtube.search().list(
            q=query,
            type="channel",
            part="snippet",
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            if any(kw in title + description for kw in ["먹방", "영화", "브이로그", "게임", "화장", "실험", "연주"]):
                channel_ids.add(item["snippet"]["channelId"])
                if len(channel_ids) >= max_channels:
                    break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(1)  # API 요청 간 딜레이

    return list(channel_ids)

# ✅ 키워드 기반으로 채널 수집
korean_keywords = ["먹방", "영화", "브이로그", "게임", "화장", "실험", "연주", "패션", "휴대폰"]
all_channel_ids = set()

# 예를 들어, 각 키워드에 대해 100개씩만 수집
for keyword in korean_keywords:
    ids = search_korean_channels(keyword, max_channels=60)  # 각 키워드마다 최대 100개
    all_channel_ids.update(ids)

print(f"총 수집된 한국 채널 수: {len(all_channel_ids)}")

# ✅ 채널 URL로 변환
channel_links = [f"https://www.youtube.com/channel/{cid}" for cid in all_channel_ids]

# ✅ 텍스트 파일로 저장
output_path = "/home/lasker06/workspace/junhyeok/channel_matching/channel_list.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for link in channel_links:
        f.write(link + "\n")

print(f"{len(channel_links)}개의 채널 링크를 '{output_path}'에 저장했습니다.")
