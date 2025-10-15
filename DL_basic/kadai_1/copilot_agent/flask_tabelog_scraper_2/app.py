import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template
import logging
import random

# ログ設定
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

def scrape_tabelog():
    """
    食べログのレストラン情報をスクレイピングする関数
    """
    # 食べログのURL（例：大阪のレストランランキング）
    # 注: 食べログの利用規約を遵守してください。
    # スクレイピングを頻繁に行うとアクセスがブロックされる可能性があります。
    url = "https://tabelog.com/osaka/rstLst/ranking/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # HTTPエラーの場合に例外を発生させる
        soup = BeautifulSoup(response.content, 'lxml')

        restaurants = []
        # 食べログのHTML構造は変更される可能性があるため、セレクタは適宜調整が必要です
        for item in soup.select('.list-rst'):
            try:
                name = item.select_one('.list-rst__rst-name-target').text.strip()
                # 住所、評価などのセレクタは実際のHTMLに合わせてください
                # 以下はダミーのセレクタです
                address = item.select_one('.list-rst__address').text.strip() if item.select_one('.list-rst__address') else "情報なし"
                rating = item.select_one('.c-rating__val').text.strip() if item.select_one('.c-rating__val') else "評価なし"
                price_dinner = item.select_one('.c-rating__val.c-rating__val--dinner').text.strip() if item.select_one('.c-rating__val.c-rating__val--dinner') else "情報なし"
                price_lunch = item.select_one('.c-rating__val.c-rating__val--lunch').text.strip() if item.select_one('.c-rating__val.c-rating__val--lunch') else "情報なし"
                reviews = item.select_one('.list-rst__rvw-count-num').text.strip() if item.select_one('.list-rst__rvw-count-num') else "0"
                photo_url = item.select_one('.list-rst__img-obj img')['data-src'] if item.select_one('.list-rst__img-obj img') else ""


                restaurants.append({
                    'name': name,
                    'address': address,
                    'rating': rating,
                    'price': f"夜: {price_dinner} 昼: {price_lunch}",
                    'reviews': reviews,
                    'photo_url': photo_url
                })
            except (AttributeError, KeyError) as e:
                logging.warning(f"一部の情報の取得に失敗しました: {e}")
                continue
        
        if not restaurants:
            logging.warning("レストラン情報が取得できませんでした。HTMLの構造が変更された可能性があります。")

        return restaurants

    except requests.exceptions.RequestException as e:
        logging.error(f"Webページへのアクセスに失敗しました: {e}")
        return []
    except Exception as e:
        logging.error(f"スクレイピング中に予期せぬエラーが発生しました: {e}")
        return []

@app.route('/')
def index():
    """
    トップページ。スクレイピング結果をランダムに1件表示する。
    """
    restaurants = scrape_tabelog()
    restaurant = random.choice(restaurants) if restaurants else None
    return render_template('index.html', restaurant=restaurant)

if __name__ == '__main__':
    # 外部からアクセス可能にする場合は host='0.0.0.0' を指定
    app.run(debug=True, port=5001)
