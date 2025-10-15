import requests
import random # randomモジュールをインポート
from flask import Flask, render_template
from bs4 import BeautifulSoup

app = Flask(__name__)

def scrape_osaka_tabelog():
    """食べログの大阪のレストラン情報をスクレイピングする関数"""

    # スクレイピング対象のURLを大阪に変更
    url = "https://tabelog.com/osaka/rstLst/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        restaurant_list = []
        
        # 以前と同様に店舗情報が含まれる要素を取得
        items = soup.select(".list-rst") 

        for item in items:
            # 各情報を抽出
            name = item.select_one(".list-rst__rst-name-target").text.strip()
            address = item.select_one(".list-rst__address").text.strip()
            rating = item.select_one(".c-rating__val").text.strip()
            
            try:
                price_night = item.select_one(".c-rating-v3__time--night .c-rating-v3__val").text.strip()
            except AttributeError:
                price_night = "---"
            try:
                price_day = item.select_one(".c-rating-v3__time--day .c-rating-v3__val").text.strip()
            except AttributeError:
                price_day = "---"

            reviews = item.select_one(".list-rst__rvw-count-num").text.strip()
            img_tag = item.select_one(".list-rst__img-obj .cpy-main-image")
            image_url = img_tag.get('data-original') if img_tag else "取得できませんでした"

            restaurant_list.append({
                "name": name,
                "address": address,
                "rating": rating,
                "price": f"夜: {price_night} / 昼: {price_day}",
                "reviews": reviews,
                "image_url": image_url
            })
            
        return restaurant_list

    except requests.exceptions.RequestException as e:
        print(f"リクエストエラー: {e}")
        return []
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []


@app.route("/")
def index():
    """トップページを表示する関数"""
    restaurant_list = scrape_osaka_tabelog()
    
    # 取得したリストからランダムに1件選ぶ
    random_restaurant = None
    if restaurant_list: # リストが空でないことを確認
        random_restaurant = random.choice(restaurant_list)
        
    return render_template("index.html", restaurant=random_restaurant) # 単一の店舗情報を渡す


if __name__ == "__main__":
    app.run(debug=True)