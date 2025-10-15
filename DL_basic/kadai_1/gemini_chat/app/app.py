import requests
from flask import Flask, render_template
from bs4 import BeautifulSoup

app = Flask(__name__)

def scrape_tabelog():
    """食べログのレストラン情報をスクレイピングする関数"""

    # スクレイピング対象のURL (例: 東京のレストランランキング)
    # 注意: URLは変更される可能性があります。
    url = "https://tabelog.com/tokyo/rstLst/"
    
    # 偽のユーザーエージェントを設定してアクセスを試みる
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # ステータスコードが200以外の場合に例外を発生させる
        soup = BeautifulSoup(response.content, "html.parser")

        restaurant_list = []
        
        # 店舗情報が含まれる要素をすべて取得
        # 注意: 食べログのサイト構成が変更されると、このセレクタは機能しなくなる可能性があります。
        items = soup.select(".list-rst") 

        for item in items:
            # 各情報を抽出
            name = item.select_one(".list-rst__rst-name-target").text.strip()
            address = item.select_one(".list-rst__address").text.strip()
            rating = item.select_one(".c-rating__val").text.strip()
            
            # 価格帯（夜・昼）を取得。存在しない場合もあるためtry-exceptで処理
            try:
                price_night = item.select_one(".c-rating-v3__time--night .c-rating-v3__val").text.strip()
            except AttributeError:
                price_night = "---"
            try:
                price_day = item.select_one(".c-rating-v3__time--day .c-rating-v3__val").text.strip()
            except AttributeError:
                price_day = "---"

            reviews = item.select_one(".list-rst__rvw-count-num").text.strip()
            
            # 写真のURLを取得
            # 'data-original'属性に遅延読み込み用の画像URLが格納されている場合がある
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
    restaurants = scrape_tabelog()
    return render_template("index.html", restaurants=restaurants)


if __name__ == "__main__":
    app.run(debug=True)