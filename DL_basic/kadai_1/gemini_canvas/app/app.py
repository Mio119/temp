# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template

# Flaskアプリケーションのインスタンスを作成
app = Flask(__name__)

def scrape_tabelog():
    """
    食べログの新宿エリアのレストラン一覧ページから情報をスクレイピングする関数
    """
    # スクレイピング対象のURL
    # URLは変更される可能性があるため、適宜確認してください
    url = "https://tabelog.com/tokyo/A1304/A130401/rstLst/"
    
    # User-Agentを設定しないとアクセスがブロックされることがある
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Webページを取得
        response = requests.get(url, headers=headers)
        response.raise_for_status() # ステータスコードが200以外の場合に例外を発生させる

        # BeautifulSoupオブジェクトを作成し、HTMLをパース
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # レストラン情報が格納されている要素をすべて取得
        # サイトの構造変更により、以下のクラス名は使えなくなる可能性があります
        restaurant_list = soup.find_all('div', class_='list-rst')
        
        restaurants_data = []
        
        # 各レストラン情報から必要なデータを抽出
        for item in restaurant_list:
            data = {}
            # try-exceptで各要素の取得を試み、失敗した場合はNoneを設定
            try:
                # 店名
                data['name'] = item.find('a', class_='list-rst__rst-name-target').text.strip()
            except AttributeError:
                data['name'] = 'N/A'
                
            try:
                # 住所
                data['address'] = item.find(class_='list-rst__address').text.strip()
            except AttributeError:
                data['address'] = 'N/A'
                
            try:
                # 評価
                data['rating'] = item.find('span', class_='c-rating__val').text.strip()
            except AttributeError:
                data['rating'] = '評価なし'
            
            try:
                # 口コミ件数
                data['reviews'] = item.find('em', class_='list-rst__rvw-count-num').text.strip()
            except AttributeError:
                data['reviews'] = '0'

            try:
                # 価格帯（夜）
                # 属性(rel)も指定して正確に要素を特定
                price_dinner_tag = item.find('span', class_='c-rating__val', rel='dinner')
                data['price_dinner'] = price_dinner_tag.text.strip() if price_dinner_tag else '-'
            except AttributeError:
                data['price_dinner'] = '-'
            
            try:
                # 価格帯（昼）
                price_lunch_tag = item.find('span', class_='c-rating__val', rel='lunch')
                data['price_lunch'] = price_lunch_tag.text.strip() if price_lunch_tag else '-'
            except AttributeError:
                data['price_lunch'] = '-'

            try:
                # 料理写真
                # `img`タグの`src`属性から画像のURLを取得
                data['image'] = item.find('div', class_='list-rst__frame-img').find('img')['src']
            except (AttributeError, TypeError):
                data['image'] = 'https://placehold.co/200x200/eee/ccc?text=No+Image' # 代替画像

            restaurants_data.append(data)
            
        return restaurants_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# ルートURL ('/') にアクセスがあった場合に実行される関数
@app.route('/')
def index():
    # スクレイピング関数を呼び出してデータを取得
    restaurants = scrape_tabelog()
    # 取得したデータをHTMLテンプレートに渡してレンダリング
    return render_template('index.html', restaurants=restaurants)

# このスクリプトが直接実行された場合にサーバーを起動
if __name__ == '__main__':
    # デバッグモードを有効にし、ホスト0.0.0.0、ポート5001で実行
    # ブラウザからは http://127.0.0.1:5001/ でアクセス
    app.run(debug=True, host='0.0.0.0', port=5001)
