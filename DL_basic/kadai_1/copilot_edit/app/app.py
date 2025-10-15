import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template

app = Flask(__name__)

def scrape_tabelog():
    """食べログの東京レストランランキングページから情報をスクレイピングする"""
    # スクレイピング対象のURL
    url = "https://tabelog.com/tokyo/rstLst/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる
        soup = BeautifulSoup(response.text, 'html.parser')

        restaurants = []
        # レストラン一覧の要素を取得
        rst_list = soup.find_all('div', class_='list-rst')

        for rst in rst_list:
            # 各情報を取得（要素が存在しない場合も考慮）
            name = rst.find('a', class_='list-rst__rst-name-target').text.strip() if rst.find('a', class_='list-rst__rst-name-target') else 'N/A'
            address = rst.find('div', class_='list-rst__address').text.strip() if rst.find('div', class_='list-rst__address') else 'N/A'
            rating = rst.find('span', class_='c-rating__val').text.strip() if rst.find('span', class_='c-rating__val') else 'N/A'
            
            price_ranges = rst.find_all('span', class_='c-rating__val--strong')
            price = f"夜: {price_ranges[0].text.strip() if len(price_ranges) > 0 else 'N/A'} / 昼: {price_ranges[1].text.strip() if len(price_ranges) > 1 else 'N/A'}"

            review_count = rst.find('em', class_='list-rst__rvw-count-num').text.strip() if rst.find('em', class_='list-rst__rvw-count-num') else 'N/A'
            photo = rst.find('img', class_='c-img--frame').get('data-src') if rst.find('img', class_='c-img--frame') else ''

            restaurants.append({
                'name': name,
                'address': address,
                'rating': rating,
                'price': price,
                'review_count': review_count,
                'photo': photo
            })
        
        return restaurants

    except requests.exceptions.RequestException as e:
        print(f"Error during requests: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


@app.route('/')
def index():
    """トップページ。スクレイピング結果を表示する"""
    restaurants_data = scrape_tabelog()
    return render_template('index.html', restaurants=restaurants_data)

if __name__ == '__main__':
    # http://127.0.0.1:5000 でサーバーを起動
    app.run(debug=True)