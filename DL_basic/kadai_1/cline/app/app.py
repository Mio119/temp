from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/')
def index():
    # 食べログのURL
    url = 'https://tabelog.com/tokyo/rstLst/'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        restaurants = []
        # 店舗リストを取得
        restaurant_list = soup.find_all('div', class_='list-rst')
        
        for item in restaurant_list[:5]: # 上位5件を取得
            name = item.find('a', class_='list-rst__rst-name-target').text.strip()
            address = item.find('div', class_='list-rst__address').text.strip()
            rating = item.find('span', class_='c-rating__val').text.strip()
            price = item.find('span', class_='c-rating__val--strong').text.strip()
            reviews = item.find('a', class_='list-rst__rvw-count-target').text.strip()
            photo = item.find('img', class_='c-img--frame')['src']
            
            restaurants.append({
                'name': name,
                'address': address,
                'rating': rating,
                'price': price,
                'reviews': reviews,
                'photo': photo
            })
            
        return render_template('index.html', restaurants=restaurants)
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
