import dotenv
dotenv.load_dotenv()
import os
import requests

token = os.getenv('METACULUS_TOKEN')
headers = {"Authorization": f"Token {token}"}

# Get MiniBench open questions
resp = requests.get('https://www.metaculus.com/api/posts/', params={
    'tournaments': 'minibench',
    'status': 'open',
    'limit': 10
}, headers=headers)

results = resp.json().get('results', [])
print(f"Open MiniBench questions: {len(results)}\n")

for q in results[:10]:
    title = q.get('title', 'No title')[:55]
    url = f"https://www.metaculus.com/questions/{q.get('id')}/"
    close_time = q.get('scheduled_close_time', 'N/A')[:16] if q.get('scheduled_close_time') else 'N/A'
    print(f"{title}")
    print(f"  URL: {url}")
    print(f"  Closes: {close_time}\n")
