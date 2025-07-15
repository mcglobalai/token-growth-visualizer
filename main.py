import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, HPacker
from dotenv import load_dotenv
from tqdm import tqdm
import os
import time
import functools
from datetime import datetime, timedelta
import numpy as np

# ====== Configurable parameters ======
DAYS_AGO = 365  # Number of days to look back for price comparison
NUM_TOKENS = 250  # Number of tokens to fetch
MAX_REQUESTS_PER_MINUTE = 60
EXCLUDE_WRAP_TOKENS = {
    'WBTC', 'CBTC', 'CBBTC', 'EBTC', 'XSOLVBTC', 'TBTC', 'LBTC', 'BTC.B', 'CLBTC', 'TETH',
    'STETH', 'WSTETH', 'LSETH', 'OSETH', 'ETHX', 'CBETH', 'RETH', 'METH', 'CMETH', 'SWETH',
    'WETH', 'WEETH', 'RSETH', 'EZETH', 'SUPEROETH', 'CGETH.HASHKEY', 'ETH+', 'FRXETH', 'EETH', 'SOLVBTC'
    'STHYPE', 'WHYPE'
}
# ====================================

# ✅ Load API KEY
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("YOUR_API_KEY", "")

# ✅ Auto switch BASE_URL
if api_key:
    BASE_URL = "https://pro-api.coingecko.com/api/v3"
else:
    BASE_URL = "https://api.coingecko.com/api/v3"

def retry(max_retries=3, wait_seconds=5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}, retrying {attempt+1}/{max_retries} ...")
                    time.sleep(wait_seconds)
            print("Max retries reached, skipping.")
            return None
        return wrapper
    return decorator

@retry(max_retries=5, wait_seconds=10)
def cg_get(path, params=None):
    headers = {}
    if api_key:
        headers['X-CG-Pro-API-Key'] = api_key
    url = f"{BASE_URL}{path}"
    response = requests.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response

# ✅ Get top N tokens by market cap
def get_top_tokens(n=100):
    path = "/coins/markets"
    per_page = 250  # CoinGecko max per request
    tokens = []
    total_pages = (n + per_page - 1) // per_page
    for page in range(1, total_pages + 1):
        fetch_num = min(per_page, n - len(tokens))
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': fetch_num,
            'page': page,
            'sparkline': False
        }
        resp = cg_get(path, params=params)
        if resp is None:
            continue
        data = resp.json()
        if not data:
            break
        tokens.extend(data)
        if len(tokens) >= n:
            break
    return tokens[:n]

def get_price_on_date(coin_id, date_str):
    path = f"/coins/{coin_id}/history"
    params = {'date': date_str}
    resp = cg_get(path, params=params)
    if resp is None:
        return None
    data = resp.json()
    try:
        return data['market_data']['current_price']['usd']
    except Exception:
        return None

def get_first_available_price(coin_id):
    path = f"/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': 'max'}
    resp = cg_get(path, params=params)
    if resp is None:
        return None, None
    data = resp.json()
    if 'prices' in data and data['prices']:
        ts, price = data['prices'][0]
        date_str = datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d')
        return price, date_str
    return None, None

# ✅ Rate limit config
request_interval = 60 / MAX_REQUESTS_PER_MINUTE

# ✅ Fetch data with progress bar and rate limit
top_tokens = get_top_tokens(NUM_TOKENS)
top_tokens = [t for t in top_tokens if t['symbol'].upper() not in EXCLUDE_WRAP_TOKENS]

# Get current prices for all tokens in one request
symbols = [t['symbol'].upper() for t in top_tokens]
ids = [t['id'] for t in top_tokens]
# Use /coins/markets to get current prices
current_prices = {}
for i in range(0, len(ids), 100):
    batch_ids = ids[i:i+100]
    params = {
        'vs_currency': 'usd',
        'ids': ','.join(batch_ids),
        'order': 'market_cap_desc',
        'per_page': len(batch_ids),
        'page': 1,
        'sparkline': False
    }
    resp = cg_get("/coins/markets", params=params)
    if resp is None:
        continue
    for coin in resp.json():
        current_prices[coin['id']] = coin['current_price']

result = []
btc_growth = None
skipped_tokens = []
tokens_used_first_price = []

# Calculate the date string for DAYS_AGO days ago
price_date_str = (datetime.utcnow() - timedelta(days=DAYS_AGO)).strftime('%d-%m-%Y')

with tqdm(top_tokens, desc=f"Fetching {DAYS_AGO}d ago price history") as pbar:
    for coin in pbar:
        try:
            cid = coin['id']
            old_price = get_price_on_date(cid, price_date_str)
            used_first_price = False
            first_price_date = None
            if old_price is None:
                old_price, first_price_date = get_first_available_price(cid)
                if old_price is not None:
                    used_first_price = True
                    tokens_used_first_price.append({'symbol': coin['symbol'].upper(), 'id': cid, 'date': first_price_date})
            time.sleep(request_interval)  # Rate limit
            cur_price = current_prices.get(cid)
            if old_price and cur_price:
                change = (cur_price - old_price) / old_price * 100
                result.append({
                    'symbol': coin['symbol'].upper(),
                    'change_pct': change,
                    'logo': coin['image'],
                    'id': cid
                })
                if cid == 'bitcoin':
                    btc_growth = change
            else:
                skipped_tokens.append(cid)
        except Exception as e:
            print(f"Error processing token {coin.get('id', '')}: {e}")
            skipped_tokens.append(coin.get('id', ''))
            continue
        pbar.set_postfix(skipped=len(skipped_tokens))

print(f"Total tokens skipped (no price data): {len(skipped_tokens)}")
if skipped_tokens:
    print("Skipped tokens (symbol, id):")
    for coin in top_tokens:
        if coin['id'] in skipped_tokens:
            print(f"  {coin['symbol'].upper()} ({coin['id']})")

if tokens_used_first_price:
    print(f"Tokens using first available price instead of {DAYS_AGO}d ago:")
    for t in tokens_used_first_price:
        print(f"  {t['symbol']} ({t['id']}), first price date: {t['date']}")

# ====== Build dataframe and filter ======
df = pd.DataFrame(result)
df = df[df['change_pct'] >= btc_growth]
df.sort_values("change_pct", ascending=True, inplace=True)

# ====== Nonlinear (log) scaling for bar length ======
# Make bar lengths more visually even
min_val = df['change_pct'].min()
df['bar_length'] = np.log1p(df['change_pct'] - min_val + 1)
df['bar_length'] = df['bar_length'] / df['bar_length'].max() * df['change_pct'].max()

# ====== Bar color: BTC orange, others green ======
colors = ['#F7931A' if symbol == 'BTC' else '#3CB371' for symbol in df['symbol']]

# ====== Plotting: compact, not full width, BTC special color ======
fig, ax = plt.subplots(figsize=(13, len(df) * 0.38))
bars = ax.barh(df['symbol'], df['bar_length'], color=colors, height=0.32)

# Set bar max length, leave space on the right
ax.set_xlim(0, df['bar_length'].max() * 1.1)

# Compact layout, reduce margins
plt.subplots_adjust(top=0.99, bottom=0.01, left=0.18, right=0.98)
ax.axis('off')

# ====== Add annotation: percent, logo, symbol ======
for bar, pct, logo_url, symbol in zip(bars, df['change_pct'], df['logo'], df['symbol']):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    pct_txt = TextArea(f" ▲ {pct:.2f}%", textprops=dict(color='green', fontsize=11, weight='bold'))
    try:
        r = requests.get(logo_url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGBA")  # Force RGBA
        img = img.resize((16, 16))  # Smaller size
        icon = OffsetImage(img, zoom=0.8)
    except (UnidentifiedImageError, Exception) as e:
        print(f"Warning: Failed to load logo for {symbol} ({logo_url}): {e}")
        icon = OffsetImage(Image.new("RGBA", (16, 16), (255, 255, 255, 0)))
    symbol_txt = TextArea(symbol, textprops=dict(fontsize=11, weight='bold'))
    pack = HPacker(children=[pct_txt, icon, symbol_txt], align="center", pad=1.2, sep=3)
    ab = AnnotationBbox(pack, (x + 1.5, y), frameon=False, box_alignment=(0, 0.5))
    ax.add_artist(ab)

plt.tight_layout()
plt.savefig("./output/token_growth.png", dpi=300, bbox_inches='tight')
# plt.show()

# ====== Save data to csv ======
pd.DataFrame(top_tokens).to_csv("./output/top_tokens.csv", index=False)
df.to_csv("./output/token_growth_data.csv", index=False)
