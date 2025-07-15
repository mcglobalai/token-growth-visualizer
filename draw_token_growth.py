import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, HPacker
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import numpy as np

# Read data
csv_file = './output/token_growth_data.csv'
df = pd.read_csv(csv_file)

# Sort and prepare data
if 'change_pct' not in df.columns or 'symbol' not in df.columns:
    raise ValueError('CSV must contain columns: change_pct, symbol, logo')
df.sort_values('change_pct', ascending=True, inplace=True)

# Nonlinear (log) scaling for bar length
min_val = df['change_pct'].min()
df['bar_length'] = np.log1p(df['change_pct'] - min_val + 1)
df['bar_length'] = df['bar_length'] / df['bar_length'].max() * df['change_pct'].max()

# Bar color: BTC orange, others green
colors = ['#F7931A' if symbol == 'BTC' else '#3CB371' for symbol in df['symbol']]

# Plotting
fig, ax = plt.subplots(figsize=(13, len(df) * 0.38))
bars = ax.barh(df['symbol'], df['bar_length'], color=colors, height=0.32)

# Set bar max length, leave space on the right
ax.set_xlim(0, df['bar_length'].max() * 1.1)

# Compact layout, reduce margins
plt.subplots_adjust(top=0.99, bottom=0.01, left=0.18, right=0.98)
ax.axis('off')

# Add annotation: percent, logo, symbol
for bar, pct, logo_url, symbol in zip(bars, df['change_pct'], df['logo'], df['symbol']):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    pct_txt = TextArea(f" ▲ {pct:.2f}%", textprops=dict(color='green', fontsize=11, weight='bold'))
    try:
        r = requests.get(logo_url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        img = img.resize((16, 16))
        icon = OffsetImage(img, zoom=0.8)
    except (UnidentifiedImageError, Exception) as e:
        print(f"Warning: Failed to load logo for {symbol} ({logo_url}): {e}")
        icon = OffsetImage(Image.new("RGBA", (16, 16), (255, 255, 255, 0)))
    symbol_txt = TextArea(symbol, textprops=dict(fontsize=11, weight='bold'))
    pack = HPacker(children=[pct_txt, icon, symbol_txt], align="center", pad=1.2, sep=3)
    ab = AnnotationBbox(pack, (x + 1.5, y), frameon=False, box_alignment=(0, 0.5))
    ax.add_artist(ab)

plt.tight_layout()
plt.savefig("./output/token_growth_from_csv.png", dpi=300, bbox_inches='tight')
# plt.show() 