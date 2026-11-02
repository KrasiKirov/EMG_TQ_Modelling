"""
Render every slide of presentation/deck.html to a PNG with headless Chrome and make contact
sheets (presentation/previews/sheet_<k>.png) for checking the layout.

    python EXPERIMENTS/code/deck_previews.py [first last]
"""
import os
import re
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
PRES = os.path.join(os.path.dirname(HERE), 'presentation')
OUT = os.path.join(PRES, 'previews')
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'


def shot(i):
    png = os.path.join(OUT, f'slide_{i:02d}.png')
    url = 'file:///' + os.path.abspath(os.path.join(PRES, 'deck.html')).replace(os.sep, '/') + f'#s{i}'
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--window-size=1280,720',
                    f'--screenshot={png}', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return png


def main():
    os.makedirs(OUT, exist_ok=True)
    n = len(re.findall(r'<section class="slide', open(os.path.join(PRES, 'deck.html'), encoding='utf-8').read()))
    a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1, n)
    with ThreadPoolExecutor(4) as ex:
        pngs = list(ex.map(shot, range(a, b + 1)))
    for k in range(0, len(pngs), 9):
        fig, axs = plt.subplots(3, 3, figsize=(24, 13.6))
        for ax, p in zip(axs.flat, pngs[k:k + 9]):
            ax.imshow(mpimg.imread(p))
            ax.set_title(os.path.basename(p), fontsize=10)
        for ax in axs.flat:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'sheet_{a + k:02d}.png'), dpi=80)
        plt.close(fig)
    print(n, 'slides; previews in', OUT)


if __name__ == '__main__':
    main()
