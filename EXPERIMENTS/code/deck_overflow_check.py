"""
Report slides of presentation/deck.html whose content runs into the footer or past the slide
(rendered in headless Chrome at 1280 x 720).

    python EXPERIMENTS/code/deck_overflow_check.py        -> e.g. "overflowing slides: none" or "2: 151 px, 13: 19 px"
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PRES = os.path.join(os.path.dirname(HERE), 'presentation')
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
JS = ('<script>window.addEventListener("load",function(){var out=[];document.querySelectorAll(".slide").forEach(function(sl,i){'
      'var foot=sl.querySelector(".foot");var limit=sl.getBoundingClientRect().top+sl.clientHeight-(foot?36:0);var mx=0;'
      'sl.querySelectorAll("*").forEach(function(e){if(e.closest(".foot"))return;var r=e.getBoundingClientRect();if(r.height>0&&r.bottom>mx)mx=r.bottom;});'
      'var over=mx-limit;if(over>2)out.push((i+1)+": "+Math.round(over)+" px");});'
      'var d=document.createElement("pre");d.textContent="OVERFLOW{"+out.join(", ")+"}";document.body.appendChild(d);});</script>')


def main():
    src = open(os.path.join(PRES, 'deck.html'), encoding='utf-8').read()
    tmp = os.path.join(PRES, '_overflow_check.html')
    open(tmp, 'w', encoding='utf-8').write(src.replace('</body>', JS + '</body>'))
    try:
        dom = subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--window-size=1280,720', '--virtual-time-budget=15000',
                              '--dump-dom', 'file:///' + os.path.abspath(tmp).replace(os.sep, '/')],
                             capture_output=True, text=True, encoding='utf-8', timeout=180).stdout
    finally:
        os.remove(tmp)
    m = re.findall(r'OVERFLOW\{([^}]*)\}', dom)
    print('overflowing slides:', (m[-1] or 'none') if m else 'check failed')


if __name__ == '__main__':
    main()
