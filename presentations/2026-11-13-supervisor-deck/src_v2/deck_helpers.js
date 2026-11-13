// Shared helpers for the v2 deck (supervisor version).
const fs = require('fs');
const path = require('path');

const HEX = { ink: '1B1B1B', navy: '1F2A44', light: 'F2F1EE', orange: 'EB6834', blue: '2A78D6', green: '1BAF7A',
              red: 'C0392B', muted: '8A8984', grid: 'E6E5E1', ink2: '52514E', paleBlue: 'E3EDFB', paleOrange: 'FCE6DB',
              paleGreen: 'DDF3EA' };

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function make(pres) {
  const C = pres.SchemeColor;
  const H = {};
  H.txt = (slide, text, opts) => slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: 'top' }, opts));
  H.pill = (slide, x, y, label, color, w = 1.2) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.36, rectRadius: 0.18, fill: { color }, line: { color } });
    H.txt(slide, label, { x, y, w, h: 0.36, fontSize: 13, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
  };
  H.card = (slide, x, y, w, h, fill = HEX.light) =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill } });
  H.box = (slide, x, y, w, h, fill, line) =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: line || fill, width: 1.5 } });
  H.arrow = (slide, x1, y1, x2, y2, color = HEX.ink2) =>
    slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
      flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 2, endArrowType: 'triangle' } });
  H.circleNum = (slide, x, y, n, color, d = 0.5) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color } });
    H.txt(slide, String(n), { x, y, w: d, h: d, fontSize: 16, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
  };
  // bullets: items are strings or {text, options} or [runs] (array = one paragraph made of runs)
  H.bullets = (slide, items, opts) => {
    const runs = [];
    items.forEach((it, i) => {
      const last = i === items.length - 1;
      const para = Array.isArray(it) ? it : [typeof it === 'string' ? { text: it, options: {} } : it];
      para.forEach((r, j) => {
        const o = Object.assign({}, r.options || {});
        if (j === 0) { const nb = (opts && opts.noBullet) || (r.options && r.options.bullet === false); Object.assign(o, { bullet: nb ? false : { indent: 15 }, paraSpaceAfter: (opts && opts.gap) || 6 }); }
        if (j === para.length - 1 && !last) o.breakLine = true;
        runs.push({ text: r.text, options: o });
      });
    });
    const o2 = Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: 'top' }, opts);
    delete o2.noBullet; delete o2.gap;
    slide.addText(runs, o2);
  };
  H.b = (text, extra) => ({ text, options: Object.assign({ bold: true, color: HEX.navy }, extra || {}) });
  H.t = (text, extra) => ({ text, options: Object.assign({}, extra || {}) });
  // image fitted inside a box, keeping its aspect ratio
  H.img = (slide, file, x, y, w, h, align = 'center', valign = 'top') => {
    if (!fs.existsSync(file)) {
      H.card(slide, x, y, w, h, 'FDECEA');
      H.txt(slide, 'MISSING FIGURE: ' + path.basename(file), { x: x + 0.2, y: y + 0.2, w: w - 0.4, h: 0.6, color: HEX.red, fontSize: 14 });
      return { x, y, w, h };
    }
    const s = pngSize(file);
    const k = Math.min(w / s.w, h / s.h);
    const iw = s.w * k, ih = s.h * k;
    const ix = align === 'left' ? x : align === 'right' ? x + w - iw : x + (w - iw) / 2;
    const iy = valign === 'top' ? y : valign === 'bottom' ? y + h - ih : y + (h - ih) / 2;
    slide.addImage({ path: file, x: ix, y: iy, w: iw, h: ih });
    return { x: ix, y: iy, w: iw, h: ih };
  };
  return H;
}

module.exports = { HEX, make, pngSize };
