const { chromium } = require('playwright');
const path = require('path');
const EXE = 'C:\\Users\\ufo00\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe';
const URL = 'file:///' + path.join(__dirname, '..', 'outputs', 'AI算法大纲-交互版.html').replace(/\\/g, '/');
const OUT = process.argv[2] || path.join(process.env.TEMP || '.', 'demo_shots');
const fs = require('fs');
if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });

const PICKS = process.argv.slice(3);

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1180, height: 1000 }, deviceScaleFactor: 1.5 });
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForTimeout(300);

  const names = await page.evaluate(() => DATA.map(a => a.n).filter(n => !!DEMOS[n]));
  const targets = PICKS.length ? PICKS : names;

  for (const nm of targets) {
    const gi = await page.evaluate(n => DATA.findIndex(a => a.n === n), nm);
    if (gi < 0) { console.log('跳过（未找到）: ' + nm); continue; }
    await page.evaluate(i => { PICK.open = {}; PICK.open[i] = true; render('all', ''); }, gi);
    await page.waitForTimeout(420);
    const box = await page.evaluate(i => {
      const node = document.querySelectorAll('.node')[i];
      if (!node) return null;
      node.scrollIntoView({ block: 'start' });
      const r = node.getBoundingClientRect();
      return { x: r.x, y: r.y, w: r.width, h: r.height };
    }, gi);
    if (!box) { console.log('跳过（无节点）: ' + nm); continue; }
    await page.waitForTimeout(150);
    const file = path.join(OUT, nm.replace(/[\\/:*?"<>|]/g, '_') + '.png');
    await page.screenshot({
      path: file,
      clip: {
        x: Math.max(0, box.x - 6), y: Math.max(0, box.y - 6),
        width: Math.min(box.w + 12, 1180), height: Math.min(box.h + 12, 2400)
      }
    });
    console.log('截图 ' + nm);
  }
  console.log(errs.length ? '运行时错误: ' + errs.join(' | ') : '无运行时错误');
  console.log('输出目录: ' + OUT);
  await browser.close();
})();
