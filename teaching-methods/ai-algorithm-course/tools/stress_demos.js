const { chromium } = require('playwright');
const path = require('path');
const EXE = 'C:\\Users\\ufo00\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe';
const URL = 'file:///' + path.join(__dirname, '..', 'outputs', 'AI算法大纲-交互版.html').replace(/\\/g, '/');
const ONLY = process.argv.slice(2);

/* 把每个演示的所有滑块推到 min / 中 / max，点一遍所有按钮，检查 KPI 与画布是否仍然正常 */
(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errs = [];
  page.on('pageerror', e => errs.push('[pageerror] ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('[console] ' + m.text()); });
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForTimeout(400);

  const names = await page.evaluate(() => DATA.map(a => a.n).filter(n => !!DEMOS[n]));
  const targets = ONLY.length ? names.filter(n => ONLY.some(o => n.indexOf(o) >= 0)) : names;
  let fail = 0;

  for (const nm of targets) {
    const gi = await page.evaluate(n => DATA.findIndex(a => a.n === n), nm);
    await page.evaluate(i => { PICK.open[i] = true; render('all', ''); }, gi);
    await page.waitForTimeout(200);

    const probe = () => page.evaluate(i => {
      const node = document.querySelectorAll('.node')[i];
      const zone = node.querySelector('.demozone');
      if (!zone) return { bad: ['无演示区'], kpi: '' };
      const kpis = Array.from(zone.querySelectorAll('.kpi b')).map(b => b.textContent.trim());
      const bad = kpis.filter(v => /NaN|undefined|Infinity|null/.test(v));
      const note = (zone.querySelector('.dnote') || {}).textContent || '';
      if (/NaN|undefined|Infinity/.test(note)) bad.push('note 异常');
      let blank = 0;
      zone.querySelectorAll('canvas').forEach(cv => {
        const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
        let nz = 0;
        for (let p = 3; p < d.length; p += 4 * 41) if (d[p] > 8) nz++;
        if (nz <= 8) blank++;
      });
      if (blank) bad.push(blank + ' 张画布空白');
      return { bad, kpi: kpis.join('|') };
    }, gi);

    const probs = [];
    let r = await probe();
    if (r.bad.length) probs.push('初始: ' + r.bad.join(','));

    const sliders = await page.evaluate(i => {
      const node = document.querySelectorAll('.node')[i];
      return Array.from(node.querySelectorAll('input[type=range]')).map(el => ({
        id: el.id, min: el.min, max: el.max
      }));
    }, gi);

    for (const s of sliders) {
      for (const v of [s.min, s.max, String((parseFloat(s.min) + parseFloat(s.max)) / 2)]) {
        await page.evaluate(([id, val]) => {
          const el = document.getElementById(id);
          if (!el) return;
          el.value = val;
          el.dispatchEvent(new Event('input'));
          el.dispatchEvent(new Event('change'));
        }, [s.id, v]);
        await page.waitForTimeout(60);
        r = await probe();
        if (r.bad.length) probs.push(s.id + '=' + v + ': ' + r.bad.join(','));
      }
    }

    const btns = await page.evaluate(i => {
      const node = document.querySelectorAll('.node')[i];
      return Array.from(node.querySelectorAll('.demozone button')).map(b => b.id);
    }, gi);
    for (const b of btns) {
      await page.evaluate(id => { const el = document.getElementById(id); if (el) el.click(); }, b);
      await page.waitForTimeout(80);
      r = await probe();
      if (r.bad.length) probs.push(b + ' 点击后: ' + r.bad.join(','));
    }

    if (probs.length) { fail++; console.log('FAIL  ' + nm + '\n      ' + probs.slice(0, 4).join('\n      ')); }
    else console.log('PASS  ' + nm + '  (' + sliders.length + ' 滑块 / ' + btns.length + ' 按钮)');
  }

  console.log('\n' + (fail ? fail + ' 个演示存在问题' : '全部 ' + targets.length + ' 个演示通过压力测试'));
  if (errs.length) console.log('运行时错误：\n' + errs.slice(0, 10).join('\n'));
  await browser.close();
  process.exit(fail || errs.length ? 1 : 0);
})();
