const { chromium } = require('playwright');
const path = require('path');
const EXE = 'C:\\Users\\ufo00\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe';
const URL = 'file:///' + path.join(__dirname, '..', 'outputs', 'AI算法大纲-交互版.html').replace(/\\/g, '/');
const ONLY = process.argv.slice(2);            // 只测指定算法名

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
  console.log('待验证演示：' + targets.length + ' 个');

  for (const nm of targets) {
    const gi = await page.evaluate(n => DATA.findIndex(a => a.n === n), nm);
    // 展开该条目（直接改 PICK.open 后重渲染更稳）
    await page.evaluate(i => {
      PICK.open[i] = true;
      render('all', '');
    }, gi);
    await page.waitForTimeout(260);

    const r = await page.evaluate(i => {
      const node = document.querySelectorAll('.node')[i];
      if (!node) return { ok: false, why: '找不到节点' };
      const zone = node.querySelector('.demozone');
      if (!zone) return { ok: false, why: '没有演示区' };
      if (zone.querySelector('.minit') && /加载失败/.test(zone.textContent)) {
        return { ok: false, why: '加载失败：' + zone.textContent.trim().slice(0, 90) };
      }
      const cvs = Array.from(zone.querySelectorAll('canvas'));
      if (!cvs.length) return { ok: false, why: '没有画布' };
      let painted = 0, detail = [];
      cvs.forEach(cv => {
        const c = cv.getContext('2d');
        const d = c.getImageData(0, 0, cv.width, cv.height).data;
        let nonzero = 0;
        for (let p = 3; p < d.length; p += 4 * 37) if (d[p] > 8) nonzero++;
        if (nonzero > 12) painted++;
        detail.push(cv.width + 'x' + cv.height + ':' + nonzero);
      });
      const kpis = Array.from(zone.querySelectorAll('.kpi b')).map(b => b.textContent.trim());
      const bad = kpis.filter(v => /NaN|undefined|Infinity|—/.test(v));
      return {
        ok: painted === cvs.length && !bad.length,
        painted, total: cvs.length, detail: detail.join(' '),
        kpi: kpis.join(' | '), bad: bad.join(','),
        note: (zone.querySelector('.dnote') || {}).textContent
      };
    }, gi);

    const flag = r.ok ? 'PASS' : 'FAIL';
    console.log(flag + '  ' + nm + (r.ok
      ? '  画布 ' + r.painted + '/' + r.total + '  KPI[' + r.kpi + ']'
      : '  → ' + (r.why || ('未绘制 ' + r.painted + '/' + r.total + ' (' + r.detail + ')  异常KPI=' + r.bad))));
    if (!r.ok && r.note) console.log('      note: ' + r.note.slice(0, 120));
  }

  console.log(errs.length ? '\n运行时错误 ' + errs.length + ' 条：\n' + errs.slice(0, 12).join('\n') : '\n无运行时错误');
  await browser.close();
  process.exit(errs.length ? 2 : 0);
})();
