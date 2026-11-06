/**
 * 合并产物 outputs/AI课程平台.html 的 Playwright 验收。
 *
 * 运行（必须带 NODE_PATH）:
 *   NODE_PATH="C:/Users/ufo00/.workbuddy/binaries/node/workspace/node_modules" \
 *   "C:/Users/ufo00/.workbuddy/binaries/node/versions/22.22.2-6/node.exe" tools/verify_merged.js
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const EXE = 'C:\\Users\\ufo00\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe';
const FILE = 'file:///' + path.join(
  'D:/Agent/teaching-works-lab/courses/AI_Application', 'outputs', 'AI课程平台.html').replace(/\\/g, '/');
const SHOTS = path.join('D:/Agent/teaching-works-lab/courses/AI_Application', 'outputs', '_shots');

let PASS = 0, FAIL = 0;
const errors = [];
function ok(name, cond, extra) {
  if (cond) { PASS++; console.log('PASS  ' + name + (extra ? '   ' + extra : '')); }
  else { FAIL++; console.log('FAIL  ' + name + (extra ? '   ' + extra : '')); }
}

(async () => {
  if (!fs.existsSync(SHOTS)) fs.mkdirSync(SHOTS, { recursive: true });
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console.error: ' + m.text()); });

  await page.goto(FILE, { waitUntil: 'load' });
  await page.waitForTimeout(1200);

  // ---------- 1. 零报错 ----------
  ok('1 页面零 pageerror / console.error', errors.length === 0, errors.slice(0, 3).join(' | '));

  // ---------- 2. 一级导航：4 个模块 ----------
  const mods = ['algo', 'pick', 'book', 'coef'];
  for (const m of mods) {
    await page.click(`#topnav button[data-m="${m}"]`);
    await page.waitForTimeout(250);
    const vis = await page.$$eval('.pg', els => els.filter(e => !e.hidden).map(e => e.id));
    const onBtn = await page.$$eval('#topnav button[data-m]',
      els => els.filter(e => e.classList.contains('on')).map(e => e.dataset.m));
    ok(`2 模块 ${m} 切换`, vis.length === 1 && vis[0] === 'pg-' + m &&
      onBtn.length === 1 && onBtn[0] === m, '可见=' + vis.join(','));
  }

  // ---------- 3. 二级导航：主页面 7 个标签 ----------
  await page.click('#topnav button[data-m="algo"]');
  await page.waitForTimeout(200);
  const panes = ['map', 'svd', 'pca', 'lab', 'book', 'plan', 'stu'];
  for (const p of panes) {
    await page.click(`#pg-algo header nav button[data-p="${p}"]`);
    await page.waitForTimeout(150);
    const g = await page.$$eval('.pane.on', els => els.map(e => e.id));
    const inside = await page.$$eval('#pg-algo .pane.on', els => els.length);
    if (!(g.length === 1 && g[0] === 'p-' + p && inside === 1)) {
      ok(`3 二级标签 ${p}`, false, '全局 on=' + g.join(',') + ' algo 内=' + inside);
    }
  }
  ok('3 二级 7 个标签全部独立切换', true);

  // ---------- 4. 主页面搜索 ----------
  await page.click('#pg-algo header nav button[data-p="map"]');
  await page.waitForTimeout(200);
  const n0 = await page.$$eval('#pg-algo #layers .node', e => e.length);
  await page.fill('#q', 'PCA');
  await page.waitForTimeout(350);
  const n1 = await page.$$eval('#pg-algo #layers .node', e => e.length);
  await page.fill('#q', '');
  await page.waitForTimeout(300);
  ok('4 主页面搜索 #q 生效', n0 > 0 && n1 > 0 && n1 < n0, `${n0} -> ${n1}`);

  // ---------- 5. 选题系统搜索 ----------
  await page.click('#topnav button[data-m="pick"]');
  await page.waitForTimeout(300);
  const c0 = await page.$$eval('#pg-pick .card', e => e.length);
  await page.fill('#pk-q', 'Transformer');
  await page.waitForTimeout(350);
  const c1 = await page.$$eval('#pg-pick .card', e => e.length);
  await page.fill('#pk-q', '');
  await page.waitForTimeout(300);
  ok('5 选题搜索 #pk-q 生效', c0 === 30 && c1 > 0 && c1 < c0, `${c0} -> ${c1}`);

  // ---------- 6. 弹窗隔离 ----------
  await page.click('#pg-pick .card button[data-pick]');
  await page.waitForTimeout(300);
  const maskVis = await page.isVisible('#pk-mask');
  const modalHas = await page.$$eval('#pk-modal', e => e.length && e[0].innerHTML.length > 50);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);
  const maskGone = await page.isVisible('#pk-mask');
  ok('6 选题弹窗打开 + Esc 关闭', maskVis && modalHas && !maskGone,
    `mask=${maskVis} modal=${!!modalHas} 关闭后=${maskGone}`);

  await page.click('#topnav button[data-m="algo"]');
  await page.waitForTimeout(200);
  await page.keyboard.press('Escape');       // 在 algo 模块按 Esc 不应触发 B
  await page.waitForTimeout(200);
  const stillAlgo = await page.$eval('#pg-algo', e => !e.hidden);
  ok('6b 跨模块 Esc 不串台', stillAlgo);

  // ---------- 7. 样式作用域未泄漏 ----------
  const widths = await page.evaluate(() => {
    const g = s => { const e = document.querySelector(s); return e ? getComputedStyle(e).maxWidth : null; };
    return { algo: g('#pg-algo .wrap'), pick: g('#pg-pick .wrap'), book: g('#pg-book .wrap') };
  });
  ok('7 三个 .wrap 各有独立 maxWidth',
    widths.algo && widths.pick && widths.book &&
    new Set([widths.algo, widths.pick, widths.book]).size >= 2, JSON.stringify(widths));

  // md2html 的边框定义在 th 背景而非 table border，故用「表格行数 + th 底色」判断
  const coefTable = await page.evaluate(() => {
    const ts = document.querySelectorAll('#pg-coef table');
    const th = ts.length ? ts[0].querySelector('th') : null;
    return {
      n: ts.length,
      rows: document.querySelectorAll('#pg-coef table tr').length,
      thBg: th ? getComputedStyle(th).backgroundColor : null,
    };
  });
  ok('7b 难度系数表表格已还原',
    coefTable.n >= 5 && coefTable.rows >= 30 && coefTable.thBg !== 'rgba(0, 0, 0, 0)',
    JSON.stringify(coefTable));

  const bookTable = await page.$$eval('#pg-book table', e => e.length);
  ok('7c 任务书一页速览表格已还原', bookTable >= 1, '表格数=' + bookTable);

  // ---------- 8. 打印 ----------
  await page.emulateMedia({ media: 'print' });
  const pr = await page.evaluate(() => ({
    nav: getComputedStyle(document.getElementById('topnav')).display,
    visiblePg: Array.from(document.querySelectorAll('.pg')).filter(e => !e.hidden).map(e => e.id),
  }));
  await page.emulateMedia({ media: 'screen' });
  ok('8 打印时顶栏隐藏且仅当前模块', pr.nav === 'none' && pr.visiblePg.length === 1,
    JSON.stringify(pr));

  // ---------- 9. 全局符号隔离 ----------
  const leak = await page.evaluate(() => ({
    render: typeof window.render, esc: typeof window.esc,
    DATA: typeof window.DATA, TOPICS: typeof window.TOPICS,
  }));
  ok('9 顶层符号未泄漏到 window',
    leak.render === 'undefined' && leak.esc === 'undefined' &&
    leak.DATA === 'undefined' && leak.TOPICS === 'undefined', JSON.stringify(leak));

  // ---------- 10. 截图 ----------
  for (const m of mods) {
    await page.click(`#topnav button[data-m="${m}"]`);
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(SHOTS, `merged-${m}.png`) });
  }
  ok('10 四张模块截图已生成', fs.readdirSync(SHOTS).filter(f => f.startsWith('merged-')).length === 4);

  ok('Z 全程零运行时错误（复检）', errors.length === 0, errors.slice(0, 3).join(' | '));

  console.log('\n===== PASS ' + PASS + ' / FAIL ' + FAIL + ' =====');
  if (errors.length) console.log('错误明细:\n' + errors.slice(0, 10).join('\n'));
  await browser.close();
  process.exit(FAIL ? 1 : 0);
})().catch(e => { console.error('脚本异常:', e); process.exit(2); });
