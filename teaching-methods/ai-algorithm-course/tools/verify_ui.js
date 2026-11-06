/* 主页面 UI 回归测试：Tab 切换 + 导出实验计划按钮
 * 用法：node tools/verify_ui.js
 * 说明：Chromium 路径可用环境变量 PLAYWRIGHT_CHROME 覆盖。 */
const { chromium } = require('playwright');
const path = require('path');
const EXE = process.env.PLAYWRIGHT_CHROME
  || 'C:\\Users\\ufo00\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe';
const FILE = 'file:///' + path.join(__dirname, '..', 'outputs', 'AI算法大纲-交互版.html').replace(/\\/g, '/');

const TABS = [
  ['map', 'p-map'], ['svd', 'p-svd'], ['pca', 'p-pca'], ['plan', 'p-plan'],
  ['lab', 'p-lab'], ['book', 'p-book'], ['stu', 'p-stu']
];

let fail = 0;
function chk(ok, msg, extra) {
  console.log((ok ? 'PASS  ' : 'FAIL  ') + msg + (extra ? '  → ' + extra : ''));
  if (!ok) fail++;
}

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push('[pageerror] ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('[console] ' + m.text()); });
  await page.goto(FILE, { waitUntil: 'load' });
  await page.waitForTimeout(500);

  // ---------- 1. Tab 切换 ----------
  console.log('== Tab 切换 ==');
  const navCount = await page.locator('nav button[data-p]').count();
  chk(navCount === 7, 'nav 按钮数量为 7', String(navCount));

  for (const [key, paneId] of TABS) {
    await page.click(`nav button[data-p="${key}"]`);
    await page.waitForTimeout(120);
    const st = await page.evaluate(() => {
      const on = Array.from(document.querySelectorAll('.pane.on'));
      const navOn = Array.from(document.querySelectorAll('nav button.on'));
      return {
        panes: on.map(p => p.id).join(','),
        visible: on.length === 1 && getComputedStyle(on[0]).display !== 'none',
        nav: navOn.map(b => b.getAttribute('data-p')).join(',')
      };
    });
    chk(st.panes === paneId && st.visible && st.nav === key,
      `点击 ${key} → 显示 ${paneId}`,
      `实际 pane=${st.panes} nav=${st.nav} visible=${st.visible}`);
  }

  // 逐个 Tab 检查是否有非零高度内容
  for (const [key, paneId] of TABS) {
    await page.click(`nav button[data-p="${key}"]`);
    await page.waitForTimeout(150);
    const h = await page.evaluate(id => {
      const el = document.getElementById(id);
      return el ? { h: el.getBoundingClientRect().height, txt: el.innerText.trim().length } : null;
    }, paneId);
    chk(h && h.h > 60 && h.txt > 40, `${paneId} 有可见内容`, h ? `高度 ${Math.round(h.h)}px / 文字 ${h.txt} 字` : 'null');
  }

  // ---------- 2. 导出实验计划 ----------
  console.log('\n== 导出实验计划 ==');
  await page.click('nav button[data-p="map"]');
  await page.waitForTimeout(150);

  // 清空后立即导出应弹 alert
  await page.evaluate(() => { PICK.sel = {}; savePick(); render('all', ''); });
  let alerted = false;
  page.once('dialog', async d => { alerted = true; await d.dismiss(); });
  await page.click('#pkExp');
  await page.waitForTimeout(300);
  chk(alerted, '未勾选时导出给出提示', alerted ? '' : '未弹出 alert');

  // 按重点推荐后导出
  await page.click('#pkAuto');
  await page.waitForTimeout(300);
  const num = await page.textContent('#pkNum');
  chk(Number(num) > 0, 'pkAuto 后有勾选数量', String(num));

  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 8000 }),
    page.click('#pkExp')
  ]);
  const name = dl.suggestedFilename();
  chk(/\.md$/.test(name), '下载文件名以 .md 结尾', name);
  const fs = require('fs');
  const tmp = path.join(process.env.TEMP || '.', '_plan_' + Date.now() + '.md');
  await dl.saveAs(tmp);
  const md = fs.readFileSync(tmp, 'utf8');
  fs.unlinkSync(tmp, () => {});
  chk(md.indexOf('实验计划') >= 0 && md.indexOf('L1-01') >= 0 && md.length > 400,
    'Markdown 含实验编号与内容', `${md.length} 字符`);
  chk(!/undefined|NaN/.test(md), 'Markdown 无 undefined/NaN',
    (md.match(/undefined|NaN/g) || []).slice(0, 3).join(','));

  console.log('\n示例内容：\n' + md.split('\n').slice(0, 12).join('\n'));

  console.log(errs.length ? '\n运行时错误 ' + errs.length + ' 条：\n' + errs.slice(0, 10).join('\n') : '\n无运行时错误');
  await browser.close();
  console.log(fail ? `\n${fail} 项未通过` : '\n全部通过');
  process.exit(fail || errs.length ? 1 : 0);
})();
