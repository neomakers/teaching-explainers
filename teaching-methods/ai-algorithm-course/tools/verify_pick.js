/* 选题系统功能验证
 * 用法：node tools/verify_pick.js
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const EXE = process.env.PLAYWRIGHT_CHROME
  || 'C:\\Users\\ufo00\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe';
const FILE = 'file:///' + path.join(__dirname, '..', 'outputs', '选题系统.html').replace(/\\/g, '/');

let fail = 0;
function chk(ok, msg, extra) {
  console.log((ok ? 'PASS  ' : 'FAIL  ') + msg + (extra ? '  → ' + extra : ''));
  if (!ok) fail++;
}
// 捕获 alert 文本
let lastAlert = '';
async function expectAlert(page, fn, msg) {
  lastAlert = '';
  const h = async d => { lastAlert = d.message(); await d.dismiss(); };
  page.on('dialog', h);
  await fn();
  await page.waitForTimeout(250);
  page.off('dialog', h);
  return lastAlert;
}

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, acceptDownloads: true });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push('[pageerror] ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('[console] ' + m.text()); });
  await page.goto(FILE, { waitUntil: 'load' });
  await page.waitForTimeout(400);

  // ---------- 渲染 ----------
  console.log('== 渲染与筛选 ==');
  chk(await page.locator('.card').count() === 30, '渲染 30 张选题卡',
    String(await page.locator('.card').count()));

  const dims = await page.locator('.card').first().locator('.dim').count();
  chk(dims === 5, '每张卡有 5 个难度维度柱', String(dims));

  for (const [t, want] of [['A', 5], ['B', 13], ['C', 10], ['D', 2]]) {
    await page.click(`.chip[data-t="${t}"]`);
    await page.waitForTimeout(120);
    const n = await page.locator('.card').count();
    chk(n === want, `${t} 档筛出 ${want} 题`, String(n));
  }
  await page.click('.chip[data-t="all"]');
  await page.waitForTimeout(120);

  await page.click('.chip[data-l="L3"]');
  await page.waitForTimeout(120);
  chk(await page.locator('.card').count() === 6, 'L3 层筛出 6 题',
    String(await page.locator('.card').count()));
  await page.click('.chip[data-l="L3"]');
  await page.waitForTimeout(120);

  await page.fill('#q', '降维');
  await page.waitForTimeout(160);
  const nq = await page.locator('.card').count();
  chk(nq > 0 && nq < 30, '搜索「降维」有结果且收窄', nq + ' 题');
  await page.fill('#q', 'xyz不存在xyz');
  await page.waitForTimeout(160);
  chk(await page.locator('#empty').isVisible(), '无结果时显示空状态');
  await page.fill('#q', '');
  await page.waitForTimeout(160);

  // ---------- 锁定题目 ----------
  console.log('\n== 选开放状态 ==');
  const locked = await page.locator('.card[data-id="L8-02"]').getAttribute('class');
  chk(/locked/.test(locked), 'L8-02 生成模型默认锁定', locked);
  chk(await page.locator('.card[data-id="L8-02"] button[data-pick]').count() === 0,
    '锁定题目没有「我要选」按钮');

  // ---------- 登记流程 ----------
  console.log('\n== 登记流程 ==');
  await page.click('.card[data-id="L3-01"] button[data-pick]');
  await page.waitForTimeout(220);
  chk(await page.locator('#mask').isVisible(), '弹出登记表单');
  chk((await page.locator('#modal').innerHTML()).indexOf('K=1.15') >= 0,
    '表单显示该题难度系数 K=1.15');

  // 校验：成员不全
  await page.selectOption('#f-g', 'G01');
  await page.fill('#f-m0', '张三');
  await page.selectOption('#f-ralgo', '0');
  await expectAlert(page, () => page.click('#f-ok'));
  chk(/三名成员/.test(lastAlert), '校验：成员不足 3 人被拦下', lastAlert);

  // 校验：非法仓库
  await page.fill('#f-m1', '李四');
  await page.fill('#f-m2', '王五');
  await page.selectOption('#f-rviz', '1');
  await page.selectOption('#f-rdoc', '2');
  await page.fill('#f-repo', 'https://notareal.com/x');
  await expectAlert(page, () => page.click('#f-ok'));
  chk(/仓库地址/.test(lastAlert), '校验：非 github/gitee 仓库被拦下', lastAlert);

  // 正常提交
  await page.fill('#f-repo', 'https://gitee.com/demo/ai-pca');
  await page.check('input.f-ai[value="CodeBuddy"]');
  await expectAlert(page, () => page.click('#f-ok'));
  await page.waitForTimeout(250);
  chk(lastAlert === '', '正常提交无异常拦截', lastAlert || '(无 alert)');
  chk(!(await page.locator('#mask').isVisible()), '提交后关闭表单');
  const used = await page.locator('.card[data-id="L3-01"] .used b').textContent();
  chk(used === '1', '卡片显示已选 1 组', used);
  const tag = await page.locator('.card[data-id="L3-01"] .gtag').textContent();
  chk(tag === 'G01', '卡片显示组号 G01', tag);

  // ---------- 撞题检测 ----------
  console.log('\n== 撞题检测 ==');
  await page.click('.card[data-id="L3-01"] button[data-pick]');
  await page.waitForTimeout(220);
  chk((await page.locator('#modal .warnbox').first().textContent()).indexOf('撞题') >= 0,
    '同题第二次选中给出撞题警告');
  await page.selectOption('#f-g', 'G02');
  await page.waitForTimeout(120);
  await page.fill('#f-m0', '赵六'); await page.fill('#f-m1', '孙七'); await page.fill('#f-m2', '周八');
  await page.selectOption('#f-ralgo', '0');
  await page.selectOption('#f-rviz', '1');
  await page.selectOption('#f-rdoc', '2');
  await page.fill('#f-repo', 'https://gitee.com/demo/ai-pca-2');
  await page.click('#f-ok');
  await page.waitForTimeout(250);
  chk(await page.locator('.card[data-id="L3-01"] .gtag.conf').count() === 2,
    '两个组号都标为撞题色');

  // 名额满
  await page.click('.card[data-id="L3-01"] button[data-pick]');
  await page.waitForTimeout(220);
  await page.selectOption('#f-g', 'G03');
  await page.waitForTimeout(120);
  await page.fill('#f-m0', '吴九'); await page.fill('#f-m1', '郑十'); await page.fill('#f-m2', '何一');
  await page.selectOption('#f-ralgo', '0');
  await page.selectOption('#f-rviz', '1');
  await page.selectOption('#f-rdoc', '2');
  await page.fill('#f-repo', 'https://gitee.com/demo/ai-pca-3');
  await page.click('#f-ok');
  await page.waitForTimeout(250);
  chk(await page.locator('.card[data-id="L3-01"] button[data-pick]').count() === 0,
    '满 3 组后不再显示「我要选」');

  // ---------- 教师看板 ----------
  console.log('\n== 教师看板 ==');
  await page.click('#toggleBoard');
  await page.waitForTimeout(200);
  chk(await page.locator('#board').isVisible(), '看板可见');
  const bs = await page.locator('#bstats').textContent();
  chk(bs.indexOf('3') >= 0 && bs.indexOf('7 组未选题') >= 0, '看板统计正确',
    bs.replace(/\s+/g, ' ').trim().slice(0, 90));
  chk(await page.locator('.gi.conf').count() === 3, '撞题的 3 个组在看板标红',
    String(await page.locator('.gi.conf').count()));
  chk(/A 档 0\/5/.test(await page.locator('#tierbar').textContent()),
    '难度分布条显示 A 档 0/5');

  // ---------- 导出 ----------
  console.log('\n== 导出 ==');
  const grab = async (sel, ext) => {
    const [dl] = await Promise.all([
      page.waitForEvent('download', { timeout: 8000 }),
      page.click(sel)
    ]);
    const tmp = path.join(process.env.TEMP || '.', '_exp_' + Date.now() + ext);
    await dl.saveAs(tmp);
    const txt = fs.readFileSync(tmp, 'utf8');
    fs.unlinkSync(tmp);
    return { name: dl.suggestedFilename(), txt };
  };
  const md = await grab('#btnExpMd', '.md');
  chk(/\.md$/.test(md.name) && md.txt.indexOf('选题登记') >= 0, '导出 Markdown', md.name);
  chk(md.txt.indexOf('L3-01') >= 0 && md.txt.indexOf('gitee.com/demo/ai-pca') >= 0,
    'Markdown 含选题与仓库地址');
  chk(md.txt.indexOf('撞题') >= 0, 'Markdown 含撞题章节');
  const csv = await grab('#btnExpCsv', '.csv');
  chk(/\.csv$/.test(csv.name) && csv.txt.indexOf('难度系数K') >= 0, '导出 CSV', csv.name);
  chk(csv.txt.split('\n').length === 4, 'CSV 有 3 行数据（表头 + 3 组）',
    String(csv.txt.split('\n').length));
  const js = await grab('#btnExpJson', '.json');
  let parsed = null;
  try { parsed = JSON.parse(js.txt); } catch (e) {}
  chk(parsed && Object.keys(parsed.groups).length === 3, '导出 JSON 可被解析且含 3 组',
    parsed ? Object.keys(parsed.groups).join(',') : '解析失败');

  // ---------- 持久化 ----------
  console.log('\n== 持久化 ==');
  const beforeReload = await page.locator('.card[data-id="L3-01"] .used b').textContent();
  await page.reload({ waitUntil: 'load' });
  await page.waitForTimeout(400);
  const after = await page.locator('.card[data-id="L3-01"] .used b').textContent();
  chk(beforeReload === after && after === '3', '刷新后选题仍在', after);

  // ---------- 骨架代码 ----------
  console.log('\n== 骨架代码 ==');
  await page.click('#btnSkeleton');
  await page.waitForTimeout(250);
  const sk = await page.locator('#modal pre').textContent();
  chk(sk.indexOf('runDemo') >= 0 && sk.indexOf('update();') >= 0, '骨架内容正确',
    sk.split('\n').length + ' 行');
  chk(sk.indexOf('<script src="demo.js">') >= 0, '骨架里的 script 标签正确还原');
  await page.click('#sk-close');
  await page.waitForTimeout(150);
  chk(!(await page.locator('#mask').isVisible()), '骨架弹层可关闭');

  // ---------- 清空 ----------
  page.once('dialog', d => d.accept());
  await page.click('#toggleBoard');          // reload 后看板已收起，先展开
  await page.waitForTimeout(200);
  await page.click('#btnClearAll');
  await page.waitForTimeout(300);
  chk(await page.locator('.card[data-id="L3-01"] .used b').textContent() === '0',
    '清空后归零');

  console.log(errs.length ? '\n运行时错误 ' + errs.length + ' 条：\n' + errs.slice(0, 10).join('\n')
    : '\n无运行时错误');
  await browser.close();
  console.log(fail ? `\n${fail} 项未通过` : '\n全部通过');
  process.exit(fail || errs.length ? 1 : 0);
})();
