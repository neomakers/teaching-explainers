/* 检查 HTML 内所有内联 script 的语法。
 * 用法：node tools/check_syntax.js [文件1] [文件2] ...
 * 不给参数时检查主页面 + 选题系统。
 */
const fs = require('fs');
const path = require('path');

const DEFAULTS = [
  path.join(__dirname, '..', 'outputs', 'AI算法大纲-交互版.html'),
  path.join(__dirname, '..', 'outputs', '选题系统.html')
];
const files = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULTS;

let bad = 0;
for (const file of files) {
  if (!fs.existsSync(file)) {
    console.log('跳过（不存在）：' + file);
    continue;
  }
  const html = fs.readFileSync(file, 'utf8');
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  let m, i = 0;
  const name = path.basename(file);
  while ((m = re.exec(html))) {
    i++;
    const code = m[1];
    if (!code.trim()) continue;
    try {
      new Function(code);
      console.log(name + '  script #' + i + ' OK  (' + code.split('\n').length + ' 行)');
    } catch (e) {
      bad++;
      console.log(name + '  script #' + i + ' 语法错误: ' + e.message);
    }
  }
}
console.log(bad ? '\n存在 ' + bad + ' 处语法错误' : '\n全部 script 语法通过');
process.exit(bad ? 1 : 0);
