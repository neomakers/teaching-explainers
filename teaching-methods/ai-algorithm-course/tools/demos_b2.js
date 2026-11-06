/* DEMOS_B2:BEGIN */
/* ===== 13. 特征值分解（2×2 对称矩阵） ===== */
regDemo(['特征值分解', 'Eigendecomposition'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '单位圆 ‖v‖=1 → 变换后椭圆', w: 280, h: 250},
            {k: 'b', t: '网格被拉伸 / 压缩', w: 280, h: 250}],
    legend: '<span><i style="background:#b4b2a9"></i>变换前</span>'
          + '<span><i style="background:#185fa5"></i>变换后</span>'
          + '<span><i style="background:#c0392b"></i>v₁ 方向</span>'
          + '<span><i style="background:#0f6e56"></i>v₂ 方向</span>',
    ctl: [{k: 's', label: 'a = A₁₁', min: -2.5, max: 2.5, step: 0.1, val: 1.5},
          {k: 't', label: 'c = A₂₂', min: -2.5, max: 2.5, step: 0.1, val: 0.8},
          {k: 'u', label: 'b = A₁₂ = A₂₁', min: -1.5, max: 1.5, step: 0.1, val: 0.7}],
    kpi: [{k: 'p', label: 'λ₁'}, {k: 'q', label: 'λ₂'},
          {k: 'r', label: '条件数 κ'}, {k: 'z', label: 'det A'}]
  });
  const g = x => document.getElementById(id + x);
  const S = 30;
  function axes(c, W, H){
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
    c.beginPath(); c.moveTo(W / 2, 0); c.lineTo(W / 2, H); c.stroke();
  }
  function arrow(c, x0, y0, x1, y1, col, w){
    c.strokeStyle = col; c.fillStyle = col; c.lineWidth = w || 2.2;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    const an = Math.atan2(y1 - y0, x1 - x0), L = 8;
    c.beginPath(); c.moveTo(x1, y1);
    c.lineTo(x1 - L * Math.cos(an - 0.42), y1 - L * Math.sin(an - 0.42));
    c.lineTo(x1 - L * Math.cos(an + 0.42), y1 - L * Math.sin(an + 0.42));
    c.closePath(); c.fill();
  }
  function up(){
    const a = parseFloat(g('s').value), cc = parseFloat(g('t').value), b = parseFloat(g('u').value);
    g('sv').textContent = a.toFixed(1);
    g('tv').textContent = cc.toFixed(1);
    g('uv').textContent = b.toFixed(1);
    const eg = eigen2([[a, b], [b, cc]]), l1 = eg.l1, l2 = eg.l2;
    const v1 = eg.e1, v2 = [-v1[1], v1[0]];

    const cv = g('cva'), c = dclr(cv), W = cv.width, H = cv.height;
    const X = x => W / 2 + x * S, Y = y => H / 2 - y * S;
    axes(c, W, H);
    c.strokeStyle = '#b4b2a9'; c.lineWidth = 1.6; c.beginPath();
    for(let i = 0; i <= 120; i++){
      const th = i / 120 * 6.2832;
      i ? c.lineTo(X(Math.cos(th)), Y(Math.sin(th))) : c.moveTo(X(1), Y(0));
    }
    c.closePath(); c.stroke();
    c.strokeStyle = '#185fa5'; c.lineWidth = 2.2; c.beginPath();
    for(let i = 0; i <= 120; i++){
      const th = i / 120 * 6.2832, co = Math.cos(th), si = Math.sin(th);
      i ? c.lineTo(X(a * co + b * si), Y(b * co + cc * si)) : c.moveTo(X(a), Y(b));
    }
    c.closePath(); c.stroke();
    arrow(c, X(0), Y(0), X(v1[0]), Y(v1[1]), '#c0392b');
    arrow(c, X(0), Y(0), X(l1 * v1[0]), Y(l1 * v1[1]), 'rgba(192,57,43,.45)', 1.4);
    arrow(c, X(0), Y(0), X(v2[0]), Y(v2[1]), '#0f6e56');
    arrow(c, X(0), Y(0), X(l2 * v2[0]), Y(l2 * v2[1]), 'rgba(15,110,86,.45)', 1.4);

    const cv2 = g('cvb'), c2 = dclr(cv2);
    const X2 = x => cv2.width / 2 + x * S, Y2 = y => cv2.height / 2 - y * S;
    axes(c2, cv2.width, cv2.height);
    const R = 3.4, st = 0.68;
    c2.strokeStyle = '#e3e0d8'; c2.lineWidth = 1;
    for(let t = -R; t <= R + 1e-9; t += st){
      c2.beginPath(); c2.moveTo(X2(t), Y2(-R)); c2.lineTo(X2(t), Y2(R)); c2.stroke();
      c2.beginPath(); c2.moveTo(X2(-R), Y2(t)); c2.lineTo(X2(R), Y2(t)); c2.stroke();
    }
    c2.strokeStyle = 'rgba(24,95,165,.6)'; c2.lineWidth = 1.1;
    for(let t = -R; t <= R + 1e-9; t += st){
      c2.beginPath(); c2.moveTo(X2(a * t + b * -R), Y2(b * t + cc * -R));
      c2.lineTo(X2(a * t + b * R), Y2(b * t + cc * R)); c2.stroke();
      c2.beginPath(); c2.moveTo(X2(a * -R + b * t), Y2(b * -R + cc * t));
      c2.lineTo(X2(a * R + b * t), Y2(b * R + cc * t)); c2.stroke();
    }
    arrow(c2, X2(0), Y2(0), X2(l1 * v1[0]), Y2(l1 * v1[1]), '#c0392b');
    arrow(c2, X2(0), Y2(0), X2(l2 * v2[0]), Y2(l2 * v2[1]), '#0f6e56');

    const det = a * cc - b * b;
    const kap = Math.abs(l2) < 1e-6 ? Infinity : Math.abs(l1 / l2);
    g('p').textContent = l1.toFixed(2);
    g('q').textContent = l2.toFixed(2);
    g('r').textContent = isFinite(kap) ? kap.toFixed(1) : '∞';
    g('z').textContent = det.toFixed(2);
    g('rw').className = 'kpi' + (kap > 12 ? ' warn' : ' ok');
    let note;
    if(Math.abs(b) < 0.06) note = 'b≈0：矩阵已是<b>对角形</b>，特征向量就是坐标轴 x / y，分解 trivial。往右拖动 b 看看椭圆怎么「歪」起来。';
    else if(det < 0) note = 'det A = <b>' + det.toFixed(2) + '</b> < 0：两个特征值异号，这是一个<b>鞍形</b>变换——一个方向被拉伸、另一个被翻转压缩。';
    else if(kap > 12) note = 'κ = <b>' + kap.toFixed(1) + '</b>：λ₂ 已接近 0，圆被压成一条细线，A 接近<b>奇异</b>（不可逆）。κ 越大，求解 Ax=b 越不稳。';
    else note = '红色箭头（v₁，λ₁=' + l1.toFixed(2) + '）与绿色箭头（v₂，λ₂=' + l2.toFixed(2)
      + '）就是特征向量：<b>A 作用在上面只改变长度、不改变方向</b>。椭圆的长短半轴恰好是 |λ₁| 与 |λ₂|。';
    g('n').innerHTML = note;
  }
  ['s', 't', 'u'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 14. 标准化与归一化 ===== */
regDemo(['标准化与归一化', 'Standardization / Normalization'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '原始特征分布（含异常值）', w: 300, h: 230},
            {k: 'b', t: '变换后分布', w: 300, h: 230}],
    legend: '<span><i style="background:#185fa5"></i>特征 1（偏态收入型）</span>'
          + '<span><i style="background:#c0392b"></i>特征 2（量纲 100×）</span>',
    ctl: [{k: 's', label: '偏态程度 σ', min: 0.2, max: 1.6, step: 0.05, val: 0.9},
          {k: 't', label: '异常值倍数', min: 0, max: 8, step: 0.5, val: 4}],
    btns: [{k: 'g1', label: '原始'}, {k: 'g2', label: 'Z-score'},
           {k: 'g3', label: 'Min-Max'}, {k: 'g4', label: 'Robust'}],
    kpi: [{k: 'p', label: '均值'}, {k: 'q', label: '标准差'},
          {k: 'r', label: '极差'}, {k: 'z', label: '两特征距离占比'}]
  });
  const g = x => document.getElementById(id + x);
  const MODES = ['raw', 'zscore', 'minmax', 'robust'];
  const BKEY = ['g1', 'g2', 'g3', 'g4'];
  let mode = 0;
  function rawData(){
    const sg = parseFloat(g('s').value), k = parseFloat(g('t').value);
    const r = gaussRnd(20241033);
    const f1 = [], f2 = [];
    for(let i = 0; i < 260; i++) f1.push(Math.exp(sg * r()));
    for(let i = 0; i < 260; i++) f2.push(100 + 24 * r());
    if(k > 0){ f1.push(Math.exp(sg * 3.1) * (1 + 0.9 * k)); f2.push(100 + 24 * 3.4 * (1 + 0.6 * k)); }
    return [f1, f2];
  }
  function stat(v){
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / v.length);
    const sv = v.slice().sort((a, b) => a - b);
    const q1 = sv[Math.floor(sv.length * 0.25)], q3 = sv[Math.floor(sv.length * 0.75)];
    return {m: m, sd: sd, lo: sv[0], hi: sv[sv.length - 1], q1: q1, q3: q3, iqr: q3 - q1};
  }
  function apply(v, md){
    const s = stat(v);
    if(md === 'zscore') return v.map(x => (x - s.m) / (s.sd || 1));
    if(md === 'minmax') return v.map(x => (x - s.lo) / ((s.hi - s.lo) || 1));
    if(md === 'robust'){ const m2 = (s.q1 + s.q3) / 2; return v.map(x => (x - m2) / (s.iqr || 1)); }
    return v.slice();
  }
  function hist(cv, series, unit){
    const c = dclr(cv), W = cv.width, H = cv.height, PB = 22;
    const all = series[0].concat(series[1]);
    let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if(unit){ const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
    if(hi - lo < 1e-9){ hi = lo + 1; }
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, H - PB); c.lineTo(W, H - PB); c.stroke();
    const NB = 34, cols = ['rgba(24,95,165,.55)', 'rgba(192,57,43,.55)'];
    series.forEach(function(v, si){
      const bins = new Array(NB).fill(0);
      v.forEach(function(x){ const i = Math.min(NB - 1, Math.floor((x - lo) / (hi - lo) * NB)); bins[i]++; });
      const mx = Math.max.apply(null, bins) || 1;
      const bw = W / NB;
      c.fillStyle = cols[si];
      bins.forEach(function(n, i){
        const h = n / mx * (H - PB - 8);
        c.fillRect(i * bw + (si ? bw * 0.42 : 0), H - PB - h, bw * 0.58, h);
      });
    });
    c.fillStyle = '#888780'; c.font = '10px sans-serif';
    c.fillText(lo.toFixed(1), 2, H - 7);
    c.textAlign = 'right'; c.fillText(hi.toFixed(1), W - 2, H - 7); c.textAlign = 'left';
  }
  function up(){
    const md = MODES[mode];
    g('sv').textContent = parseFloat(g('s').value).toFixed(2);
    g('tv').textContent = '×' + parseFloat(g('t').value).toFixed(1);
    BKEY.forEach(function(k, i){
      const b = g(k);
      b.style.background = i === mode ? '#185fa5' : '#fff';
      b.style.color = i === mode ? '#fff' : '#185fa5';
    });
    const raw = rawData();
    const out = raw.map(v => apply(v, md));
    hist(g('cva'), raw, false);
    hist(g('cvb'), out, md !== 'raw');
    const s = stat(out[0]);
    g('p').textContent = s.m.toFixed(2);
    g('q').textContent = s.sd.toFixed(2);
    g('r').textContent = (s.hi - s.lo).toFixed(2);
    let d1 = 0, d2 = 0;
    for(let i = 1; i < out[0].length; i++){
      d1 += Math.abs(out[0][i] - out[0][i - 1]);
      d2 += Math.abs(out[1][i] - out[1][i - 1]);
    }
    const share = d1 + d2 > 0 ? d2 / (d1 + d2) : 0;
    g('z').textContent = (share * 100).toFixed(0) + '%';
    g('zw').className = 'kpi' + (share > 0.7 || share < 0.3 ? ' warn' : ' ok');
    let note;
    if(md === 'raw') note = '特征 2 只是量纲大 100 倍，却贡献了 <b>' + (share * 100).toFixed(0)
      + '%</b> 的距离。KNN / K-Means / PCA 全都会被它带跑——这就是必须做尺度变换的原因。';
    else if(md === 'zscore') note = 'Z-score：(x−μ)/σ。均值归 <b>0</b>、标准差归 <b>1</b>，两特征距离占比被拉回 <b>'
      + (share * 100).toFixed(0) + '%</b>。缺点是异常值会把 σ 撑大，压缩正常样本。';
    else if(md === 'minmax') note = 'Min-Max：压到 [0,1]，保留原始形状，适合神经网络输入与图像。'
      + '但<b>一个异常值就会把其余样本挤成一团</b>——把「异常值倍数」拉到 8 试试。';
    else note = 'Robust：用中位数与 IQR 替代均值与标准差，<b>不受异常值影响</b>。'
      + '数据脏、长尾重时优先选它。当前两特征距离占比 <b>' + (share * 100).toFixed(0) + '%</b>。';
    g('n').innerHTML = note;
  }
  BKEY.forEach(function(k, i){ g(k).onclick = function(){ mode = i; up(); }; });
  ['s', 't'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 15. 逻辑回归 ===== */
regDemo(['逻辑回归', 'Logistic Regression'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '决策边界 + 预测概率', w: 300, h: 250},
            {k: 'b', t: '交叉熵损失随迭代下降', w: 300, h: 250}],
    legend: '<span><i style="background:#c0392b"></i>类别 0</span>'
          + '<span><i style="background:#0f6e56"></i>类别 1</span>'
          + '<span><i style="background:#185fa5"></i>决策边界 p=0.5</span>',
    ctl: [{k: 's', label: '迭代轮数', min: 0, max: 400, step: 10, val: 120},
          {k: 't', label: '学习率 lr（0.01→2 对数）', min: 0, max: 6, step: 1, val: 4},
          {k: 'u', label: 'L2 正则 λ', min: 0, max: 0.2, step: 0.01, val: 0}],
    kpi: [{k: 'p', label: '训练准确率'}, {k: 'q', label: '交叉熵'},
          {k: 'r', label: '‖w‖'}, {k: 'z', label: '边界截距 b'}]
  });
  const g = x => document.getElementById(id + x);
  const LRS = [0.01, 0.05, 0.1, 0.3, 0.6, 1.2, 2.0];
  const rnd0 = gaussRnd(20241044);
  const Xr = [], yr = [];
  for(let i = 0; i < 90; i++){ Xr.push([-0.78 + 0.64 * rnd0(), 0.55 + 0.64 * rnd0()]); yr.push(0); }
  for(let i = 0; i < 90; i++){ Xr.push([0.78 + 0.64 * rnd0(), -0.55 + 0.64 * rnd0()]); yr.push(1); }
  const mu = [0, 0], sd = [1, 1];
  [0, 1].forEach(function(j){
    mu[j] = Xr.reduce(function(s, p){ return s + p[j]; }, 0) / Xr.length;
    sd[j] = Math.sqrt(Xr.reduce(function(s, p){ return s + (p[j] - mu[j]) * (p[j] - mu[j]); }, 0) / Xr.length) || 1;
  });
  const X = Xr.map(p => [(p[0] - mu[0]) / sd[0], (p[1] - mu[1]) / sd[1]]);
  const y = yr, n = X.length;
  const sg = z => z > 30 ? 1 : (z < -30 ? 0 : 1 / (1 + Math.exp(-z)));
  function lossf(w, b){
    let L = 0;
    for(let i = 0; i < n; i++){
      const p = Math.min(1 - 1e-9, Math.max(1e-9, sg(w[0] * X[i][0] + w[1] * X[i][1] + b)));
      L -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
    }
    return L / n;
  }
  function train(iters, lr, l2){
    let w = [0, 0], b = 0, hist = [];
    for(let it = 0; it < iters; it++){
      let gw0 = 0, gw1 = 0, gb = 0;
      for(let i = 0; i < n; i++){
        const e = sg(w[0] * X[i][0] + w[1] * X[i][1] + b) - y[i];
        gw0 += e * X[i][0]; gw1 += e * X[i][1]; gb += e;
      }
      w[0] -= lr * (gw0 / n + l2 * w[0]);
      w[1] -= lr * (gw1 / n + l2 * w[1]);
      b -= lr * gb / n;
      if(it % 5 === 0) hist.push(lossf(w, b));
    }
    return {w: w, b: b, hist: hist};
  }
  function curve(cv, hist){
    const c = dclr(cv), W = cv.width, H = cv.height;
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H - 20 - i / 4 * (H - 34);
      c.beginPath(); c.moveTo(26, yy); c.lineTo(W - 6, yy); c.stroke();
    }
    if(hist.length < 2){
      c.fillStyle = '#888780'; c.font = '11px sans-serif';
      c.fillText('迭代轮数为 0，尚未开始训练', 30, H / 2);
      return;
    }
    const lo = Math.min.apply(null, hist), hi = Math.max.apply(null, hist);
    const rg = (hi - lo) || 1;
    c.strokeStyle = '#185fa5'; c.lineWidth = 2; c.beginPath();
    hist.forEach(function(v, i){
      const px = 26 + i / (hist.length - 1) * (W - 32), py = H - 20 - (v - lo) / rg * (H - 34);
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    });
    c.stroke();
    c.fillStyle = '#888780'; c.font = '10px sans-serif';
    c.fillText(hi.toFixed(3), 2, 12);
    c.fillText(lo.toFixed(3), 2, H - 22);
  }
  function up(){
    const it = parseInt(g('s').value, 10);
    const lri = parseInt(g('t').value, 10), lr = LRS[lri];
    const l2 = parseFloat(g('u').value);
    g('sv').textContent = it;
    g('tv').textContent = lr;
    g('uv').textContent = l2.toFixed(2);
    const r = train(it, lr, l2), w = r.w, b = r.b;
    const ex = extent(X);
    function pred(wx, wy){ return sg(w[0] * wx + w[1] * wy + b); }
    paintCls(g('cva'), X, y, pred, ex);
    const cv = g('cva'), c = cv.getContext('2d'), W = cv.width, H = cv.height;
    const pts = [];
    for(let t = -3; t <= 3; t += 0.05){
      const wn = Math.hypot(w[0], w[1]);
      if(wn < 1e-6) break;
      const px = t, py = (-b - w[0] * t) / w[1];
      if(py > ex.y0 - 1 && py < ex.y1 + 1) pts.push([px, py]);
    }
    if(pts.length > 1){
      c.strokeStyle = '#185fa5'; c.lineWidth = 2.2; c.beginPath();
      pts.forEach(function(p, i){
        const px = (p[0] - ex.x0) / (ex.x1 - ex.x0) * W;
        const py = H - (p[1] - ex.y0) / (ex.y1 - ex.y0) * H;
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      });
      c.stroke();
    }
    curve(g('cvb'), r.hist);
    const acc = accOf(X.map(p => pred(p[0], p[1]) > 0.5 ? 1 : 0), y);
    const L = lossf(w, b);
    g('p').textContent = (acc * 100).toFixed(1) + '%';
    g('q').textContent = L.toFixed(4);
    g('r').textContent = Math.hypot(w[0], w[1]).toFixed(2);
    g('z').textContent = b.toFixed(2);
    g('pw').className = 'kpi' + (acc > 0.85 ? ' ok' : ' warn');
    let note;
    if(it === 0) note = '迭代 0：w=0、b=0，整张图概率都是 0.5，<b>还没有任何边界</b>。向右拖动看它如何长出来。';
    else if(lr >= 1.2 && it > 200) note = 'lr=<b>' + lr + '</b> 偏大：损失已经贴近最优但仍会震荡。'
      + '逻辑回归的交叉熵是凸的，不会发散，但大学习率会<b>在谷底两侧来回跳</b>。';
    else if(lr <= 0.05) note = 'lr=<b>' + lr + '</b> 太小：<b>' + it + '</b> 轮还没走完，'
      + '边界歪着、损失仍在高位。学习率是逻辑回归最值得先调的那一个。';
    else if(l2 > 0.08) note = 'λ=<b>' + l2.toFixed(2) + '</b>：‖w‖ 被压到 '
      + Math.hypot(w[0], w[1]).toFixed(2) + '，边界明显变平。正则换来的是<b>更稳的泛化</b>，代价是训练准确率略降。';
    else note = '<b>' + it + '</b> 轮后：准确率 ' + (acc * 100).toFixed(1) + '%，交叉熵 '
      + L.toFixed(4) + '。输出是<b>概率</b>而非硬标签——这是它比感知机实用得多的地方。';
    g('n').innerHTML = note;
  }
  ['s', 't', 'u'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 16. 朴素贝叶斯 ===== */
regDemo(['朴素贝叶斯'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '决策边界（含后验等高线）', w: 300, h: 250},
            {k: 'b', t: '后验概率 P(y=1|x) 热图', w: 300, h: 250}],
    legend: '<span><i style="background:#c0392b"></i>类别 0</span>'
          + '<span><i style="background:#0f6e56"></i>类别 1</span>'
          + '<span><i style="background:#185fa5"></i>决策边界</span>',
    ctl: [{k: 's', label: '类先验 π（类别 1 占比）', min: 0.05, max: 0.95, step: 0.05, val: 0.5},
          {k: 't', label: '特征 2 方差放大倍数', min: 0.3, max: 3, step: 0.1, val: 1}],
    btns: [{k: 'g1', label: '各类独立方差（二次边界）'},
           {k: 'g2', label: '共享方差（线性边界）'}],
    kpi: [{k: 'p', label: '准确率'}, {k: 'q', label: '边界类型'},
          {k: 'r', label: '平均置信度'}, {k: 'z', label: '错分样本'}]
  });
  const g = x => document.getElementById(id + x);
  let shared = false;
  const rnd = gaussRnd(20241055);
  const X = [], y = [];
  for(let i = 0; i < 80; i++){ X.push([-1.1 + 0.62 * rnd(), 0.5 + 0.62 * rnd()]); y.push(0); }
  for(let i = 0; i < 80; i++){ X.push([1.1 + 0.62 * rnd(), -0.5 + 0.62 * rnd()]); y.push(1); }
  const ex = extent(X);
  function fit(pi){
    const m = [[0, 0], [0, 0]], v = [[0, 0], [0, 0]], cnt = [0, 0];
    X.forEach(function(p, i){ m[y[i]][0] += p[0]; m[y[i]][1] += p[1]; cnt[y[i]]++; });
    [0, 1].forEach(function(c){ m[c][0] /= cnt[c]; m[c][1] /= cnt[c]; });
    X.forEach(function(p, i){
      v[y[i]][0] += (p[0] - m[y[i]][0]) * (p[0] - m[y[i]][0]);
      v[y[i]][1] += (p[1] - m[y[i]][1]) * (p[1] - m[y[i]][1]);
    });
    [0, 1].forEach(function(c){ v[c][0] = v[c][0] / cnt[c]; v[c][1] = v[c][1] / cnt[c]; });
    const vs = [(v[0][0] + v[1][0]) / 2, (v[0][1] + v[1][1]) / 2];
    return {m: m, v: v, vs: vs, cnt: cnt, pi: pi};
  }
  function logit(f, x, sh, k1){
    const p1 = f.pi, p0 = 1 - f.pi;
    let s = Math.log(p1 / p0);
    const v1 = sh ? [f.vs[0], f.vs[1] * k1] : [f.v[1][0], f.v[1][1] * k1];
    const v0 = sh ? [f.vs[0], f.vs[1] * k1] : [f.v[0][0], f.v[0][1] * k1];
    for(let j = 0; j < 2; j++){
      if(sh) continue;
      s += -0.5 * Math.log(v1[j] / v0[j]);
    }
    for(let j = 0; j < 2; j++){
      s -= 0.5 * ((x[j] - f.m[1][j]) * (x[j] - f.m[1][j]) / v1[j]
                - (x[j] - f.m[0][j]) * (x[j] - f.m[0][j]) / v0[j]);
    }
    return s;
  }
  function up(){
    const pi = parseFloat(g('s').value), k1 = parseFloat(g('t').value);
    g('sv').textContent = pi.toFixed(2);
    g('tv').textContent = '×' + k1.toFixed(1);
    [g('g1'), g('g2')].forEach(function(b, i){
      const on = (i === 1) === shared;
      b.style.background = on ? '#185fa5' : '#fff';
      b.style.color = on ? '#fff' : '#185fa5';
    });
    const f = fit(pi);
    const sc = x => logit(f, x, shared, k1);
    paintCls(g('cva'), X, y, function(wx, wy){ return sc([wx, wy]) > 0 ? 1 : 0; }, ex);
    const cv = g('cva'), c = cv.getContext('2d'), W = cv.width, H = cv.height;
    const P = (wx, wy) => (wx - ex.x0) / (ex.x1 - ex.x0) * W;
    const Q = (wy) => H - (wy - ex.y0) / (ex.y1 - ex.y0) * H;
    [[1.0986, 0.8], [-1.0986, 0.8]].forEach(function(L, idx){
      c.strokeStyle = idx ? 'rgba(24,95,165,.45)' : 'rgba(24,95,165,.9)';
      c.lineWidth = idx ? 1.2 : 2.2;
      c.beginPath();
      let started = false;
      for(let a = 0; a < 360; a += 1){
        let lo = 0, hi = 6;
        for(let it = 0; it < 40; it++){
          const md = (lo + hi) / 2;
          const ang = a * Math.PI / 180;
          const cx = (ex.x0 + ex.x1) / 2, cy = (ex.y0 + ex.y1) / 2;
          const vx = cx + md * Math.cos(ang), vy = cy + md * Math.sin(ang);
          if(sc([vx, vy]) > L) lo = md; else hi = md;
        }
        const ang = a * Math.PI / 180;
        const cx = (ex.x0 + ex.x1) / 2, cy = (ex.y0 + ex.y1) / 2;
        const vx = cx + lo * Math.cos(ang), vy = cy + lo * Math.sin(ang);
        started ? c.lineTo(P(vx, 0), Q(vy)) : (c.moveTo(P(vx, 0), Q(vy)), started = true);
      }
      c.stroke();
    });
    const cv2 = g('cvb'), c2 = dclr(cv2), W2 = cv2.width, H2 = cv2.height, G = 5;
    for(let py = 0; py < H2; py += G){
      for(let px = 0; px < W2; px += G){
        const wx = ex.x0 + (px + 0.5) / W2 * (ex.x1 - ex.x0);
        const wy = ex.y0 + (1 - (py + 0.5) / H2) * (ex.y1 - ex.y0);
        const p = 1 / (1 + Math.exp(-sc([wx, wy])));
        const a = 0.10 + 0.85 * Math.abs(p - 0.5) * 2;
        c2.fillStyle = p > 0.5 ? 'rgba(15,110,86,' + Math.min(1, a) + ')'
                              : 'rgba(192,57,43,' + Math.min(1, a) + ')';
        c2.fillRect(px, py, G, G);
      }
    }
    X.forEach(function(p, i){
      c2.fillStyle = '#fff'; c2.strokeStyle = '#2c2c2a'; c2.lineWidth = 1;
      c2.beginPath(); c2.arc(P(p[0], 0), Q(p[1]), 2, 0, 6.284); c2.fill(); c2.stroke();
    });
    const pr = X.map(p => 1 / (1 + Math.exp(-sc(p))));
    const acc = accOf(pr.map(p => p > 0.5 ? 1 : 0), y);
    const conf = pr.reduce(function(s, p){ return s + Math.max(p, 1 - p); }, 0) / pr.length;
    const err = y.reduce(function(s, v, i){ return s + ((pr[i] > 0.5 ? 1 : 0) !== v ? 1 : 0); }, 0);
    g('p').textContent = (acc * 100).toFixed(1) + '%';
    g('q').textContent = shared ? '线性' : '二次';
    g('r').textContent = (conf * 100).toFixed(1) + '%';
    g('z').textContent = err;
    g('pw').className = 'kpi' + (acc > 0.85 ? ' ok' : ' warn');
    let note;
    if(!shared) note = '各类<b>独立估计方差</b> → 判别式里出现 x<sub>j</sub>² 项，边界是<b>二次曲线</b>。'
      + '这就是「朴素」（特征独立）换来的：只估 2 个均值 + 2 个方差，不需要算协方差矩阵。';
    else if(Math.abs(pi - 0.5) > 0.3) note = 'π=<b>' + pi.toFixed(2) + '</b>：先验一变，'
      + '整条边界就<b>整条平移</b>——这是生成式模型特有的能力，调先验不需要重新训练。';
    else note = '<b>共享方差</b>时二次项相消，边界退化成<b>直线</b>，与 LDA 表达式一致。'
      + '当前错分 <b>' + err + '</b> 个样本，平均置信度 ' + (conf * 100).toFixed(1) + '%。';
    g('n').innerHTML = note;
  }
  g('g1').onclick = function(){ shared = false; up(); };
  g('g2').onclick = function(){ shared = true; up(); };
  ['s', 't'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 17. 层次聚类 / GMM ===== */
regDemo(['层次聚类 / GMM', 'GMM', '层次聚类'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: 'GMM · EM 迭代结果', w: 300, h: 250},
            {k: 'b', t: '层次聚类树状图（切割线）', w: 300, h: 250}],
    legend: '<span><i style="background:#c0392b"></i>簇 1</span>'
          + '<span><i style="background:#185fa5"></i>簇 2</span>'
          + '<span><i style="background:#0f6e56"></i>簇 3</span>'
          + '<span><i style="background:#ba7517"></i>簇 4</span>'
          + '<span><i style="background:#7f77dd"></i>簇 5</span>',
    ctl: [{k: 's', label: '簇数 K', min: 2, max: 5, step: 1, val: 3},
          {k: 't', label: 'EM 迭代步数', min: 0, max: 30, step: 1, val: 12}],
    kpi: [{k: 'p', label: '对数似然'}, {k: 'q', label: 'BIC'},
          {k: 'r', label: '层次法簇数'}, {k: 'z', label: '两者一致率'}]
  });
  const g = x => document.getElementById(id + x);
  const COL = ['#c0392b', '#185fa5', '#0f6e56', '#ba7517', '#7f77dd'];
  const rnd = gaussRnd(20241066);
  const X = [];
  [[-1.5, 1.2], [1.5, 1.0], [0, -1.4], [-1.8, -1.4], [1.8, -1.5]].forEach(function(c){
    for(let i = 0; i < 18; i++) X.push([c[0] + 0.42 * rnd(), c[1] + 0.42 * rnd()]);
  });
  const N = X.length, ex = extent(X);
  function em(K, steps){
    let mu = [], sg = [], pi = [];
    for(let k = 0; k < K; k++){
      const p = X[Math.floor(k * N / K) % N];
      mu.push([p[0] + 0.3, p[1] - 0.3]); sg.push(1.1); pi.push(1 / K);
    }
    let ll = 0;
    for(let it = 0; it <= steps; it++){
      const R = [];
      let tot = 0;
      for(let i = 0; i < N; i++){
        const r = [];
        for(let k = 0; k < K; k++){
          const d2 = (X[i][0] - mu[k][0]) * (X[i][0] - mu[k][0]) + (X[i][1] - mu[k][1]) * (X[i][1] - mu[k][1]);
          r.push(pi[k] * Math.exp(-d2 / (2 * sg[k])) / (2 * Math.PI * sg[k]));
        }
        const s = r.reduce((a, b) => a + b, 0) || 1e-12;
        R.push(r.map(v => v / s)); tot += Math.log(s);
      }
      ll = tot;
      if(it === steps) return {mu: mu, sg: sg, pi: pi, R: R, ll: ll};
      for(let k = 0; k < K; k++){
        let nk = 0, sx = 0, sy = 0;
        for(let i = 0; i < N; i++){ nk += R[i][k]; sx += R[i][k] * X[i][0]; sy += R[i][k] * X[i][1]; }
        if(nk < 1e-9) continue;
        mu[k] = [sx / nk, sy / nk];
        let vv = 0;
        for(let i = 0; i < N; i++){
          vv += R[i][k] * ((X[i][0] - mu[k][0]) * (X[i][0] - mu[k][0]) + (X[i][1] - mu[k][1]) * (X[i][1] - mu[k][1]));
        }
        sg[k] = Math.max(0.05, vv / (2 * nk)); pi[k] = nk / N;
      }
    }
    return {mu: mu, sg: sg, pi: pi, R: [], ll: ll};
  }
  const merge = (function(){
    let cl = X.map(function(p, i){ return [i]; });
    const H = [];
    while(cl.length > 1){
      let bi = 0, bj = 1, bd = Infinity;
      for(let i = 0; i < cl.length; i++){
        for(let j = i + 1; j < cl.length; j++){
          let s = 0, c = 0;
          cl[i].forEach(function(a){ cl[j].forEach(function(b){
            s += Math.hypot(X[a][0] - X[b][0], X[a][1] - X[b][1]); c++;
          }); });
          const d = s / c;
          if(d < bd){ bd = d; bi = i; bj = j; }
        }
      }
      H.push({a: cl[bi].slice(), b: cl[bj].slice(), h: bd});
      const nn = cl[bi].concat(cl[bj]);
      cl = cl.filter(function(_, i){ return i !== bi && i !== bj; });
      cl.push(nn);
    }
    return H;
  })();
  function hierLabel(K){
    let cl = X.map(function(p, i){ return [i]; });
    const hs = merge.slice(0, N - K);
    hs.forEach(function(m){
      let ii = -1, jj = -1;
      cl.forEach(function(c, i){
        if(ii < 0 && c.length === m.a.length && c[0] === m.a[0]) ii = i;
        if(jj < 0 && c.length === m.b.length && c[0] === m.b[0]) jj = i;
      });
      if(ii < 0 || jj < 0) return;
      const nn = cl[ii].concat(cl[jj]);
      cl = cl.filter(function(_, i){ return i !== ii && i !== jj; });
      cl.push(nn);
    });
    const lab = new Array(N).fill(0);
    cl.forEach(function(c, i){ c.forEach(function(idx){ lab[idx] = i; }); });
    return lab;
  }
  function drawGMM(cv, r){
    const c = dclr(cv), W = cv.width, H = cv.height;
    const P = x => (x - ex.x0) / (ex.x1 - ex.x0) * W;
    const Q = y => H - (y - ex.y0) / (ex.y1 - ex.y0) * H;
    r.mu.forEach(function(m, k){
      const rad = Math.sqrt(r.sg[k]) * 2;
      const sc = (ex.x1 - ex.x0) / W;
      c.strokeStyle = COL[k]; c.lineWidth = 1.8;
      c.beginPath(); c.arc(P(m[0]), Q(m[1]), Math.abs(rad / sc), 0, 6.284); c.stroke();
      c.fillStyle = COL[k]; c.globalAlpha = 0.13; c.fill(); c.globalAlpha = 1;
    });
    X.forEach(function(p, i){
      let best = 0;
      if(r.R.length){ let bv = -1; r.R[i].forEach(function(v, k){ if(v > bv){ bv = v; best = k; } }); }
      else { let bd = Infinity; r.mu.forEach(function(m, k){
        const d = Math.hypot(p[0] - m[0], p[1] - m[1]); if(d < bd){ bd = d; best = k; } }); }
      c.fillStyle = COL[best % 5];
      c.beginPath(); c.arc(P(p[0]), Q(p[1]), 3.4, 0, 6.284); c.fill();
      c.strokeStyle = '#fff'; c.lineWidth = 1; c.stroke();
    });
  }
  function drawTree(cv, K){
    const c = dclr(cv), W = cv.width, H = cv.height;
    const pos = {}, hmax = merge[merge.length - 1].h || 1;
    X.forEach(function(_, i){ pos[i] = {x: 10 + i / (N - 1) * (W - 24), h: 0}; });
    let nxt = N;
    const Y = h => H - 16 - h / hmax * (H - 30);
    merge.forEach(function(m){
      const pa = pos[m.a[0]], pb = pos[m.b[0]];
      const nx = (pa.x + pb.x) / 2;
      c.strokeStyle = '#b4b2a9'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(pa.x, Y(pa.h)); c.lineTo(pa.x, Y(m.h)); c.lineTo(pb.x, Y(m.h));
      c.lineTo(pb.x, Y(pb.h)); c.stroke();
      pos[m.a[0]] = {x: nx, h: m.h};
      pos[nxt++] = {x: nx, h: m.h};
    });
    const cut = merge[N - K - 1] ? merge[N - K - 1].h : 0;
    c.strokeStyle = '#c0392b'; c.lineWidth = 1.6; c.setLineDash([5, 4]);
    c.beginPath(); c.moveTo(4, Y(cut)); c.lineTo(W - 4, Y(cut)); c.stroke();
    c.setLineDash([]);
    c.fillStyle = '#c0392b'; c.font = '10px sans-serif';
    c.fillText('切割高度 ' + cut.toFixed(2) + ' → ' + K + ' 簇', 6, Y(cut) - 4);
  }
  function up(){
    const K = parseInt(g('s').value, 10), steps = parseInt(g('t').value, 10);
    g('sv').textContent = 'K = ' + K;
    g('tv').textContent = steps;
    const r = em(K, steps);
    drawGMM(g('cva'), r);
    drawTree(g('cvb'), K);
    const bic = -2 * r.ll + (K * 4 - 1) * Math.log(N);
    const hg = hierLabel(K);
    const gm = X.map(function(p, i){
      let best = 0, bv = -Infinity;
      if(r.R.length) r.R[i].forEach(function(v, k){ if(v > bv){ bv = v; best = k; } });
      else r.mu.forEach(function(m, k){
        const d = -Math.hypot(p[0] - m[0], p[1] - m[1]);
        if(d > bv){ bv = d; best = k; }
      });
      return best;
    });
    /* 簇编号是任意的：枚举 K! 种标签对应取最高一致率 */
    function perms(arr){
      if(arr.length <= 1) return [arr];
      const out = [];
      arr.forEach(function(v, i){
        perms(arr.slice(0, i).concat(arr.slice(i + 1))).forEach(function(p){ out.push([v].concat(p)); });
      });
      return out;
    }
    const base = [];
    for(let i = 0; i < K; i++) base.push(i);
    let agree = 0;
    perms(base).forEach(function(pm){
      let ag = 0;
      for(let i = 0; i < N; i++) if(pm[gm[i]] === hg[i]) ag++;
      if(ag > agree) agree = ag;
    });
    g('p').textContent = r.ll.toFixed(1);
    g('q').textContent = bic.toFixed(1);
    g('r').textContent = K;
    g('z').textContent = (agree / N * 100).toFixed(0) + '%';
    g('zw').className = 'kpi' + (agree / N > 0.8 ? ' ok' : ' warn');
    let note;
    if(steps === 0) note = '第 0 步：高斯圆还<b>堆在数据中央</b>，尚未分开。'
      + '向右拖动看 EM 的 E 步（算归属概率）与 M 步（更新圆心/方差）如何交替把圆推到位。';
    else if(steps < 6) note = '才 <b>' + steps + '</b> 步：圆还在往各自的团挪。'
      + 'EM 是<b>软分配</b>——每个点以一定概率属于所有簇，不像 K-Means 硬切。';
    else note = '<b>' + steps + '</b> 步后对数似然 <b>' + r.ll.toFixed(1) + '</b>（越大越好），'
      + 'BIC ' + bic.toFixed(0) + '（越小越好）。GMM 软分配与层次法硬切割的一致率 <b>'
      + (agree / N * 100).toFixed(0) + '%</b>：两种思路殊途同归，差别主要在簇重叠处。';
    g('n').innerHTML = note;
  }
  ['s', 't'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 18. 卷积神经网络 CNN ===== */
regDemo(['卷积神经网络 CNN', 'CNN'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '输入图像（48×48 合成纹理）', w: 240, h: 240},
            {k: 'b', t: '特征图（卷积输出）', w: 240, h: 240},
            {k: 'c', t: '卷积核数值（3×3）', w: 240, h: 240}],
    legend: '<span><i style="background:#185fa5"></i>正激活</span>'
          + '<span><i style="background:#c0392b"></i>负激活</span>'
          + '<span><i style="background:#fff;border:1px solid #d3d1c7"></i>接近 0</span>',
    ctl: [{k: 's', label: '显示增益', min: 0.3, max: 4, step: 0.1, val: 1.4},
          {k: 't', label: 'ReLU 阈值', min: 0, max: 0.6, step: 0.02, val: 0}],
    btns: [{k: 'g1', label: '边缘'}, {k: 'g2', label: '竖直'}, {k: 'g3', label: '水平'},
           {k: 'g4', label: '锐化'}, {k: 'g5', label: '模糊'}, {k: 'g6', label: '浮雕'}],
    kpi: [{k: 'p', label: '输出尺寸'}, {k: 'q', label: '平均 |激活|'},
          {k: 'r', label: '强响应占比'}, {k: 'z', label: '参数量'}]
  });
  const g = x => document.getElementById(id + x);
  const KER = [
    {n: '边缘（Laplacian）', k: [0, -1, 0, -1, 4, -1, 0, -1, 0]},
    {n: '竖直边缘（Sobel-X）', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1]},
    {n: '水平边缘（Sobel-Y）', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1]},
    {n: '锐化', k: [0, -1, 0, -1, 5, -1, 0, -1, 0]},
    {n: '模糊（均值）', k: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9]},
    {n: '浮雕', k: [-2, -1, 0, -1, 1, 1, 0, 1, 2]}
  ];
  let ki = 0;
  const M = 48, IM = [];
  for(let i = 0; i < M; i++){
    IM.push([]);
    for(let j = 0; j < M; j++){
      const dx = i - 17, dy = j - 17, r = Math.hypot(dx, dy);
      let v = r < 11 ? 0.8 : 0.25;
      v += 0.22 * Math.sin((i * 2 + j * 3) * 0.32);
      if(j > 33 && i < 22) v = 0.12;
      if(i > 36 && j < 14) v = 0.95;
      v += 0.05 * Math.sin(i * 12.9898 + j * 78.233);
      IM[i].push(Math.max(0, Math.min(1, v)));
    }
  }
  function conv(ker){
    const O = [];
    let mx = 0;
    for(let i = 1; i < M - 1; i++){
      O.push([]);
      for(let j = 1; j < M - 1; j++){
        let s = 0;
        for(let a = -1; a <= 1; a++) for(let b = -1; b <= 1; b++) s += IM[i + a][j + b] * ker[(a + 1) * 3 + (b + 1)];
        O[i - 1].push(s); mx = Math.max(mx, Math.abs(s));
      }
    }
    return {O: O, mx: mx};
  }
  function drawImg(cv, mat, sym){
    const c = dclr(cv), W = cv.width, H = cv.height;
    const R = mat.length, C = mat[0].length, s = W / C;
    let mx = 0;
    mat.forEach(function(row){ row.forEach(function(v){ mx = Math.max(mx, Math.abs(v)); }); });
    mx = mx || 1;
    for(let i = 0; i < R; i++) for(let j = 0; j < C; j++){
      const v = mat[i][j] / mx;
      if(sym) c.fillStyle = v > 0 ? 'rgba(24,95,165,' + Math.min(1, Math.abs(v)) + ')'
                                  : 'rgba(192,57,43,' + Math.min(1, Math.abs(v)) + ')';
      else { const t = Math.round(255 * (1 - v)); c.fillStyle = 'rgb(' + t + ',' + t + ',' + t + ')'; }
      c.fillRect(j * s, i * s, s + 0.6, s + 0.6);
    }
  }
  function drawKer(cv, ker){
    const c = dclr(cv), W = cv.width, H = cv.height;
    const s = Math.min(W, H) / 3;
    for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
      const v = ker[i * 3 + j];
      const a = Math.min(0.92, Math.abs(v) / 4.2);
      c.fillStyle = v >= 0 ? 'rgba(24,95,165,' + a + ')' : 'rgba(192,57,43,' + a + ')';
      c.fillRect(j * s, i * s, s - 1, s - 1);
      c.fillStyle = Math.abs(v) > 2 ? '#fff' : '#2c2c2a';
      c.font = '13px sans-serif'; c.textAlign = 'center';
      c.fillText(v.toFixed(Math.abs(v) < 1 ? 2 : 1), j * s + s / 2, i * s + s / 2 + 5);
    }
    c.textAlign = 'left';
    c.fillStyle = '#5f5e5a'; c.font = '12px sans-serif';
    c.fillText(KER[ki].n, 6, H - 8);
  }
  function up(){
    const gain = parseFloat(g('s').value), th = parseFloat(g('t').value);
    g('sv').textContent = '×' + gain.toFixed(1);
    g('tv').textContent = th.toFixed(2);
    [0, 1, 2, 3, 4, 5].forEach(function(i){
      const b = g('g' + (i + 1));
      b.style.background = i === ki ? '#185fa5' : '#fff';
      b.style.color = i === ki ? '#fff' : '#185fa5';
    });
    drawImg(g('cva'), IM, false);
    const res = conv(KER[ki].k);
    const O = res.O.map(function(row){ return row.map(function(v){ return v * gain; }); });
    drawImg(g('cvb'), O, true);
    drawKer(g('cvc'), KER[ki].k);
    const flat = [];
    O.forEach(function(row){ row.forEach(function(v){ flat.push(Math.abs(v)); }); });
    const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    const strong = flat.filter(v => v > 0.25 * Math.max.apply(null, flat)).length / flat.length;
    const relued = O.map(function(row){ return row.map(v => Math.max(0, v - th)); });
    let nz = 0, tot = 0;
    relued.forEach(function(row){ row.forEach(function(v){ tot++; if(v > 1e-6) nz++; }); });
    g('p').textContent = O.length + '×' + O[0].length;
    g('q').textContent = mean.toFixed(3);
    g('r').textContent = (strong * 100).toFixed(0) + '%';
    g('z').textContent = 9;
    g('zw').className = 'kpi ok';
    g('n').innerHTML = '<b>' + KER[ki].n + '</b>：整张图共用这一组 <b>9</b> 个权重（参数量与图像尺寸无关，'
      + '这就是<b>权值共享</b>）。有效响应占 <b>' + (strong * 100).toFixed(0) + '%</b>，'
      + 'ReLU 后仍有 <b>' + (nz / tot * 100).toFixed(0) + '%</b> 的像素存活——'
      + '把它拉到 0 再对比，可以直观看到稀疏化。真实 CNN 里这一层的输出会再送给下一个核，逐层从「边缘」抽象到「部件」。';
  }
  [0, 1, 2, 3, 4, 5].forEach(function(i){ g('g' + (i + 1)).onclick = function(){ ki = i; up(); }; });
  ['s', 't'].forEach(function(k){ g(k).oninput = up; });
  up();
});
/* DEMOS_B2:END */
