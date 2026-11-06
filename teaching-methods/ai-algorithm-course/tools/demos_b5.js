/* DEMOS_B5:BEGIN */
/* ===== 25. AdaBoost ===== */
regDemo(['AdaBoost'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '集成边界（点越大＝权重越大）', w: 320, h: 260},
            {k: 'b', t: '训练 / 测试误差随轮数变化', w: 320, h: 260}],
    legend: '<span><i style="background:#c0392b"></i>类别 −1</span>'
          + '<span><i style="background:#0f6e56"></i>类别 +1</span>'
          + '<span><i style="background:#185fa5"></i>集成决策边界</span>',
    ctl: [{k: 's', label: '弱分类器轮数 M', min: 1, max: 60, step: 1, val: 12},
          {k: 't', label: '数据噪声', min: 0, max: 1.2, step: 0.05, val: 0.35}],
    kpi: [{k: 'p', label: '训练误差'}, {k: 'q', label: '测试误差'},
          {k: 'r', label: '最重样本权重'}, {k: 'z', label: 'Σα'}]
  });
  const g = x => document.getElementById(id + x);
  function makeData(ns, seed){
    const r = gaussRnd(seed);
    const A = [], B = [];
    for(let i = 0; i < 55; i++) A.push([-0.9 + (0.45 + ns * 0.5) * r(), 0.7 + (0.45 + ns * 0.5) * r()]);
    for(let i = 0; i < 55; i++) B.push([0.9 + (0.45 + ns * 0.5) * r(), -0.7 + (0.45 + ns * 0.5) * r()]);
    return {X: A.concat(B), y: A.map(() => -1).concat(B.map(() => 1))};
  }
  function stump(X, y, w){
    let best = null;
    for(let f = 0; f < 2; f++){
      const vs = X.map((p, i) => [p[f], i]).sort((a, b) => a[0] - b[0]);
      for(let i = 0; i < vs.length - 1; i += 1){
        const th = (vs[i][0] + vs[i + 1][0]) / 2;
        for(let dir = 0; dir < 2; dir++){
          let err = 0;
          for(let k = 0; k < X.length; k++){
            const h = dir === 0 ? (X[k][f] < th ? -1 : 1) : (X[k][f] < th ? 1 : -1);
            if(h !== y[k]) err += w[k];
          }
          if(!best || err < best.err) best = {f: f, th: th, dir: dir, err: err};
        }
      }
    }
    return best;
  }
  const pv = (st, p) => (st.dir === 0 ? (p[st.f] < st.th ? -1 : 1) : (p[st.f] < st.th ? 1 : -1));
  function up(){
    const M = parseInt(g('s').value, 10), ns = parseFloat(g('t').value);
    g('sv').textContent = M;
    g('tv').textContent = ns.toFixed(2);
    const tr = makeData(ns, 20241144), te = makeData(ns, 20241155);
    const N = tr.X.length;
    let w = new Array(N).fill(1 / N);
    const al = [], sts = [], htr = [], hte = [];
    for(let m = 0; m < M; m++){
      const st = stump(tr.X, tr.y, w);
      const e = Math.max(1e-6, Math.min(1 - 1e-6, st.err));
      const a = 0.5 * Math.log((1 - e) / e);
      let z = 0;
      for(let i = 0; i < N; i++){
        w[i] = w[i] * Math.exp(-a * tr.y[i] * pv(st, tr.X[i]));
        z += w[i];
      }
      for(let i = 0; i < N; i++) w[i] /= z;
      al.push(a); sts.push(st);
      const F = (X) => X.map(p => {
        let s = 0;
        for(let k = 0; k < al.length; k++) s += al[k] * pv(sts[k], p);
        return s > 0 ? 1 : -1;
      });
      const pt = F(tr.X), pe = F(te.X);
      htr.push(pt.reduce((s, v, i) => s + (v !== tr.y[i] ? 1 : 0), 0) / N);
      hte.push(pe.reduce((s, v, i) => s + (v !== te.y[i] ? 1 : 0), 0) / te.y.length);
    }
    const F = p => {
      let s = 0;
      for(let k = 0; k < al.length; k++) s += al[k] * pv(sts[k], p);
      return s > 0 ? 1 : -1;
    };
    const ex = extent(tr.X.concat(te.X));
    const cv = g('cva'), c = dclr(cv), W = cv.width, H = cv.height, G = 6;
    for(let py = 0; py < H; py += G){
      for(let px = 0; px < W; px += G){
        const wx = ex.x0 + (px + 0.5) / W * (ex.x1 - ex.x0);
        const wy = ex.y0 + (1 - (py + 0.5) / H) * (ex.y1 - ex.y0);
        const v = F([wx, wy]);
        c.fillStyle = v > 0 ? 'rgba(15,110,86,.13)' : 'rgba(192,57,43,.13)';
        c.fillRect(px, py, G, G);
      }
    }
    const wmax = Math.max.apply(null, w) || 1;
    tr.X.forEach(function(p, i){
      const px = (p[0] - ex.x0) / (ex.x1 - ex.x0) * W, py = H - (p[1] - ex.y0) / (ex.y1 - ex.y0) * H;
      const rr = 2 + 7 * Math.min(1, w[i] / wmax);
      c.fillStyle = tr.y[i] > 0 ? 'rgba(15,110,86,.95)' : 'rgba(192,57,43,.95)';
      c.beginPath(); c.arc(px, py, rr, 0, 6.284); c.fill();
      c.strokeStyle = '#fff'; c.lineWidth = 1; c.stroke();
    });
    sts.forEach(function(st){
      c.strokeStyle = 'rgba(24,95,165,.35)'; c.lineWidth = 1;
      if(st.f === 0){ const px = (st.th - ex.x0) / (ex.x1 - ex.x0) * W; c.beginPath(); c.moveTo(px, 0); c.lineTo(px, H); c.stroke(); }
      else { const py = H - (st.th - ex.y0) / (ex.y1 - ex.y0) * H; c.beginPath(); c.moveTo(0, py); c.lineTo(W, py); c.stroke(); }
    });
    const cv2 = g('cvb'), c2 = dclr(cv2), W2 = cv2.width, H2 = cv2.height;
    c2.strokeStyle = '#eceae2'; c2.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H2 - 22 - i / 4 * (H2 - 34);
      c2.beginPath(); c2.moveTo(30, yy); c2.lineTo(W2 - 8, yy); c2.stroke();
    }
    [[htr, '#185fa5'], [hte, '#c0392b']].forEach(function(pr){
      c2.strokeStyle = pr[1]; c2.lineWidth = 2; c2.beginPath();
      pr[0].forEach(function(v, i){
        const px = 30 + i / Math.max(1, pr[0].length - 1) * (W2 - 38), py = H2 - 22 - v / 0.5 * (H2 - 34);
        i ? c2.lineTo(px, py) : c2.moveTo(px, py);
      });
      c2.stroke();
    });
    c2.fillStyle = '#5f5e5a'; c2.font = '10px sans-serif';
    c2.fillText('蓝=训练 红=测试', 32, 12);
    c2.fillText('50%', 2, H2 - 22); c2.fillText('0%', 2, 10);
    const sa = al.reduce((a, b) => a + b, 0);
    g('p').textContent = (htr[M - 1] * 100).toFixed(1) + '%';
    g('q').textContent = (hte[M - 1] * 100).toFixed(1) + '%';
    g('r').textContent = (Math.max.apply(null, w) * N).toFixed(1) + '×';
    g('z').textContent = sa.toFixed(2);
    g('qw').className = 'kpi' + (hte[M - 1] < 0.12 ? ' ok' : ' warn');
    let note;
    if(M === 1) note = 'M=1：只有<b>一个树桩</b>（一条水平或竖直线）。'
      + 'AdaBoost 的弱分类器必须「比瞎猜好一点点」就行—— err&lt;0.5 即可，α=½ln((1−err)/err)。';
    else if(ns > 0.8) note = '噪声 <b>' + ns.toFixed(2) + '</b>：有几怎么分都分不开的点。'
      + '它们的权重被一路放大（最重已达 <b>' + (Math.max.apply(null, w) * N).toFixed(1)
      + '×</b> 均匀权重），AdaBoost 会<b>死磕噪声</b>——这是它对离群点敏感的根源。';
    else note = '<b>' + M + '</b> 轮：训练误差 <b>' + (htr[M - 1] * 100).toFixed(1)
      + '%</b>、测试误差 <b>' + (hte[M - 1] * 100).toFixed(1) + '%</b>。'
      + '每轮把分错的样本权重放大，下一个树桩就会<b>优先去啃硬骨头</b>；'
      + '半透明的线是历代树桩，它们的加权投票（Σα=' + sa.toFixed(1) + '）拼出了弯折的边界。';
    g('n').innerHTML = note;
  }
  ['s', 't'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 26. 循环网络 RNN / LSTM ===== */
regDemo(['循环网络 RNN / LSTM', 'RNN', 'LSTM'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: 'RNN：梯度回传 t 步后还剩多少', w: 300, h: 230},
            {k: 'b', t: 'LSTM：同一把尺子下的对照', w: 300, h: 230},
            {k: 'c', t: 'RNN vs LSTM（当前参数）', w: 300, h: 230}],
    legend: '<span><i style="background:#185fa5"></i>当前 RNN 参数</span>'
          + '<span><i style="background:#0f6e56"></i>当前 LSTM 参数</span>'
          + '<span><i style="background:#b4b2a9"></i>其它参数取值</span>'
          + '<span><i style="background:#c0392b"></i>可用梯度下限</span>',
    ctl: [{k: 's', label: '序列长度 T', min: 4, max: 60, step: 1, val: 20},
          {k: 't', label: 'RNN 循环权重 |w_h|', min: 0.3, max: 2.2, step: 0.05, val: 0.9},
          {k: 'u', label: 'LSTM 遗忘门 f', min: 0.3, max: 1, step: 0.01, val: 0.95}],
    kpi: [{k: 'p', label: 'RNN 末端梯度'}, {k: 'q', label: 'LSTM 末端梯度'},
          {k: 'r', label: '相差倍数'}, {k: 'z', label: 'RNN 半衰步数'}]
  });
  const g = x => document.getElementById(id + x);
  const NSE = (function(){
    let st = 20241166 >>> 0;
    const u = function(){ st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
    const a = [];
    for(let i = 0; i < 80; i++) a.push(0.6 * (u() - 0.5));
    return a;
  })();
  /* RNN：h_t = tanh(w_h·h_{t-1} + x_t)，∂h_T/∂h_t = ∏ w_h(1−h_s²) */
  function rnnDecay(wh, T){
    const h = [0];
    for(let t = 0; t < T; t++) h.push(Math.tanh(wh * h[t] + NSE[t % 80]));
    const out = [];
    let k = 1;
    for(let t = T; t >= 1; t--){
      k *= Math.abs(wh * (1 - h[t] * h[t]));
      out.push(k);
    }
    return out;
  }
  /* LSTM：c_t = f_t·c_{t-1} + i_t·g_t，∂c_T/∂c_t = ∏ f_s（加性路径，不乘 tanh'） */
  function lstmDecay(f, T){
    const out = [];
    let k = 1;
    for(let t = T - 1; t >= 0; t--){
      k *= Math.max(0.02, Math.min(1, f + 0.06 * NSE[t % 80]));
      out.push(k);
    }
    return out;
  }
  function grid(c, W, H, lo, hi){
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    for(let i = 0; i <= 9; i++){
      const yy = H - 24 - i / 9 * (H - 38);
      c.beginPath(); c.moveTo(34, yy); c.lineTo(W - 8, yy); c.stroke();
    }
    c.strokeStyle = '#c0392b'; c.lineWidth = 1; c.setLineDash([3, 3]);
    const yFloor = H - 24 - (Math.log10(1e-6) - lo) / (hi - lo) * (H - 38);
    if(yFloor > 8 && yFloor < H - 24){ c.beginPath(); c.moveTo(34, yFloor); c.lineTo(W - 8, yFloor); c.stroke(); }
    c.setLineDash([]);
    c.fillStyle = '#888780'; c.font = '10px sans-serif';
    c.fillText('10⁰', 2, 12);
    c.fillText('10⁻' + Math.round(hi - lo), 2, H - 26);
  }
  const LO = -6, HI = 1;
  function plot(cv, series, cur){
    const c = dclr(cv), W = cv.width, H = cv.height;
    grid(c, W, H, LO, HI);
    series.forEach(function(s){
      const isCur = s.v === cur;
      c.strokeStyle = isCur ? s.col : '#d3d1c7';
      c.lineWidth = isCur ? 2.6 : 1.2;
      c.beginPath();
      s.d.forEach(function(val, i){
        const lg = Math.max(LO, Math.min(HI, Math.log10(Math.max(1e-12, val))));
        const px = 34 + i / Math.max(1, s.d.length - 1) * (W - 42);
        const py = H - 24 - (lg - LO) / (HI - LO) * (H - 38);
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      });
      c.stroke();
    });
  }
  function up(){
    const T = parseInt(g('s').value, 10);
    const wh = parseFloat(g('t').value), f = parseFloat(g('u').value);
    g('sv').textContent = 'T = ' + T;
    g('tv').textContent = wh.toFixed(2);
    g('uv').textContent = f.toFixed(2);
    const rWh = [0.5, 0.9, 1.0, 1.5, 2.0].concat([wh]);
    const sR = rWh.map(function(v){ return {v: v, d: rnnDecay(v, T), col: '#185fa5'}; });
    plot(g('cva'), sR, wh);
    const rF = [0.5, 0.8, 0.95, 1.0].concat([f]);
    const sL = rF.map(function(v){ return {v: v, d: lstmDecay(v, T), col: '#0f6e56'}; });
    plot(g('cvb'), sL, f);
    const cv3 = g('cvc'), c3 = dclr(cv3), W3 = cv3.width, H3 = cv3.height;
    grid(c3, W3, H3, LO, HI);
    [[rnnDecay(wh, T), '#185fa5', 'RNN |w_h|=' + wh.toFixed(2)],
     [lstmDecay(f, T), '#0f6e56', 'LSTM f=' + f.toFixed(2)]].forEach(function(pr){
      c3.strokeStyle = pr[1]; c3.lineWidth = 2.4; c3.beginPath();
      pr[0].forEach(function(val, i){
        const lg = Math.max(LO, Math.min(HI, Math.log10(Math.max(1e-12, val))));
        const px = 34 + i / Math.max(1, pr[0].length - 1) * (W3 - 42);
        const py = H3 - 24 - (lg - LO) / (HI - LO) * (H3 - 38);
        i ? c3.lineTo(px, py) : c3.moveTo(px, py);
      });
      c3.stroke();
      c3.fillStyle = pr[1]; c3.font = '10px sans-serif';
      c3.fillText(pr[2], 40, 12 + (pr[1] === '#185fa5' ? 0 : 12));
    });
    const rd = rnnDecay(wh, T), ld = lstmDecay(f, T);
    const re = rd[rd.length - 1], le = ld[ld.length - 1];
    let half = T;
    for(let i = 0; i < rd.length; i++) if(rd[i] < 0.5){ half = i + 1; break; }
    g('p').textContent = re.toExponential(1);
    g('q').textContent = le.toExponential(1);
    g('r').textContent = (le / Math.max(1e-12, re)).toExponential(1) + '×';
    g('z').textContent = half + ' 步';
    g('pw').className = 'kpi' + (re < 1e-4 ? ' warn' : ' ok');
    g('qw').className = 'kpi' + (le > 1e-2 ? ' ok' : ' warn');
    let note;
    if(wh < 0.75) note = '|w_h|=<b>' + wh.toFixed(2) + '</b>：每回传一步就乘一个小于 1 的数，'
      + 'T=' + T + ' 步后只剩 <b>' + re.toExponential(1) + '</b>（半衰期仅 <b>' + half
      + '</b> 步）。早期时刻的梯度<b>根本传不到输出端</b>——这就是梯度消失。';
    else if(wh > 1.35) note = '|w_h|=<b>' + wh.toFixed(2) + '</b>：大于 1，梯度会<b>指数爆炸</b>。'
      + '实践中靠梯度裁剪硬截断，代价是更新方向被破坏。'
      + '所以简单 RNN 只在 |w_h|≈1 附近很窄的区间里能用。';
    else if(f > 0.97) note = 'f=<b>' + f.toFixed(2) + '</b>：遗忘门几乎全开，'
      + '梯度沿细胞状态<b>近似无损</b>地传了 ' + T + ' 步（剩 <b>' + le.toExponential(1)
      + '</b>）。对比 RNN 的 <b>' + re.toExponential(1) + '</b>，相差 <b>'
      + (le / Math.max(1e-12, re)).toExponential(1) + '</b> 倍。';
    else note = 'LSTM 的关键不是「记忆更好」，而是<b>换了一条梯度路径</b>：'
      + 'c_t = f·c<sub>t−1</sub> + i·g 是<b>加性</b>的，'
      + '∂c_T/∂c_t 只连乘遗忘门 <b>f=' + f.toFixed(2) + '</b>，不乘 tanh′（RNN 那个 (1−h²) 才是元凶）。'
      + '把 f 调到 0.9 以上，曲线几乎打平。';
    g('n').innerHTML = note;
  }
  ['s', 't', 'u'].forEach(function(k){ g(k).oninput = up; });
  up();
});

/* ===== 27. Transformer / 注意力 ===== */
regDemo(['Transformer / 注意力', 'Transformer', '注意力'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '注意力权重矩阵（当前头）', w: 300, h: 250},
            {k: 'b', t: '当前 query 的注意力分布', w: 300, h: 250}],
    legend: '<span><i style="background:#f2f0ea"></i>权重 0</span>'
          + '<span><i style="background:#185fa5"></i>权重 1</span>'
          + '<span><i style="background:#c0392b"></i>当前 query 行</span>',
    ctl: [{k: 's', label: 'query 位置', min: 0, max: 5, step: 1, val: 5},
          {k: 't', label: '注意力头', min: 0, max: 3, step: 1, val: 0},
          {k: 'u', label: 'softmax 温度 τ', min: 0.15, max: 3, step: 0.05, val: 1}],
    btns: [{k: 'g1', label: '因果掩码（解码器）'}],
    kpi: [{k: 'p', label: '关注对象'}, {k: 'q', label: '最大权重'},
          {k: 'r', label: '分布熵 (bit)'}, {k: 'z', label: '掩码'}]
  });
  const g = x => document.getElementById(id + x);
  const TOK = ['我', '昨天', '买', '的', '手机', '坏了'];
  const N = TOK.length;
  const HD = ['语义依存', '前一词', '自身', '句首'];
  let masked = false;
  let st = 20241177 >>> 0;
  const ur = function(){ st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  const base = [];
  for(let i = 0; i < N; i++){
    base.push([]);
    for(let j = 0; j < N; j++) base[i].push(0.7 * (ur() - 0.5));
  }
  function bias(h, i, j){
    if(h === 0){
      if(i === 5 && j === 4) return 3.2;
      if(i === 2 && j === 4) return 1.6;
      if(i === 0 && j === 1) return 0.8;
      return 0;
    }
    if(h === 1) return j === i - 1 ? 3.0 : 0;
    if(h === 2) return j === i ? 2.6 : 0;
    return j === 0 ? (i === 0 ? 1.2 : 2.2) : 0;
  }
  function attn(h, tau){
    const A = [];
    for(let i = 0; i < N; i++){
      const row = [];
      for(let j = 0; j < N; j++){
        let v = base[i][j] + bias(h, i, j);
        if(masked && j > i) v = -1e9;
        row.push(v / tau);
      }
      const mx = Math.max.apply(null, row);
      const ex = row.map(v => Math.exp(v - mx));
      const s = ex.reduce((a, b) => a + b, 0);
      A.push(ex.map(v => v / s));
    }
    return A;
  }
  function up(){
    const qi = parseInt(g('s').value, 10), h = parseInt(g('t').value, 10), tau = parseFloat(g('u').value);
    g('sv').textContent = TOK[qi];
    g('tv').textContent = 'H' + h + ' ' + HD[h];
    g('uv').textContent = tau.toFixed(2);
    g('g1').style.background = masked ? '#185fa5' : '#fff';
    g('g1').style.color = masked ? '#fff' : '#185fa5';
    const A = attn(h, tau);
    const cv = g('cva'), c = dclr(cv), W = cv.width, H = cv.height;
    const pad = 34, cell = Math.min((W - pad - 8) / N, (H - pad - 8) / N);
    c.font = '11px sans-serif';
    for(let i = 0; i < N; i++){
      for(let j = 0; j < N; j++){
        const v = A[i][j];
        c.fillStyle = 'rgba(24,95,165,' + (0.05 + 0.95 * Math.min(1, v * 1.15)) + ')';
        c.fillRect(pad + j * cell, pad + i * cell, cell - 1.5, cell - 1.5);
        if(i === qi){
          c.strokeStyle = '#c0392b'; c.lineWidth = 1.6;
          c.strokeRect(pad + j * cell, pad + i * cell, cell - 1.5, cell - 1.5);
        }
        if(cell > 26){
          c.fillStyle = v > 0.5 ? '#fff' : '#444441';
          c.textAlign = 'center';
          c.fillText(v.toFixed(2).slice(1), pad + j * cell + cell / 2, pad + i * cell + cell / 2 + 4);
        }
      }
    }
    c.textAlign = 'right'; c.fillStyle = '#5f5e5a';
    for(let i = 0; i < N; i++) c.fillText(TOK[i], pad - 4, pad + i * cell + cell / 2 + 4);
    c.textAlign = 'center';
    for(let j = 0; j < N; j++) c.fillText(TOK[j], pad + j * cell + cell / 2, pad - 8);
    c.textAlign = 'left';
    const cv2 = g('cvb'), c2 = dclr(cv2), W2 = cv2.width, H2 = cv2.height;
    const bw = (W2 - 40) / N;
    const mxv = Math.max.apply(null, A[qi]) || 1;
    A[qi].forEach(function(v, j){
      const hgt = v / 1.05 * (H2 - 56);
      c2.fillStyle = j === qi ? '#c0392b' : 'rgba(24,95,165,' + (0.35 + 0.6 * v / mxv) + ')';
      c2.fillRect(30 + j * bw + 4, H2 - 30 - hgt, bw - 8, hgt);
      c2.fillStyle = '#2c2c2a'; c2.font = '12px sans-serif'; c2.textAlign = 'center';
      c2.fillText(TOK[j], 30 + j * bw + bw / 2, H2 - 12);
      c2.fillStyle = '#5f5e5a'; c2.font = '10px sans-serif';
      c2.fillText((v * 100).toFixed(0) + '%', 30 + j * bw + bw / 2, H2 - 34 - hgt);
    });
    c2.textAlign = 'left';
    c2.fillStyle = '#2c2c2a'; c2.font = '12px sans-serif';
    c2.fillText('「' + TOK[qi] + '」在关注谁（头 H' + h + '：' + HD[h] + '）', 26, 16);
    let ent = 0;
    A[qi].forEach(function(v){ if(v > 1e-9) ent -= v * Math.log2(v); });
    let bi = 0;
    A[qi].forEach(function(v, j){ if(v > A[qi][bi]) bi = j; });
    g('p').textContent = TOK[bi];
    g('q').textContent = (A[qi][bi] * 100).toFixed(0) + '%';
    g('r').textContent = ent.toFixed(2);
    g('z').textContent = masked ? '开' : '关';
    let note;
    if(tau < 0.4) note = 'τ=<b>' + tau.toFixed(2) + '</b>：softmax 被「冻住」，分布几乎变成 <b>one-hot</b>——'
      + '只看一个词。温度太低会让梯度变陡、训练不稳。';
    else if(tau > 2) note = 'τ=<b>' + tau.toFixed(2) + '</b>：分布被抹平，熵升到 <b>' + ent.toFixed(2)
      + ' bit</b>（接近均匀分布的 ' + Math.log2(N).toFixed(2) + ' bit）。'
      + '注意力变平均 = <b>什么都关注＝什么都没关注</b>。';
    else if(masked) note = '开启<b>因果掩码</b>：上三角被置为 −∞，第 i 个词只能看自己和之前的词。'
      + '这是解码器自回归生成的关键——否则模型会「偷看答案」。';
    else note = '头 <b>H' + h + '（' + HD[h] + '）</b>：「' + TOK[qi] + '」把 <b>'
      + (A[qi][bi] * 100).toFixed(0) + '%</b> 的权重给了「' + TOK[bi] + '」。'
      + '真实的 Transformer 每个头学到的模式不同（切到 H1/H2/H3 看差别），'
      + '多头拼接后再过一层线性变换——这就是「注意力」能替代循环结构的原因：'
      + '<b>任意两个位置的距离都是 1</b>，梯度不必再沿时间步连乘。';
    g('n').innerHTML = note;
  }
  g('g1').onclick = function(){ masked = !masked; up(); };
  ['s', 't', 'u'].forEach(function(k){ g(k).oninput = up; });
  up();
});
/* DEMOS_B5:END */
