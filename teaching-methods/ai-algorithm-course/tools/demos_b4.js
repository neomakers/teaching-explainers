/* DEMOS_B4:BEGIN */
/* ===== 22. t-SNE / UMAP ===== */
regDemo(['t-SNE / UMAP', 'Manifold Visualization', 't-SNE', 'UMAP'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '原始空间前 2 维', w: 260, h: 230},
            {k: 'b', t: 'PCA 线性投影', w: 260, h: 230},
            {k: 'c', t: 't-SNE 嵌入（迭代展开）', w: 260, h: 230}],
    legend: '<span><i style="background:#c0392b"></i>簇 A</span>'
          + '<span><i style="background:#185fa5"></i>簇 B</span>'
          + '<span><i style="background:#0f6e56"></i>簇 C</span>',
    ctl: [{k: 's', label: '困惑度 perplexity', min: 2, max: 40, step: 1, val: 12},
          {k: 't', label: '迭代步数（自动播放中）', min: 0, max: 1000, step: 1, val: 0}],
    btns: [{k: 'g1', label: '▶ 播放'}, {k: 'g2', label: '↺ 重置'}],
    kpi: [{k: 'p', label: 'KL 散度'}, {k: 'q', label: '迭代步数'},
          {k: 'r', label: '近邻保持率'}, {k: 'z', label: 'PCA 对比'}]
  });
  const g = x => document.getElementById(id + x);
  const COLS = ['#c0392b', '#185fa5', '#0f6e56'];
  const rnd = gaussRnd(20241110);
  const X = [], lab = [];
  [[0, 0, 0, 0], [2.6, 2.2, -1.4, 0.6], [-2.2, 1.8, 2.4, -1.2]].forEach(function(c, ci){
    for(let i = 0; i < 50; i++){
      X.push([c[0] + 0.55 * rnd(), c[1] + 0.55 * rnd(), c[2] + 0.55 * rnd(), c[3] + 0.55 * rnd()]);
      lab.push(ci);
    }
  });
  const n = X.length, DIM = 4;
  const D2 = [];
  for(let i = 0; i < n; i++){
    D2.push(new Float64Array(n));
    for(let j = 0; j < n; j++){
      if(j === i) continue;
      let s = 0;
      for(let d = 0; d < DIM; d++){ const t = X[i][d] - X[j][d]; s += t * t; }
      D2[i][j] = s;
    }
  }
  function affP(perp){
    const P = [], tgt = Math.log(perp);
    for(let i = 0; i < n; i++){
      let lo = 1e-12, hi = 1e7;
      for(let it = 0; it < 50; it++){
        const be = (lo + hi) / 2;
        let s = 0;
        for(let j = 0; j < n; j++) if(j !== i) s += Math.exp(-be * D2[i][j]);
        let H = 0;
        for(let j = 0; j < n; j++) if(j !== i){
          const p = Math.exp(-be * D2[i][j]) / (s || 1e-30);
          if(p > 1e-30) H -= p * Math.log(p);
        }
        if(H > tgt) lo = be; else hi = be;
      }
      const be = (lo + hi) / 2;
      let s = 0;
      for(let j = 0; j < n; j++) if(j !== i) s += Math.exp(-be * D2[i][j]);
      const row = new Float64Array(n);
      for(let j = 0; j < n; j++) if(j !== i) row[j] = Math.exp(-be * D2[i][j]) / (s || 1e-30);
      P.push(row);
    }
    for(let i = 0; i < n; i++) for(let j = 0; j < n; j++) P[i][j] = (P[i][j] + P[j][i]) / (2 * n);
    return P;
  }
  function pca2(){
    const mu = [];
    for(let d = 0; d < DIM; d++) mu.push(X.reduce(function(s, p){ return s + p[d]; }, 0) / n);
    const Cc = X.map(p => p.map((v, d) => v - mu[d]));
    const S = [[0, 0], [0, 0]];
    Cc.forEach(function(p){
      S[0][0] += p[0] * p[0]; S[1][1] += p[1] * p[1]; S[0][1] += p[0] * p[1];
    });
    S[0][0] /= n; S[1][1] /= n; S[0][1] /= n; S[1][0] = S[0][1];
    const eg = eigen2(S);
    return Cc.map(p => [p[0] * eg.e1[0] + p[1] * eg.e1[1], p[0] * 0 + p[1] * 0]);
  }
  function scatter(cv, pts, byLab){
    const c = dclr(cv), W = cv.width, H = cv.height;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    pts.forEach(function(p){
      x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]);
      y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
    });
    const px = (x1 - x0) || 1, py = (y1 - y0) || 1;
    const P = x => 14 + (x - x0) / px * (W - 28);
    const Q = y => H - 14 - (y - y0) / py * (H - 28);
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    c.strokeRect(6, 6, W - 12, H - 12);
    pts.forEach(function(p, i){
      c.fillStyle = byLab ? COLS[lab[i]] : '#5f5e5a';
      c.beginPath(); c.arc(P(p[0]), Q(p[1]), 3.2, 0, 6.284); c.fill();
    });
  }
  const raw2 = X.map(p => [p[0], p[1]]);
  let Y = null, iY = null, gains = null, P = null, step = 0, kl = 0, play = false;
  function reset(){
    let st = 20241111 >>> 0;
    const u = function(){ st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296 - 0.5; };
    Y = []; iY = []; gains = [];
    for(let i = 0; i < n; i++){
      Y.push([u() * 0.02, u() * 0.02]);
      iY.push([0, 0]);
      gains.push([1, 1]);
    }
    step = 0; kl = 0;
    P = affP(parseInt(g('s').value, 10));
  }
  function iterate(){
    let sum = 0;
    const num = new Float64Array(n), Qm = [];
    for(let i = 0; i < n; i++) Qm.push(new Float64Array(n));
    for(let i = 0; i < n; i++){
      let s = 0;
      for(let j = 0; j < n; j++){
        if(j === i) continue;
        const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
        const v = 1 / (1 + dx * dx + dy * dy);
        Qm[i][j] = v; s += v;
      }
      for(let j = 0; j < n; j++) Qm[i][j] /= (s || 1e-30);
    }
    const grad = [];
    for(let i = 0; i < n; i++) grad.push([0, 0]);
    for(let i = 0; i < n; i++){
      for(let j = 0; j < n; j++){
        if(j === i) continue;
        const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
        const mul = (P[i][j] - Qm[i][j]) * Qm[i][j] * (1 + dx * dx + dy * dy) * 4;
        grad[i][0] += mul * dx; grad[i][1] += mul * dy;
        if(P[i][j] > 1e-12) sum += P[i][j] * Math.log(P[i][j] / (Qm[i][j] || 1e-30));
      }
    }
    kl = sum;
    for(let i = 0; i < n; i++){
      for(let d = 0; d < 2; d++){
        const same = (grad[i][d] > 0) === (iY[i][d] > 0);
        gains[i][d] = Math.max(0.01, (same ? gains[i][d] * 0.8 : gains[i][d] + 0.2));
        iY[i][d] = 0.9 * iY[i][d] + 0.15 * gains[i][d] * grad[i][d];
        Y[i][d] += iY[i][d];
      }
    }
    step++;
  }
  function knnKeep(pts, K){
    const nb = idx => {
      const ds = [];
      for(let j = 0; j < n; j++) if(j !== idx) ds.push([pts[j][0] - pts[idx][0], pts[j][1] - pts[idx][1], j]);
      ds.sort((a, b) => (a[0] * a[0] + a[1] * a[1]) - (b[0] * b[0] + b[1] * b[1]));
      const s = {};
      ds.slice(0, K).forEach(function(o){ s[o[2]] = 1; });
      return s;
    };
    let hit = 0, tot = 0;
    const hi = X.map(function(p, i){ return [p[0], p[1], p[2], p[3]]; });
    const hiN = [], loN = [];
    for(let i = 0; i < n; i++){
      const ds = [];
      for(let j = 0; j < n; j++){
        if(j === i) continue;
        let s = 0;
        for(let d = 0; d < DIM; d++){ const t = hi[i][d] - hi[j][d]; s += t * t; }
        ds.push([s, j]);
      }
      ds.sort((a, b) => a[0] - b[0]);
      const st = {};
      ds.slice(0, K).forEach(function(o){ st[o[1]] = 1; });
      hiN.push(st); loN.push(nb(i));
    }
    for(let i = 0; i < n; i++){
      Object.keys(hiN[i]).forEach(function(k){
        tot++;
        if(loN[i][k]) hit++;
      });
    }
    return tot ? hit / tot : 0;
  }
  const pcaP = (function(){
    const mu = [];
    for(let d = 0; d < DIM; d++) mu.push(X.reduce(function(s, p){ return s + p[d]; }, 0) / n);
    const Cc = X.map(p => p.map((v, d) => v - mu[d]));
    const S = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    Cc.forEach(function(p){
      for(let a = 0; a < DIM; a++) for(let b = 0; b < DIM; b++) S[a][b] += p[a] * p[b];
    });
    for(let a = 0; a < DIM; a++) for(let b = 0; b < DIM; b++) S[a][b] /= n;
    const v1 = (function(){
      let v = [1, 0.5, 0.3, 0.2];
      for(let it = 0; it < 200; it++){
        const nv = [0, 0, 0, 0];
        for(let a = 0; a < DIM; a++) for(let b = 0; b < DIM; b++) nv[a] += S[a][b] * v[b];
        const nn = Math.hypot(nv[0], nv[1], nv[2], nv[3]) || 1;
        v = nv.map(x => x / nn);
      }
      return v;
    })();
    return Cc.map(p => [p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2] + p[3] * v1[3], 0]);
  })();
  function draw(){
    scatter(g('cva'), raw2, true);
    scatter(g('cvb'), pcaP.map((p, i) => [p[0], X[i][1] * 0 + X[i][2] * 0.3]), true);
    scatter(g('cvc'), Y, true);
    g('p').textContent = kl.toFixed(3);
    g('q').textContent = step;
    const kp = step > 20 ? knnKeep(Y, 10) : 0;
    g('r').textContent = (kp * 100).toFixed(0) + '%';
    g('z').textContent = (knnKeep(pcaP.map((p, i) => [p[0], i * 0.01]), 10) * 100).toFixed(0) + '%';
    g('t').value = Math.min(1000, step);
    g('tv').textContent = step;
    let note;
    if(step === 0) note = '三个簇在 <b>4 维</b>空间里分得很开，但只看前 2 维（左图）它们糊成一团。'
      + '点「▶ 播放」，看 t-SNE 如何只用<b>点与点的邻近关系</b>把结构找回来。';
    else if(step < 120) note = '第 <b>' + step + '</b> 步：点团正在「炸开」。'
      + '早期梯度带放大因子（exaggeration），让簇间空隙先撑开，后面再收敛。';
    else note = '<b>' + step + '</b> 步：KL 散度降到 <b>' + kl.toFixed(3) + '</b>，'
      + '近邻保持率 <b>' + (kp * 100).toFixed(0) + '%</b>。t-SNE 保的是<b>局部邻域</b>——'
      + '所以簇间距离、簇的大小都不可信，只有「谁挨着谁」有意义。'
      + '困惑度调小会更碎、调大会更整体，可以拖着试试。';
    g('n').innerHTML = note;
  }
  function loop(){
    const cv = g('cvc');
    if(!cv || !cv.isConnected){ play = false; return; }
    if(play){
      for(let i = 0; i < 4; i++) iterate();
      draw();
      if(step < 1000) requestAnimationFrame(loop);
      else { play = false; g('g1').textContent = '▶ 播放'; }
    }
  }
  g('g1').onclick = function(){
    play = !play;
    g('g1').textContent = play ? '⏸ 暂停' : '▶ 播放';
    if(play) requestAnimationFrame(loop);
  };
  g('g2').onclick = function(){ play = false; g('g1').textContent = '▶ 播放'; reset(); draw(); };
  g('s').onchange = function(){ play = false; g('g1').textContent = '▶ 播放'; reset(); draw(); };
  g('s').oninput = function(){ g('sv').textContent = g('s').value; };
  g('t').oninput = function(){
    const tg = parseInt(g('t').value, 10);
    while(step < tg && step < 1000) iterate();
    draw();
  };
  reset();
  draw();
  g('sv').textContent = g('s').value;
});

/* ===== 23. 自编码器 AE ===== */
regDemo(['自编码器 AE', 'Autoencoder'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '输入（灰）与重建（蓝）', w: 280, h: 240},
            {k: 'b', t: '隐变量空间（取前 2 维）', w: 280, h: 240},
            {k: 'c', t: '重建误差随训练下降', w: 280, h: 240}],
    legend: '<span><i style="background:#b4b2a9"></i>原始点</span>'
          + '<span><i style="background:#185fa5"></i>重建点</span>'
          + '<span><i style="background:#c0392b"></i>残差</span>',
    ctl: [{k: 's', label: '隐变量维度 d', min: 1, max: 8, step: 1, val: 1},
          {k: 't', label: '隐藏层宽度 H', min: 4, max: 24, step: 2, val: 12},
          {k: 'u', label: '训练轮数', min: 0, max: 3000, step: 50, val: 600}],
    kpi: [{k: 'p', label: '重建 MSE'}, {k: 'q', label: '解释方差 R²'},
          {k: 'r', label: '压缩比'}, {k: 'z', label: '参数量'}]
  });
  const g = x => document.getElementById(id + x);
  const rnd = gaussRnd(20241122);
  const X = [];
  for(let i = 0; i < 150; i++){
    const t = -1.5 + 3 * i / 149;
    X.push([t, 0.85 * Math.sin(1.7 * t) + 0.12 * rnd()]);
  }
  const N = X.length;
  let mux = 0, muy = 0;
  X.forEach(p => { mux += p[0]; muy += p[1]; });
  mux /= N; muy /= N;
  let sdx = 0;
  X.forEach(p => { sdx += (p[0] - mux) * (p[0] - mux); });
  sdx = Math.sqrt(sdx / N);
  const Xs = X.map(p => [(p[0] - mux) / sdx, (p[1] - muy) / sdx]);
  const tanh = v => Math.tanh(v);
  function mkNet(H, d){
    let st = 20241123 >>> 0;
    const u = function(){ st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296 - 0.5; };
    const sc = 1 / Math.sqrt(2);
    const W1 = [], b1 = [];
    for(let i = 0; i < H; i++){ W1.push([u() * 2 * sc, u() * 2 * sc]); b1.push(0); }
    const W2 = [], b2 = [];
    for(let i = 0; i < d; i++){ W2.push([]); for(let j = 0; j < H; j++) W2[i].push(u() * 2 / Math.sqrt(H)); b2.push(0); }
    const W3 = [], b3 = [];
    for(let i = 0; i < H; i++){ W3.push([]); for(let j = 0; j < d; j++) W3[i].push(u() * 2 / Math.sqrt(d)); b3.push(0); }
    const W4 = [], b4 = [];
    for(let i = 0; i < 2; i++){ W4.push([]); for(let j = 0; j < H; j++) W4[i].push(u() * 2 / Math.sqrt(H)); b4.push(0); }
    return {W1: W1, b1: b1, W2: W2, b2: b2, W3: W3, b3: b3, W4: W4, b4: b4, H: H, d: d};
  }
  function fwd(net, x){
    const h1 = [];
    for(let i = 0; i < net.H; i++) h1.push(tanh(net.W1[i][0] * x[0] + net.W1[i][1] * x[1] + net.b1[i]));
    const z = [];
    for(let i = 0; i < net.d; i++){
      let s = net.b2[i];
      for(let j = 0; j < net.H; j++) s += net.W2[i][j] * h1[j];
      z.push(s);
    }
    const h2 = [];
    for(let i = 0; i < net.H; i++){
      let s = net.b3[i];
      for(let j = 0; j < net.d; j++) s += net.W3[i][j] * z[j];
      h2.push(tanh(s));
    }
    const out = [];
    for(let i = 0; i < 2; i++){
      let s = net.b4[i];
      for(let j = 0; j < net.H; j++) s += net.W4[i][j] * h2[j];
      out.push(s);
    }
    return {h1: h1, z: z, h2: h2, out: out};
  }
  function train(net, epochs, lr){
    const hist = [];
    for(let ep = 0; ep < epochs; ep++){
      const gW1 = net.W1.map(r => r.slice()), gb1 = net.b1.slice();
      const gW2 = net.W2.map(r => r.slice()), gb2 = net.b2.slice();
      const gW3 = net.W3.map(r => r.slice()), gb3 = net.b3.slice();
      const gW4 = net.W4.map(r => r.slice()), gb4 = net.b4.slice();
      gW1.forEach(r => { r[0] = 0; r[1] = 0; });
      gW2.forEach(r => r.fill(0)); gW3.forEach(r => r.fill(0)); gW4.forEach(r => r.fill(0));
      gb1.fill(0); gb2.fill(0); gb3.fill(0); gb4.fill(0);
      let mse = 0;
      Xs.forEach(function(x){
        const f = fwd(net, x);
        const e = [f.out[0] - x[0], f.out[1] - x[1]];
        mse += e[0] * e[0] + e[1] * e[1];
        const dout = [2 * e[0] / N, 2 * e[1] / N];
        const dh2 = new Array(net.H).fill(0);
        for(let i = 0; i < 2; i++) for(let j = 0; j < net.H; j++){
          gW4[i][j] += dout[i] * f.h2[j];
          dh2[j] += dout[i] * net.W4[i][j];
        }
        for(let i = 0; i < 2; i++) gb4[i] += dout[i];
        const dz = new Array(net.d).fill(0);
        for(let j = 0; j < net.H; j++){
          const dd = dh2[j] * (1 - f.h2[j] * f.h2[j]);
          for(let k = 0; k < net.d; k++){
            gW3[j][k] += dd * f.z[k];
            dz[k] += dd * net.W3[j][k];
          }
          gb3[j] += dd;
        }
        const dh1 = new Array(net.H).fill(0);
        for(let k = 0; k < net.d; k++) for(let j = 0; j < net.H; j++){
          gW2[k][j] += dz[k] * f.h1[j];
          dh1[j] += dz[k] * net.W2[k][j];
        }
        for(let k = 0; k < net.d; k++) gb2[k] += dz[k];
        for(let j = 0; j < net.H; j++){
          const dd = dh1[j] * (1 - f.h1[j] * f.h1[j]);
          gW1[j][0] += dd * x[0]; gW1[j][1] += dd * x[1];
          gb1[j] += dd;
        }
      });
      mse /= (2 * N);
      /* 全局梯度裁剪：大网络 + 大学习率时防止发散 */
      let gs = 0;
      const acc = function(v){ gs += v * v; };
      gW1.forEach(function(r){ acc(r[0]); acc(r[1]); });
      gW2.forEach(function(r){ r.forEach(acc); });
      gW3.forEach(function(r){ r.forEach(acc); });
      gW4.forEach(function(r){ r.forEach(acc); });
      gb1.forEach(acc); gb2.forEach(acc); gb3.forEach(acc); gb4.forEach(acc);
      const gnorm = Math.sqrt(gs);
      const cl = lr * (gnorm > 4 ? 4 / gnorm : 1);
      for(let i = 0; i < net.H; i++){
        net.W1[i][0] -= cl * gW1[i][0]; net.W1[i][1] -= cl * gW1[i][1]; net.b1[i] -= cl * gb1[i];
      }
      for(let i = 0; i < net.d; i++){
        for(let j = 0; j < net.H; j++) net.W2[i][j] -= cl * gW2[i][j];
        net.b2[i] -= cl * gb2[i];
      }
      for(let i = 0; i < net.H; i++){
        for(let j = 0; j < net.d; j++) net.W3[i][j] -= cl * gW3[i][j];
        net.b3[i] -= cl * gb3[i];
      }
      for(let i = 0; i < 2; i++){
        for(let j = 0; j < net.H; j++) net.W4[i][j] -= cl * gW4[i][j];
        net.b4[i] -= cl * gb4[i];
      }
      if(ep % 20 === 0 || ep === epochs - 1) hist.push(mse);
    }
    return hist;
  }
  function up(){
    const d = parseInt(g('s').value, 10), H = parseInt(g('t').value, 10);
    const ep = parseInt(g('u').value, 10);
    g('sv').textContent = d;
    g('tv').textContent = H;
    g('uv').textContent = ep;
    const net = mkNet(H, d);
    const hist = ep > 0 ? train(net, ep, 0.25) : [0];
    const R = Xs.map(x => fwd(net, x));
    const cv = g('cva'), c = dclr(cv), W = cv.width, Hh = cv.height;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    Xs.forEach(p => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); });
    const PX = v => 16 + (v - x0) / ((x1 - x0) || 1) * (W - 32);
    const PY = v => Hh - 16 - (v - y0) / ((y1 - y0) || 1) * (Hh - 32);
    c.strokeStyle = '#eceae2'; c.lineWidth = 1; c.strokeRect(8, 8, W - 16, Hh - 16);
    Xs.forEach(function(x, i){
      c.strokeStyle = 'rgba(192,57,43,.5)'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(PX(x[0]), PY(x[1])); c.lineTo(PX(R[i].out[0]), PY(R[i].out[1])); c.stroke();
    });
    Xs.forEach(function(x){
      c.fillStyle = '#b4b2a9';
      c.beginPath(); c.arc(PX(x[0]), PY(x[1]), 3, 0, 6.284); c.fill();
    });
    R.forEach(function(f){
      c.fillStyle = '#185fa5';
      c.beginPath(); c.arc(PX(f.out[0]), PY(f.out[1]), 2.6, 0, 6.284); c.fill();
    });
    const cv2 = g('cvb'), c2 = dclr(cv2), W2 = cv2.width, H2 = cv2.height;
    c2.strokeStyle = '#eceae2'; c2.lineWidth = 1; c2.strokeRect(8, 8, W2 - 16, H2 - 16);
    let z0 = Infinity, z1 = -Infinity;
    R.forEach(function(f){ z0 = Math.min(z0, f.z[0]); z1 = Math.max(z1, f.z[0]); });
    R.forEach(function(f, i){
      const zz = f.z.length > 1 ? f.z[1] : 0;
      const px = 16 + (f.z[0] - z0) / ((z1 - z0) || 1) * (W2 - 32);
      const py = f.z.length > 1 ? H2 - 16 - (zz + 1.2) / 2.4 * (H2 - 32) : H2 / 2;
      const t = i / R.length;
      c2.fillStyle = 'rgba(24,95,165,' + (0.3 + 0.7 * t) + ')';
      c2.beginPath(); c2.arc(px, py, 3.2, 0, 6.284); c2.fill();
    });
    c2.fillStyle = '#888780'; c2.font = '11px sans-serif';
    c2.fillText(d === 1 ? 'd=1：编码挤在一条直线上（按颜色看顺序）' : 'd=' + d + '：显示前 2 个隐变量', 14, H2 - 14);
    const cv3 = g('cvc'), c3 = dclr(cv3), W3 = cv3.width, H3 = cv3.height;
    c3.strokeStyle = '#eceae2'; c3.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H3 - 22 - i / 4 * (H3 - 36);
      c3.beginPath(); c3.moveTo(30, yy); c3.lineTo(W3 - 8, yy); c3.stroke();
    }
    if(hist.length > 1){
      const mx = Math.max.apply(null, hist) || 1;
      c3.strokeStyle = '#185fa5'; c3.lineWidth = 2; c3.beginPath();
      hist.forEach(function(v, i){
        const px = 30 + i / (hist.length - 1) * (W3 - 38), py = H3 - 22 - v / mx * (H3 - 36);
        i ? c3.lineTo(px, py) : c3.moveTo(px, py);
      });
      c3.stroke();
      c3.fillStyle = '#888780'; c3.font = '10px sans-serif';
      c3.fillText(mx.toExponential(1), 2, 12);
    }
    let se = 0, tv = 0;
    Xs.forEach(function(x, i){
      se += (x[0] - R[i].out[0]) * (x[0] - R[i].out[0]) + (x[1] - R[i].out[1]) * (x[1] - R[i].out[1]);
      tv += x[0] * x[0] + x[1] * x[1];
    });
    const mse = se / (2 * N);
    const r2 = 1 - se / (tv || 1);
    g('p').textContent = mse.toExponential(2);
    g('q').textContent = r2.toFixed(3);
    g('r').textContent = (2 / d).toFixed(1) + ' : 1';
    g('z').textContent = (2 * H + H + H * d + d + H * 2 + 2);
    g('qw').className = 'kpi' + (r2 > 0.9 ? ' ok' : (r2 > 0.5 ? '' : ' warn'));
    let note;
    if(ep === 0) note = '训练 0 轮：重建点（蓝）全挤在原点附近。向右拖动轮数，看它如何一点点贴回原始曲线。';
    else if(d === 1) note = 'd=<b>1</b>：整条正弦曲线被压进<b>一个</b>数里。因为数据本身就是沿曲线的 <b>1 维流形</b>——'
      + '这正是自编码器能压缩的本质原因。R² = ' + r2.toFixed(3) + '。';
    else if(d >= 6) note = 'd=<b>' + d + '</b>：隐变量维度已超过数据本身的自由度，'
      + '编码器只需「记住」每个点就能重建，<b>学不到任何结构</b>。这是自编码器的经典陷阱。';
    else note = 'd=<b>' + d + '</b>、H=<b>' + H + '</b>、<b>' + ep + '</b> 轮：MSE <b>'
      + mse.toExponential(2) + '</b>，R² <b>' + r2.toFixed(3) + '</b>。'
      + '瓶颈层就是<b>非线性版的 PCA</b>：PCA 只能做线性投影，这里靠 tanh 能把弯曲的流形「掰直」。';
    g('n').innerHTML = note;
  }
  ['s', 't', 'u'].forEach(function(kk){ g(kk).oninput = up; });
  up();
});

/* ===== 24. GBDT / XGBoost / LightGBM ===== */
regDemo(['GBDT / XGBoost / LightGBM', 'GBDT', '集成'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '集成预测曲线（竖线为当前残差）', w: 320, h: 250},
            {k: 'b', t: '训练 / 测试 RMSE 随树数变化', w: 320, h: 250}],
    legend: '<span><i style="background:#b4b2a9"></i>训练样本</span>'
          + '<span><i style="background:#c0392b"></i>真实函数</span>'
          + '<span><i style="background:#185fa5"></i>集成预测</span>',
    ctl: [{k: 's', label: '树的数量 M', min: 1, max: 120, step: 1, val: 20},
          {k: 't', label: '树深度', min: 1, max: 5, step: 1, val: 3},
          {k: 'u', label: '学习率 shrinkage', min: 0.05, max: 1, step: 0.05, val: 0.3}],
    kpi: [{k: 'p', label: '训练 RMSE'}, {k: 'q', label: '真值 RMSE'},
          {k: 'r', label: '真值−训练'}, {k: 'z', label: '叶子总数'}]
  });
  const g = x => document.getElementById(id + x);
  const rnd = gaussRnd(20241133);
  const tr = [], te = [];
  for(let i = 0; i < 70; i++){
    const x = -3 + 6 * (i + 0.5) / 70 + 0.12 * rnd();
    tr.push([x, 1.15 * Math.sin(1.55 * x) + 0.18 * x + 0.3 * rnd()]);
  }
  for(let i = 0; i < 140; i++){
    const x = -3 + 6 * (i + 0.5) / 140;
    te.push([x, 1.15 * Math.sin(1.55 * x) + 0.18 * x]);
  }
  function buildTree(pts, res, depth){
    const all = pts.map(function(p, i){ return [p[0], res[i]]; });
    const mean = res.reduce((a, b) => a + b, 0) / res.length;
    if(depth <= 0 || pts.length < 6) return {leaf: true, v: mean};
    const xs = all.map(o => o[0]).sort((a, b) => a - b);
    let best = null;
    for(let i = 1; i < 16; i++){
      const th = xs[Math.floor(i * xs.length / 16)];
      const L = all.filter(o => o[0] < th), R = all.filter(o => o[0] >= th);
      if(L.length < 3 || R.length < 3) continue;
      const sL = L.reduce((a, o) => a + o[1], 0), sR = R.reduce((a, o) => a + o[1], 0);
      const se = L.reduce(function(a, o){ const m = sL / L.length; return a + (o[1] - m) * (o[1] - m); }, 0)
               + R.reduce(function(a, o){ const m = sR / R.length; return a + (o[1] - m) * (o[1] - m); }, 0);
      if(!best || se < best.se) best = {se: se, th: th, L: L, R: R};
    }
    if(!best) return {leaf: true, v: mean};
    const ixL = [], ixR = [], rL = [], rR = [];
    pts.forEach(function(p, i){ (p[0] < best.th ? ixL : ixR).push(p); (p[0] < best.th ? rL : rR).push(res[i]); });
    return {leaf: false, th: best.th,
            l: buildTree(ixL, rL, depth - 1), r: buildTree(ixR, rR, depth - 1)};
  }
  const pev = (nd, x) => nd.leaf ? nd.v : (x < nd.th ? pev(nd.l, x) : pev(nd.r, x));
  function leaves(nd){ return nd.leaf ? 1 : leaves(nd.l) + leaves(nd.r); }
  function boost(M, depth, lr){
    let res = tr.map(o => o[1]);
    const trees = [];
    const f0 = tr.reduce((a, o) => a + o[1], 0) / tr.length;
    const htr = [], hte = [];
    for(let m = 0; m < M; m++){
      const t = buildTree(tr, res, depth);
      tr.forEach(function(o, i){ res[i] = res[i] - lr * pev(t, o[0]); });
      trees.push(t);
      let se = 0, see = 0;
      tr.forEach(function(o, i){
        let f = f0;
        trees.forEach(function(tt){ f += lr * pev(tt, o[0]); });
        se += (o[1] - f) * (o[1] - f);
      });
      te.forEach(function(o){
        let f = f0;
        trees.forEach(function(tt){ f += lr * pev(tt, o[0]); });
        see += (o[1] - f) * (o[1] - f);
      });
      htr.push(Math.sqrt(se / tr.length));
      hte.push(Math.sqrt(see / te.length));
    }
    return {trees: trees, f0: f0, htr: htr, hte: hte,
            res: tr.map(function(o, i){
              let f = f0;
              trees.forEach(function(tt){ f += lr * pev(tt, o[0]); });
              return o[1] - f;
            })};
  }
  function up(){
    const M = parseInt(g('s').value, 10), dep = parseInt(g('t').value, 10), lr = parseFloat(g('u').value);
    g('sv').textContent = M;
    g('tv').textContent = dep;
    g('uv').textContent = lr.toFixed(2);
    const r = boost(M, dep, lr);
    const cv = g('cva'), c = dclr(cv), W = cv.width, H = cv.height;
    const PX = x => 18 + (x + 3.2) / 6.4 * (W - 30);
    const PY = y => H - 18 - (y + 2.1) / 4.2 * (H - 32);
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(14, PY(0)); c.lineTo(W - 10, PY(0)); c.stroke();
    c.strokeStyle = '#c0392b'; c.lineWidth = 1.6; c.setLineDash([5, 3]); c.beginPath();
    for(let i = 0; i <= 200; i++){
      const x = -3.1 + 6.2 * i / 200, y = 1.15 * Math.sin(1.55 * x) + 0.18 * x;
      i ? c.lineTo(PX(x), PY(y)) : c.moveTo(PX(x), PY(y));
    }
    c.stroke(); c.setLineDash([]);
    c.strokeStyle = 'rgba(192,57,43,.35)'; c.lineWidth = 1;
    tr.forEach(function(o){ c.beginPath(); c.moveTo(PX(o[0]), PY(o[1])); c.lineTo(PX(o[0]), PY(o[1] - r.res[tr.indexOf(o)])); c.stroke(); });
    c.strokeStyle = '#185fa5'; c.lineWidth = 2.4; c.beginPath();
    for(let i = 0; i <= 200; i++){
      const x = -3.1 + 6.2 * i / 200;
      let f = r.f0;
      r.trees.forEach(function(t){ f += lr * pev(t, x); });
      i ? c.lineTo(PX(x), PY(f)) : c.moveTo(PX(x), PY(f));
    }
    c.stroke();
    c.fillStyle = '#b4b2a9';
    tr.forEach(function(o){ c.beginPath(); c.arc(PX(o[0]), PY(o[1]), 2.6, 0, 6.284); c.fill(); });
    const cv2 = g('cvb'), c2 = dclr(cv2), W2 = cv2.width, H2 = cv2.height;
    c2.strokeStyle = '#eceae2'; c2.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H2 - 22 - i / 4 * (H2 - 34);
      c2.beginPath(); c2.moveTo(30, yy); c2.lineTo(W2 - 8, yy); c2.stroke();
    }
    const mx = Math.max(Math.max.apply(null, r.htr), Math.max.apply(null, r.hte), 0.01) * 1.1;
    [[r.htr, '#185fa5'], [r.hte, '#c0392b']].forEach(function(pair){
      c2.strokeStyle = pair[1]; c2.lineWidth = 2; c2.beginPath();
      pair[0].forEach(function(v, i){
        const px = 30 + i / Math.max(1, pair[0].length - 1) * (W2 - 38), py = H2 - 22 - v / mx * (H2 - 34);
        i ? c2.lineTo(px, py) : c2.moveTo(px, py);
      });
      c2.stroke();
    });
    c2.fillStyle = '#5f5e5a'; c2.font = '10px sans-serif';
    c2.fillText('蓝=训练 红=测试', 32, 12);
    const a = r.htr[M - 1], b = r.hte[M - 1];
    const lv = r.trees.reduce(function(s, t){ return s + leaves(t); }, 0);
    g('p').textContent = a.toFixed(3);
    g('q').textContent = b.toFixed(3);
    g('r').textContent = (b - a).toFixed(3);
    g('z').textContent = lv;
    g('qw').className = 'kpi' + (b < 0.32 ? ' ok' : ' warn');
    let note;
    if(M === 1) note = 'M=1：只有<b>一棵树</b>，预测是阶梯状的常数块。'
      + 'Boosting 的每一棵都只拟合<b>当前残差</b>，不是拟合原目标——这就是「梯度提升」的梯度所在。';
    else if(dep >= 4 && M > 50) note = '深度 <b>' + dep + '</b> + <b>' + M + '</b> 棵树：'
      + '训练 RMSE 一路压到 ' + a.toFixed(3) + '，测试却停在 ' + b.toFixed(3)
      + '。树太深会立刻记住噪声——所以 XGBoost 默认只用深度 6 以内，靠<b>多棵弱树</b>堆出来。';
    else if(lr <= 0.1) note = '学习率 <b>' + lr.toFixed(2) + '</b>：<b>' + M
      + '</b> 棵树只走了很小一步，欠拟合。shrinkage 小意味着需要<b>更多树</b>——'
      + '它与 M 是一对需要联调的超参数。';
    else note = '<b>' + M + '</b> 棵深度 ' + dep + ' 的树：训练 RMSE <b>' + a.toFixed(3)
      + '</b>、对无噪声真值的 RMSE <b>' + b.toFixed(3) + '</b>（训练集本身含 0.3 的噪声，'
      + '所以真值误差可能更低；差值 ' + (b - a).toFixed(3) + ' 为负是正常的）。'
      + '每一棵新树都对准<b>上一轮的残差</b>，'
      + '逐步把阶梯磨成曲线。与随机森林相反——Boosting 降的是<b>偏差</b>，靠串行纠错；'
      + 'Bagging 降的是<b>方差</b>，靠并行平均。';
    g('n').innerHTML = note;
  }
  ['s', 't', 'u'].forEach(function(kk){ g(kk).oninput = up; });
  up();
});
/* DEMOS_B4:END */
