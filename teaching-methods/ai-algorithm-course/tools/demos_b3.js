/* DEMOS_B3:BEGIN */
/* ===== 19. 特征选择 ===== */
regDemo(['特征选择', 'Feature Selection'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '单变量 t 统计量排序（选中高亮）', w: 320, h: 250},
            {k: 'b', t: '准确率随保留特征数 k 变化', w: 320, h: 250}],
    legend: '<span><i style="background:#0f6e56"></i>真正有用</span>'
          + '<span><i style="background:#ba7517"></i>冗余（有用的线性组合）</span>'
          + '<span><i style="background:#b4b2a9"></i>纯噪声</span>',
    ctl: [{k: 's', label: '保留特征数 k', min: 1, max: 12, step: 1, val: 4},
          {k: 't', label: '噪声水平', min: 0.2, max: 2.5, step: 0.1, val: 1}],
    kpi: [{k: 'p', label: '交叉验证准确率'}, {k: 'q', label: '命中真实特征'},
          {k: 'r', label: '训练准确率'}, {k: 'z', label: '过拟合 gap'}]
  });
  const g = x => document.getElementById(id + x);
  const P = 12, N = 260;
  const KIND = [];
  for(let j = 0; j < P; j++) KIND.push(j < 3 ? 0 : (j < 7 ? 1 : 2));
  const COLS = ['#0f6e56', '#ba7517', '#b4b2a9'];
  function build(ns){
    const r = gaussRnd(20241077);
    const X = [], y = [];
    for(let i = 0; i < N; i++){
      const row = [];
      for(let j = 0; j < P; j++) row.push(r());
      const s = row[0] + 0.75 * row[1] - 0.6 * row[2] + 0.62 * ns * r();
      X.push(row);
      y.push(s > 0 ? 1 : 0);
    }
    X.forEach(function(row){
      row[3] = 0.8 * row[0] + 0.6 * row[1] + 0.15 * ns * r();
      row[4] = -0.7 * row[1] + 0.5 * row[2] + 0.15 * ns * r();
      row[5] = row[0] - 0.4 * row[2] + 0.15 * ns * r();
      row[6] = 0.5 * row[0] + 0.5 * row[2] + 0.15 * ns * r();
    });
    return {X: X, y: y};
  }
  function tstat(X, y, j){
    const a = [], b = [];
    X.forEach(function(row, i){ (y[i] ? a : b).push(row[j]); });
    const m = v => v.reduce((s, x) => s + x, 0) / v.length;
    const v = (arr, mu) => arr.reduce((s, x) => s + (x - mu) * (x - mu), 0) / (arr.length - 1);
    const ma = m(a), mb = m(b), va = v(a, ma), vb = v(b, mb);
    const se = Math.sqrt(va / a.length + vb / b.length) || 1e-9;
    return Math.abs(ma - mb) / se;
  }
  /* 最近质心分类器：在 trI 上估计类中心，在 teI 上算准确率 */
  function ncAcc(X, y, idx, trI, teI){
    const sub = X.map(row => idx.map(j => row[j]));
    const dim = idx.length;
    const mu = [[], []], cnt = [0, 0];
    for(let c = 0; c < 2; c++) for(let j = 0; j < dim; j++) mu[c].push(0);
    trI.forEach(function(i){
      const c = y[i]; cnt[c]++;
      for(let j = 0; j < dim; j++) mu[c][j] += sub[i][j];
    });
    for(let c = 0; c < 2; c++) for(let j = 0; j < dim; j++) mu[c][j] /= (cnt[c] || 1);
    let ok = 0;
    teI.forEach(function(i){
      let bd = Infinity, bl = 0;
      for(let c = 0; c < 2; c++){
        let d = 0;
        for(let j = 0; j < dim; j++){ const t = sub[i][j] - mu[c][j]; d += t * t; }
        if(d < bd){ bd = d; bl = c; }
      }
      if(bl === y[i]) ok++;
    });
    return ok / teI.length;
  }
  function cvAcc(X, y, idx){
    const n = X.length, F = 5, all = [];
    for(let i = 0; i < n; i++) all.push(i);
    let ok = 0;
    for(let f = 0; f < F; f++){
      const te = all.filter(i => i % F === f), tr = all.filter(i => i % F !== f);
      ok += ncAcc(X, y, idx, tr, te) * te.length;
    }
    return ok / n;
  }
  function trAcc(X, y, idx){
    const all = [];
    for(let i = 0; i < X.length; i++) all.push(i);
    return ncAcc(X, y, idx, all, all);
  }
  function up(){
    const k = parseInt(g('s').value, 10), ns = parseFloat(g('t').value);
    g('sv').textContent = 'k = ' + k;
    g('tv').textContent = ns.toFixed(1);
    const d = build(ns);
    const ts = [];
    for(let j = 0; j < P; j++) ts.push({j: j, t: tstat(d.X, d.y, j), kind: KIND[j]});
    const rank = ts.slice().sort((a, b) => b.t - a.t);
    const sel = rank.slice(0, k).map(o => o.j);
    const c = dclr(g('cva')), W = g('cva').width, H = g('cva').height;
    const bw = W / P;
    const mx = Math.max.apply(null, rank.map(o => o.t)) || 1;
    rank.forEach(function(o, i){
      const h = o.t / mx * (H - 26);
      const on = sel.indexOf(o.j) >= 0;
      c.fillStyle = COLS[o.kind];
      c.globalAlpha = on ? 1 : 0.32;
      c.fillRect(i * bw + 3, H - 18 - h, bw - 6, h);
      c.globalAlpha = 1;
      c.fillStyle = on ? '#185fa5' : '#888780';
      c.font = (on ? 'bold ' : '') + '10px sans-serif'; c.textAlign = 'center';
      c.fillText('f' + o.j, i * bw + bw / 2, H - 6);
    });
    c.textAlign = 'left';
    const hist = [];
    for(let kk = 1; kk <= P; kk++){
      const s2 = rank.slice(0, kk).map(o => o.j);
      hist.push(cvAcc(d.X, d.y, s2));
    }
    const c2 = dclr(g('cvb')), W2 = g('cvb').width, H2 = g('cvb').height;
    c2.strokeStyle = '#eceae2'; c2.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H2 - 20 - i / 4 * (H2 - 32);
      c2.beginPath(); c2.moveTo(26, yy); c2.lineTo(W2 - 6, yy); c2.stroke();
    }
    c2.fillStyle = '#888780'; c2.font = '10px sans-serif';
    c2.fillText('1.00', 2, 12); c2.fillText('0.50', 2, H2 - 22);
    c2.strokeStyle = '#185fa5'; c2.lineWidth = 2; c2.beginPath();
    hist.forEach(function(v, i){
      const px = 26 + i / (P - 1) * (W2 - 34), py = H2 - 20 - (v - 0.5) / 0.5 * (H2 - 32);
      i ? c2.lineTo(px, py) : c2.moveTo(px, py);
    });
    c2.stroke();
    c2.strokeStyle = '#c0392b'; c2.lineWidth = 1.6; c2.setLineDash([4, 3]);
    const pxk = 26 + (k - 1) / (P - 1) * (W2 - 34);
    c2.beginPath(); c2.moveTo(pxk, 8); c2.lineTo(pxk, H2 - 20); c2.stroke();
    c2.setLineDash([]);
    hist.forEach(function(v, i){
      if(i + 1 === k){
        c2.fillStyle = '#185fa5';
        c2.beginPath(); c2.arc(26 + i / (P - 1) * (W2 - 34), H2 - 20 - (v - 0.5) / 0.5 * (H2 - 32), 3.6, 0, 6.284);
        c2.fill();
      }
    });
    const cvA = hist[k - 1];
    const trA = trAcc(d.X, d.y, sel);
    const hit = sel.filter(j => KIND[j] === 0).length;
    g('p').textContent = (cvA * 100).toFixed(1) + '%';
    g('q').textContent = hit + ' / 3';
    g('r').textContent = (trA * 100).toFixed(1) + '%';
    g('z').textContent = ((trA - cvA) * 100).toFixed(1) + ' pp';
    g('pw').className = 'kpi' + (cvA > 0.85 ? ' ok' : ' warn');
    let note;
    if(k <= 2) note = '只留 <b>' + k + '</b> 个特征：交叉验证 <b>' + (cvA * 100).toFixed(1)
      + '%</b>。信息不够，欠拟合——再往右加。';
    else if(k >= 9) note = 'k=<b>' + k + '</b>：后面几条柱子都是噪声（灰色）。'
      + '它们把<b>训练集</b>抬到 ' + (trA * 100).toFixed(1) + '%，交叉验证却停在 '
      + (cvA * 100).toFixed(1) + '%——这就是<b>维度灾难</b>：每多一个无关特征，都要更多样本才撑得住。';
    else note = 'k=<b>' + k + '</b>：命中 <b>' + hit + '/3</b> 个真实特征，交叉验证 <b>'
      + (cvA * 100).toFixed(1) + '%</b>。注意橙色（冗余）特征也能拿高分——'
      + '单变量筛选看不出冗余，要剔除它得用逐步回归或 L1。';
    g('n').innerHTML = note;
  }
  ['s', 't'].forEach(function(kk){ g(kk).oninput = up; });
  up();
});

/* ===== 20. 非负矩阵分解 NMF ===== */
regDemo(['非负矩阵分解 NMF', 'NMF'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '原始矩阵 V（16×16）', w: 220, h: 220},
            {k: 'b', t: '重建 W·H', w: 220, h: 220},
            {k: 'c', t: '学到的基（每块 4×4 为一列 W）', w: 220, h: 220}],
    legend: '<span><i style="background:#f2f0ea"></i>0</span>'
          + '<span><i style="background:#185fa5"></i>最大值</span>',
    ctl: [{k: 's', label: '分解秩 r', min: 1, max: 6, step: 1, val: 3},
          {k: 't', label: '乘法更新迭代次数', min: 0, max: 250, step: 5, val: 80}],
    kpi: [{k: 'p', label: '相对误差'}, {k: 'q', label: '压缩比'},
          {k: 'r', label: 'H 稀疏度'}, {k: 'z', label: '参数量'}]
  });
  const g = x => document.getElementById(id + x);
  const M = 16;
  const IM0 = [];
  for(let i = 0; i < M; i++){
    IM0.push([]);
    for(let j = 0; j < M; j++){
      let v = 0.06;
      if(Math.hypot(i - 5, j - 5) < 4.2) v += 0.85;
      if(i >= 10 && i <= 13 && j >= 2 && j <= 13) v += 0.7;
      if(j >= 10 && j <= 13 && i >= 2 && i <= 13) v += 0.55;
      if((i + j) % 5 === 0) v += 0.2;
      IM0[i].push(v);
    }
  }
  function rndInit(r){
    let st = 20241088 >>> 0;
    const u = function(){ st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
    const W = [], H = [];
    for(let i = 0; i < M; i++){ W.push([]); for(let a = 0; a < r; a++) W[i].push(0.15 + 0.5 * u()); }
    for(let a = 0; a < r; a++){ H.push([]); for(let j = 0; j < M; j++) H[a].push(0.15 + 0.5 * u()); }
    return {W: W, H: H};
  }
  /* A(m×k) @ B(k×n) */
  const mm = (A, B) => {
    const m = A.length, n = B[0].length, k = B.length, O = [];
    for(let i = 0; i < m; i++){
      O.push([]);
      for(let j = 0; j < n; j++){
        let s = 0;
        for(let t = 0; t < k; t++) s += A[i][t] * B[t][j];
        O[i].push(s);
      }
    }
    return O;
  };
  /* A^T @ B，要求 A.length === B.length */
  const mtm = (A, B) => {
    const m = A.length, k = A[0].length, n = B[0].length, O = [];
    for(let i = 0; i < k; i++){
      O.push([]);
      for(let j = 0; j < n; j++){
        let s = 0;
        for(let t = 0; t < m; t++) s += A[t][i] * B[t][j];
        O[i].push(s);
      }
    }
    return O;
  };
  /* A @ B^T，要求 A[0].length === B[0].length */
  const mmT = (A, B) => {
    const m = A.length, n = B.length, k = A[0].length, O = [];
    for(let i = 0; i < m; i++){
      O.push([]);
      for(let j = 0; j < n; j++){
        let s = 0;
        for(let t = 0; t < k; t++) s += A[i][t] * B[j][t];
        O[i].push(s);
      }
    }
    return O;
  };
  function run(r, iters){
    const w0 = rndInit(r);
    const W = w0.W, H = w0.H;
    for(let it = 0; it < iters; it++){
      const WtV = mtm(W, IM0);
      const WtWH = mtm(W, mm(W, H));
      for(let a = 0; a < r; a++) for(let j = 0; j < M; j++){
        H[a][j] = H[a][j] * WtV[a][j] / (WtWH[a][j] + 1e-9);
      }
      const VHt = mmT(IM0, H);
      const WHHt = mmT(mm(W, H), H);
      for(let i = 0; i < M; i++) for(let a = 0; a < r; a++){
        W[i][a] = W[i][a] * VHt[i][a] / (WHHt[i][a] + 1e-9);
      }
    }
    return {W: W, H: H, R: mm(W, H)};
  }
  function paint(cv, mat, ex){
    const c = dclr(cv), W = cv.width, H2 = cv.height;
    const R = mat.length, C = mat[0].length;
    const mx = ex || Math.max.apply(null, mat.map(r => Math.max.apply(null, r))) || 1;
    const s = Math.min(W / C, H2 / R);
    for(let i = 0; i < R; i++) for(let j = 0; j < C; j++){
      const v = Math.max(0, Math.min(1, mat[i][j] / mx));
      c.fillStyle = 'rgba(24,95,165,' + (0.06 + 0.94 * v) + ')';
      c.fillRect(j * s, i * s, s + 0.5, s + 0.5);
    }
  }
  function paintBasis(cv, Wm){
    const c = dclr(cv), Wd = cv.width, Hd = cv.height;
    const r = Wm[0].length;
    const bs = Math.min(Wd / r, Hd / 5) * 0.86;
    const mx = Math.max.apply(null, Wm.map(row => Math.max.apply(null, row))) || 1;
    for(let a = 0; a < r; a++){
      const ox = a * (Wd / r) + (Wd / r - bs) / 2, oy = (Hd - bs) / 2 - 6;
      for(let p = 0; p < 16; p++){
        const v = Math.max(0, Math.min(1, Wm[p][a] / mx));
        c.fillStyle = 'rgba(24,95,165,' + (0.06 + 0.94 * v) + ')';
        c.fillRect(ox + (p % 4) * (bs / 4), oy + Math.floor(p / 4) * (bs / 4), bs / 4 + 0.5, bs / 4 + 0.5);
      }
      c.fillStyle = '#5f5e5a'; c.font = '10px sans-serif'; c.textAlign = 'center';
      c.fillText('基 ' + (a + 1), ox + bs / 2, oy + bs + 14);
    }
    c.textAlign = 'left';
  }
  function up(){
    const r = parseInt(g('s').value, 10), it = parseInt(g('t').value, 10);
    g('sv').textContent = 'r = ' + r;
    g('tv').textContent = it;
    const res = run(r, it);
    const mx = Math.max.apply(null, IM0.map(row => Math.max.apply(null, row)));
    paint(g('cva'), IM0, mx);
    paint(g('cvb'), res.R, mx);
    paintBasis(g('cvc'), res.W);
    let se = 0, tt = 0;
    for(let i = 0; i < M; i++) for(let j = 0; j < M; j++){
      const d = IM0[i][j] - res.R[i][j]; se += d * d; tt += IM0[i][j] * IM0[i][j];
    }
    const rel = Math.sqrt(se / tt);
    let nz = 0, all = 0;
    res.H.forEach(function(row){ row.forEach(function(v){ all++; if(v < 0.02 * Math.max.apply(null, row)) nz++; }); });
    const comp = (M * M) / (r * (M + M));
    g('p').textContent = (rel * 100).toFixed(2) + '%';
    g('q').textContent = comp.toFixed(2) + ' : 1';
    g('r').textContent = (nz / all * 100).toFixed(0) + '%';
    g('z').textContent = r * (M + M);
    g('pw').className = 'kpi' + (rel < 0.12 ? ' ok' : ' warn');
    let note;
    if(it === 0) note = '第 0 次迭代：W、H 还是随机数，重建是一团模糊。'
      + '向右拖动，看乘法更新规则如何一步步把它「抠」成原图的样子。';
    else if(r === 1) note = 'r=1：只能用一个基去拼整张图，误差高达 <b>' + (rel * 100).toFixed(1)
      + '%</b>。秩越低，能表达的图案越少。';
    else if(rel < 0.06) note = 'r=<b>' + r + '</b>、<b>' + it + '</b> 次迭代后相对误差仅 <b>'
      + (rel * 100).toFixed(2) + '%</b>。关键区别在于<b>非负约束</b>：NMF 的基是「可叠加的部件」'
      + '（这里能看到单独的圆盘、横条、竖条），而 PCA 的正交基允许相减，学出来的是整图模式。';
    else note = '还在收敛中（相对误差 <b>' + (rel * 100).toFixed(1)
      + '%</b>）。乘法更新保证每步都不增大误差，但收敛慢——继续加大迭代次数。';
    g('n').innerHTML = note;
  }
  ['s', 't'].forEach(function(kk){ g(kk).oninput = up; });
  up();
});

/* ===== 21. 独立成分分析 ICA ===== */
regDemo(['独立成分分析 ICA', 'ICA'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '源信号 S（未知）', w: 300, h: 190},
            {k: 'b', t: '麦克风观测 X = A·S（已知）', w: 300, h: 190},
            {k: 'c', t: 'FastICA 恢复出的信号', w: 300, h: 190}],
    legend: '<span><i style="background:#c0392b"></i>分量 1</span>'
          + '<span><i style="background:#185fa5"></i>分量 2</span>'
          + '<span><i style="background:#0f6e56"></i>分量 3</span>',
    ctl: [{k: 's', label: '混合强度 α', min: 0, max: 1.2, step: 0.05, val: 0.7},
          {k: 't', label: 'FastICA 迭代次数', min: 0, max: 60, step: 1, val: 20}],
    btns: [{k: 'g1', label: '换一组源信号'}],
    kpi: [{k: 'p', label: '恢复相似度'}, {k: 'q', label: '混合矩阵 κ'},
          {k: 'r', label: '白化后协方差'}, {k: 'z', label: 'PCA 对比'}]
  });
  const g = x => document.getElementById(id + x);
  const T = 260, COLS = ['#c0392b', '#185fa5', '#0f6e56'];
  let seedOff = 0;
  function sources(){
    const S = [[], [], []];
    const ph = seedOff * 0.7;
    for(let t = 0; t < T; t++){
      const u = t / T;
      S[0].push(Math.sin(2 * Math.PI * (2.5 + ph * 0.3) * u + ph));
      S[1].push(Math.sign(Math.sin(2 * Math.PI * 1.4 * u + 1.1 + ph)));
      const v = ((u * 6 + ph * 0.2) % 1) * 2 - 1;
      S[2].push(0.85 * v);
    }
    return S;
  }
  function jacobi3(Ain){
    const A = Ain.map(r => r.slice());
    let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for(let sw = 0; sw < 80; sw++){
      let off = 0;
      for(let i = 0; i < 3; i++) for(let j = i + 1; j < 3; j++) off += A[i][j] * A[i][j];
      if(off < 1e-16) break;
      for(let p = 0; p < 3; p++) for(let q = p + 1; q < 3; q++){
        if(Math.abs(A[p][q]) < 1e-16) continue;
        const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
        const co = Math.cos(th), si = Math.sin(th);
        const R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        R[p][p] = co; R[q][q] = co; R[p][q] = si; R[q][p] = -si;
        const B = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
          let s = 0;
          for(let k = 0; k < 3; k++) s += R[k][i] * A[k][j];
          B[i][j] = s;
        }
        for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
          let s = 0;
          for(let k = 0; k < 3; k++) s += B[i][k] * R[k][j];
          A[i][j] = s;
        }
        const Vn = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
          let s = 0;
          for(let k = 0; k < 3; k++) s += V[i][k] * R[k][j];
          Vn[i][j] = s;
        }
        V = Vn;
      }
    }
    return {d: [A[0][0], A[1][1], A[2][2]], V: V};
  }
  function corr(a, b){
    const m = v => v.reduce((s, x) => s + x, 0) / v.length;
    const ma = m(a), mb = m(b);
    let sa = 0, sb = 0, sab = 0;
    for(let i = 0; i < a.length; i++){
      sa += (a[i] - ma) * (a[i] - ma);
      sb += (b[i] - mb) * (b[i] - mb);
      sab += (a[i] - ma) * (b[i] - mb);
    }
    return Math.abs(sab / (Math.sqrt(sa * sb) || 1e-9));
  }
  function drawSig(cv, S){
    const c = dclr(cv), W = cv.width, H = cv.height;
    const band = H / 3;
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    for(let i = 1; i < 3; i++){
      c.beginPath(); c.moveTo(0, i * band); c.lineTo(W, i * band); c.stroke();
    }
    S.forEach(function(s, k){
      let mx = 0;
      s.forEach(function(v){ mx = Math.max(mx, Math.abs(v)); });
      mx = mx || 1;
      const mid = k * band + band / 2;
      c.strokeStyle = COLS[k]; c.lineWidth = 1.7; c.beginPath();
      s.forEach(function(v, i){
        const px = i / (s.length - 1) * W, py = mid - v / mx * (band * 0.38);
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      });
      c.stroke();
    });
  }
  function up(){
    const al = parseFloat(g('s').value), it0 = parseInt(g('t').value, 10);
    g('sv').textContent = al.toFixed(2);
    g('tv').textContent = it0;
    const S = sources();
    const A = [[1, al * 0.9, al * 0.5], [al * 0.7, 1, al * 0.8], [al * 0.4, al * 0.6, 1]];
    const Xm = [[], [], []];
    for(let i = 0; i < 3; i++) for(let t = 0; t < T; t++){
      Xm[i].push(A[i][0] * S[0][t] + A[i][1] * S[1][t] + A[i][2] * S[2][t]);
    }
    drawSig(g('cva'), S);
    drawSig(g('cvb'), Xm);
    /* 中心化 + 白化 */
    const Z = [];
    for(let i = 0; i < 3; i++){
      const m = Xm[i].reduce((s, x) => s + x, 0) / T;
      Z.push(Xm[i].map(v => v - m));
    }
    const Cv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
      let s = 0;
      for(let t = 0; t < T; t++) s += Z[i][t] * Z[j][t];
      Cv[i][j] = s / (T - 1);
    }
    const eg = jacobi3(Cv);
    const diag = [eg.d[0], eg.d[1], eg.d[2]];
    const kap = Math.sqrt(Math.max.apply(null, diag) / Math.max(1e-9, Math.min.apply(null, diag)));
    /* 白化矩阵 = D^(-1/2) V^T，即 Wm[i][j] = V[j][i] / sqrt(d[i]) */
    const Wm = [[], [], []];
    for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++) Wm[i].push(eg.V[j][i] / Math.sqrt(Math.max(1e-9, eg.d[i])));
    const Zc = [];
    for(let i = 0; i < 3; i++){
      Zc.push([]);
      for(let t = 0; t < T; t++){
        let s = 0;
        for(let k = 0; k < 3; k++) s += Wm[i][k] * Z[k][t];
        Zc[i].push(s);
      }
    }
    /* FastICA 固定点迭代（对称正交化） */
    let st = (20241099 + seedOff * 131) >>> 0;
    const uf = function(){ st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
    const Wf = [];
    for(let i = 0; i < 3; i++){
      const v = [uf() * 2 - 1, uf() * 2 - 1, uf() * 2 - 1];
      const nn = Math.hypot(v[0], v[1], v[2]) || 1;
      Wf.push([v[0] / nn, v[1] / nn, v[2] / nn]);
    }
    for(let it = 0; it < it0; it++){
      const Wn = [];
      for(let i = 0; i < 3; i++){
        const wx = [];
        for(let t = 0; t < T; t++) wx.push(Wf[i][0] * Zc[0][t] + Wf[i][1] * Zc[1][t] + Wf[i][2] * Zc[2][t]);
        const gx = wx.map(Math.tanh), gp = gx.map(v => 1 - v * v);
        const e1 = [0, 0, 0];
        for(let k = 0; k < 3; k++){
          let s = 0;
          for(let t = 0; t < T; t++) s += Zc[k][t] * gx[t];
          e1[k] = s / T;
        }
        const e2 = gp.reduce((s, v) => s + v, 0) / T;
        const nv = [e1[0] - e2 * Wf[i][0], e1[1] - e2 * Wf[i][1], e1[2] - e2 * Wf[i][2]];
        Wn.push(nv);
      }
      const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
        let s = 0;
        for(let k = 0; k < 3; k++) s += Wn[i][k] * Wn[j][k];
        G[i][j] = s;
      }
      const eg2 = jacobi3(G);
      const Sm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
        let s = 0;
        for(let k = 0; k < 3; k++) s += eg2.V[k][i] * eg2.V[k][j] / Math.sqrt(Math.max(1e-9, eg2.d[k]));
        Sm[i][j] = s;
      }
      for(let i = 0; i < 3; i++){
        const nv = [0, 0, 0];
        for(let j = 0; j < 3; j++){
          let s = 0;
          for(let k = 0; k < 3; k++) s += Sm[i][k] * Wn[k][j];
          nv[j] = s;
        }
        const nn = Math.hypot(nv[0], nv[1], nv[2]) || 1;
        Wf[i] = [nv[0] / nn, nv[1] / nn, nv[2] / nn];
      }
    }
    const Y = [];
    for(let i = 0; i < 3; i++){
      Y.push([]);
      for(let t = 0; t < T; t++) Y[i].push(Wf[i][0] * Zc[0][t] + Wf[i][1] * Zc[1][t] + Wf[i][2] * Zc[2][t]);
    }
    drawSig(g('cvc'), Y);
    let best = 0;
    for(let i = 0; i < 3; i++){
      let bv = 0;
      for(let j = 0; j < 3; j++) bv = Math.max(bv, corr(Y[i], S[j]));
      best += bv;
    }
    best /= 3;
    /* PCA 对照：只用二阶统计量，无法分离 */
    let pcaBest = 0;
    for(let i = 0; i < 3; i++){
      let bv = 0;
      for(let j = 0; j < 3; j++) bv = Math.max(bv, corr(Zc[i], S[j]));
      pcaBest += bv;
    }
    pcaBest /= 3;
    let werr = 0;
    for(let i = 0; i < 3; i++) for(let j = 0; j < 3; j++){
      let s = 0;
      for(let t = 0; t < T; t++) s += Zc[i][t] * Zc[j][t];
      werr += Math.abs(s / (T - 1) - (i === j ? 1 : 0));
    }
    g('p').textContent = (best * 100).toFixed(1) + '%';
    g('q').textContent = kap.toFixed(2);
    g('r').textContent = werr.toFixed(3);
    g('z').textContent = (pcaBest * 100).toFixed(1) + '%';
    g('pw').className = 'kpi' + (best > 0.9 ? ' ok' : ' warn');
    let note;
    if(it0 === 0) note = '第 0 次迭代：恢复出的三条还是随机混合。'
      + '白化（PCA）已经做过——它只能<b>去相关</b>，看最后一张卡片：白化后信号依然混在一起。';
    else if(al < 0.15) note = 'α=<b>' + al.toFixed(2) + '</b>：几乎没混合，观测≈源信号。'
      + '把 α 拉大，让三个麦克风真正「串音」，才有解混的必要。';
    else note = '<b>' + it0 + '</b> 次迭代后恢复相似度 <b>' + (best * 100).toFixed(1)
      + '%</b>，而只用二阶统计量的 PCA 只有 <b>' + (pcaBest * 100).toFixed(1)
      + '%</b>。差别在于 ICA 用 tanh 逼近<b>非高斯性</b>——'
      + '高斯信号的任意线性组合仍是高斯，只有非高斯才「分得开」。'
      + '注意 ICA 有<b>排列与幅度不确定性</b>：顺序可能变、符号可能反，但波形对得上。';
    g('n').innerHTML = note;
  }
  g('g1').onclick = function(){ seedOff++; up(); };
  ['s', 't'].forEach(function(kk){ g(kk).oninput = up; });
  up();
});
/* DEMOS_B3:END */
