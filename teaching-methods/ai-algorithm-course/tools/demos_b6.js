/* DEMOS_B6:BEGIN */
/* ===== 28. 强化学习 DQN / PPO ===== */
regDemo(['强化学习 DQN / PPO', '强化学习', 'DQN', 'PPO'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '网格世界：底色=V(s)，箭头=贪心策略', w: 300, h: 280},
            {k: 'b', t: '每回合步数（含 10 回合滑动平均）', w: 300, h: 280}],
    legend: '<span><i style="background:#0f6e56"></i>终点 +1</span>'
          + '<span><i style="background:#c0392b"></i>陷阱 −1</span>'
          + '<span><i style="background:#5f5e5a"></i>障碍</span>'
          + '<span><i style="background:#ba7517"></i>「走一遍」后的路径</span>',
    ctl: [{k: 's', label: '探索率 ε', min: 0, max: 0.6, step: 0.02, val: 0.15},
          {k: 't', label: '折扣因子 γ', min: 0.5, max: 0.99, step: 0.01, val: 0.92},
          {k: 'u', label: '训练回合数', min: 0, max: 600, step: 10, val: 200}],
    btns: [{k: 'g1', label: '▶ 自动训练'}, {k: 'g2', label: '↺ 重新学习'}, {k: 'g3', label: '走一遍'}],
    kpi: [{k: 'p', label: '近 10 回合步数'}, {k: 'q', label: '近 10 回合成功率'},
          {k: 'r', label: '平均回报'}, {k: 'z', label: '已探索状态'}]
  });
  const g = x => document.getElementById(id + x);
  const R = 6, S0 = 0, S1 = R * R - 1;
  const WALL = [7, 8, 13, 22, 28, 29], TRAP = [16, 25, 19];
  const ACT = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const ARROW = ['↑', '↓', '←', '→'];
  const passable = s => WALL.indexOf(s) < 0;
  function step(s, a){
    const r = Math.floor(s / R), c = s % R;
    const nr = Math.min(R - 1, Math.max(0, r + ACT[a][0]));
    const nc = Math.min(R - 1, Math.max(0, c + ACT[a][1]));
    const ns = nr * R + nc;
    if(!passable(ns)) return {ns: s, r: -0.04, done: false};
    if(ns === S1) return {ns: ns, r: 1, done: true};
    if(TRAP.indexOf(ns) >= 0) return {ns: ns, r: -1, done: true};
    return {ns: ns, r: -0.04, done: false};
  }
  let Q = null, hist = null, eps = null, gmm = null, oks = null, auto = false, demo = null;
  let rndSt = 20241188 >>> 0;
  /* 固定种子的伪随机：保证同一组超参数下训练结果可复现 */
  function rndf(){ rndSt = (rndSt * 1664525 + 1013904223) >>> 0; return rndSt / 4294967296; }
  /* argmax 平局时随机打破：否则初期 Q 全为 0 会一直选动作 0，卡在起点撞墙 */
  function argmaxRnd(q){
    const mx = Math.max.apply(null, q), cand = [];
    q.forEach(function(v, i){ if(v >= mx - 1e-9) cand.push(i); });
    return cand[Math.floor(rndf() * cand.length) % cand.length];
  }
  function init(){
    Q = []; hist = []; eps = []; gmm = []; oks = [];
    for(let i = 0; i < R * R; i++){ Q.push([0, 0, 0, 0]); eps.push(0); gmm.push(0); }
    rndSt = 20241188 >>> 0;
    demo = null;
  }
  function episode(gam, ex){
    let s = S0, n = 0, tot = 0, ok = 0;
    while(n < 60){
      const u = rndf();
      const a = u < ex ? Math.floor(rndf() * 4) : argmaxRnd(Q[s]);
      const r = step(s, a);
      /* 终止状态（终点/陷阱）没有后续价值，目标里不再接 γ·max Q */
      const nxt = r.done ? 0 : Math.max.apply(null, Q[r.ns]);
      Q[s][a] = Q[s][a] + 0.4 * (r.r + gam * nxt - Q[s][a]);
      eps[s] = 1; gmm[s] = 0;
      tot += r.r; s = r.ns; n++;
      if(r.done){ ok = s === S1 ? 1 : 0; break; }
    }
    hist.push(n); gmm.push(tot); oks.push(ok);
    return {n: n, r: tot, ok: ok};
  }
  function trainTo(M, gam, ex){
    while(hist.length < M) episode(gam, ex);
    while(hist.length > M){ hist.pop(); gmm.pop(); oks.pop(); }
  }
  function drawGrid(){
    const cv = g('cva'), c = dclr(cv), W = cv.width, H = cv.height;
    const cs = Math.min(W, H) / R;
    let vmin = Infinity, vmax = -Infinity;
    Q.forEach(function(q, s){
      if(!passable(s)) return;
      const v = Math.max.apply(null, q);
      vmin = Math.min(vmin, v); vmax = Math.max(vmax, v);
    });
    const rg = (vmax - vmin) || 1;
    for(let s = 0; s < R * R; s++){
      const r = Math.floor(s / R), col = s % R;
      const x = col * cs, y = r * cs;
      if(!passable(s)){ c.fillStyle = '#5f5e5a'; c.fillRect(x, y, cs - 1, cs - 1); continue; }
      const v = Math.max.apply(null, Q[s]);
      const t = (v - vmin) / rg;
      c.fillStyle = 'rgba(24,95,165,' + (0.05 + 0.5 * t) + ')';
      c.fillRect(x, y, cs - 1, cs - 1);
      if(s === S1){ c.fillStyle = '#0f6e56'; c.fillRect(x, y, cs - 1, cs - 1); }
      if(TRAP.indexOf(s) >= 0){ c.fillStyle = 'rgba(192,57,43,.75)'; c.fillRect(x, y, cs - 1, cs - 1); }
      c.strokeStyle = '#eceae2'; c.lineWidth = 1;
      c.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      const bi = Q[s].indexOf(Math.max.apply(null, Q[s]));
      const mx = Math.max.apply(null, Q[s]), mn = Math.min.apply(null, Q[s]);
      if(mx - mn > 1e-4 && passable(s) && s !== S1){
        c.fillStyle = '#185fa5'; c.font = 'bold 15px sans-serif'; c.textAlign = 'center';
        c.fillText(ARROW[bi], x + cs / 2, y + cs / 2 + 5);
      }
      if(s === S0){
        c.fillStyle = '#185fa5'; c.font = 'bold 11px sans-serif'; c.textAlign = 'center';
        c.fillText('起', x + cs / 2, y + cs / 2 + 4);
      }
      if(s === S1){
        c.fillStyle = '#fff'; c.font = 'bold 11px sans-serif'; c.textAlign = 'center';
        c.fillText('终', x + cs / 2, y + cs / 2 + 4);
      }
    }
    c.textAlign = 'left';
    if(demo){
      c.strokeStyle = '#ba7517'; c.lineWidth = 2.4; c.beginPath();
      demo.forEach(function(s, i){
        const r = Math.floor(s / R), col = s % R;
        const px = col * cs + cs / 2, py = r * cs + cs / 2;
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      });
      c.stroke();
    }
  }
  function drawCurve(){
    const cv = g('cvb'), c = dclr(cv), W = cv.width, H = cv.height;
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H - 22 - i / 4 * (H - 34);
      c.beginPath(); c.moveTo(30, yy); c.lineTo(W - 8, yy); c.stroke();
    }
    c.fillStyle = '#888780'; c.font = '10px sans-serif';
    c.fillText('60', 6, 12); c.fillText('0', 6, H - 24);
    if(!hist.length) return;
    c.fillStyle = 'rgba(180,178,169,.55)';
    hist.forEach(function(v, i){
      const px = 30 + i / Math.max(1, hist.length - 1) * (W - 38);
      const py = H - 22 - v / 60 * (H - 34);
      c.fillRect(px - 0.8, py - 0.8, 1.8, 1.8);
    });
    const ma = [];
    hist.forEach(function(_, i){
      let s = 0, k = 0;
      for(let j = Math.max(0, i - 9); j <= i; j++){ s += hist[j]; k++; }
      ma.push(s / k);
    });
    c.strokeStyle = '#185fa5'; c.lineWidth = 2.2; c.beginPath();
    ma.forEach(function(v, i){
      const px = 30 + i / Math.max(1, ma.length - 1) * (W - 38);
      const py = H - 22 - v / 60 * (H - 34);
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    });
    c.stroke();
  }
  function refresh(){
    const M = parseInt(g('u').value, 10);
    const gam = parseFloat(g('t').value), ex = parseFloat(g('s').value);
    g('uv').textContent = M;
    drawGrid(); drawCurve();
    const last = hist.slice(-10);
    const avg = last.length ? last.reduce((a, b) => a + b, 0) / last.length : 0;
    const lo = oks.slice(-10);
    const okr = lo.length ? lo.reduce((a, b) => a + b, 0) / lo.length : 0;
    const mr = gmm.slice(-10);
    const ar = mr.length ? mr.reduce((a, b) => a + b, 0) / mr.length : 0;
    const seen = eps.reduce((a, b) => a + b, 0);
    g('p').textContent = hist.length ? avg.toFixed(1) : '—';
    g('q').textContent = hist.length ? (okr * 100).toFixed(0) + '%' : '—';
    g('r').textContent = hist.length ? ar.toFixed(2) : '—';
    g('z').textContent = seen + ' / 30';
    g('pw').className = 'kpi' + (hist.length && avg < 16 ? ' ok' : ' warn');
    let note;
    if(!hist.length) note = '还没训练：所有 Q 值都是 0，箭头还没长出来。'
      + '点「▶ 自动训练」看它怎么从乱走变成走直线。';
    else if(ex > 0.45) note = 'ε=<b>' + ex.toFixed(2) + '</b>：近一半动作在随机乱试。'
      + '探索太猛，Q 值收敛不了，步数一直抖动——这正是<b>探索-利用权衡</b>的代价。';
    else if(gam < 0.7) note = 'γ=<b>' + gam.toFixed(2) + '</b>：智能体「鼠目寸光」，'
      + '几乎只看眼前一步的奖励，V 值从终点往回传不了几格，学不出绕开陷阱的长路径。';
    else if(avg < 16) note = '<b>' + hist.length + '</b> 回合后，近 10 回合平均 <b>'
      + avg.toFixed(1) + '</b> 步就到终点（起点到终点的曼哈顿距离是 10 步），成功率 <b>'
      + (okr * 100).toFixed(0) + '%</b>。注意它学会了<b>绕开红色陷阱</b>——'
      + '因为掉进去是 −1，比多走几步的 −0.04 亏得多。';
    else note = '还在摸索：<b>' + hist.length + '</b> 回合，近 10 回合平均 <b>'
      + avg.toFixed(1) + '</b> 步。Q-learning 是<b>表格型</b>的，'
      + '状态一多就存不下——这时才需要 DQN 用神经网络去近似 Q(s,a)。';
    g('n').innerHTML = note;
  }
  function loop(){
    const cv = g('cva');
    if(!cv || !cv.isConnected){ auto = false; return; }
    if(auto){
      const gam = parseFloat(g('t').value), ex = parseFloat(g('s').value);
      for(let i = 0; i < 6; i++) episode(gam, ex);
      if(hist.length > 600) auto = false;
      g('u').value = Math.min(600, hist.length);
      refresh();
      if(auto) requestAnimationFrame(loop); else g('g1').textContent = '▶ 自动训练';
    }
  }
  init();
  g('s').oninput = function(){ g('sv').textContent = g('s').value; refresh(); };
  g('t').oninput = function(){ g('tv').textContent = g('t').value; refresh(); };
  g('u').oninput = function(){
    const M = parseInt(g('u').value, 10);
    const gam = parseFloat(g('t').value), ex = parseFloat(g('s').value);
    if(M < hist.length){ init(); }
    trainTo(M, gam, ex);
    refresh();
  };
  g('g1').onclick = function(){
    auto = !auto;
    g('g1').textContent = auto ? '⏸ 暂停' : '▶ 自动训练';
    if(auto) requestAnimationFrame(loop);
  };
  g('g2').onclick = function(){
    auto = false; g('g1').textContent = '▶ 自动训练';
    const gam = parseFloat(g('t').value), ex = parseFloat(g('s').value);
    init();
    const M = parseInt(g('u').value, 10);
    trainTo(M, gam, ex);
    refresh();
  };
  g('g3').onclick = function(){
    let s = S0, path = [s];
    for(let i = 0; i < 60; i++){
      const r = step(s, Q[s].indexOf(Math.max.apply(null, Q[s])));
      s = r.ns; path.push(s);
      if(r.done) break;
    }
    demo = path;
    refresh();
  };
  g('sv').textContent = g('s').value;
  g('tv').textContent = g('t').value;
  trainTo(parseInt(g('u').value, 10), parseFloat(g('t').value), parseFloat(g('s').value));
  refresh();
});

/* ===== 29. 生成模型 GAN / Diffusion ===== */
regDemo(['生成模型 GAN / Diffusion', 'GAN', 'Diffusion', '生成模型'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '真实数据 x₀', w: 240, h: 240},
            {k: 'b', t: '当前状态 x_t', w: 240, h: 240},
            {k: 'c', t: '信噪比 schedule ᾱ_t', w: 240, h: 240}],
    legend: '<span><i style="background:#185fa5"></i>数据点</span>'
          + '<span><i style="background:#b4b2a9"></i>纯噪声</span>'
          + '<span><i style="background:#c0392b"></i>当前时间步</span>',
    ctl: [{k: 's', label: '时间步 t', min: 0, max: 200, step: 1, val: 0},
          {k: 't', label: '总步数 T', min: 20, max: 200, step: 10, val: 100}],
    btns: [{k: 'g1', label: '▶ 从噪声去噪'}, {k: 'g2', label: '↺ 回到 t=T'}],
    kpi: [{k: 'p', label: 'ᾱ_t'}, {k: 'q', label: 'SNR (dB)'},
          {k: 'r', label: '与 x₀ 的相关'}, {k: 'z', label: '噪声占比'}]
  });
  const g = x => document.getElementById(id + x);
  const rnd = gaussRnd(20241199);
  const X0 = [];
  for(let i = 0; i < 220; i++){
    if(i % 2 === 0){
      const t = Math.PI * (i / 220) * 2;
      X0.push([Math.cos(t) * 0.75 - 0.3, Math.sin(t) * 0.75 + 0.15]);
    }else{
      const t = Math.PI * (i / 220) * 2;
      X0.push([1.1 - Math.cos(t) * 0.75, 0.35 - Math.sin(t) * 0.75]);
    }
  }
  X0.forEach(p => { p[0] += 0.045 * rnd(); p[1] += 0.045 * rnd(); });
  const EPS = X0.map(() => [rnd(), rnd()]);
  let abar = null, T = 100, curT = 0, Xt = null, playing = false;
  function schedule(t){
    const b0 = 1e-4, b1 = 0.055;
    const out = [1];
    let a = 1;
    for(let i = 1; i <= t; i++){
      const bt = b0 + (b1 - b0) * (i - 1) / Math.max(1, t - 1);
      a *= (1 - bt);
      out.push(a);
    }
    return out;
  }
  function xAt(t){
    const a = abar[Math.min(t, T)];
    return X0.map(function(p, i){
      return [Math.sqrt(a) * p[0] + Math.sqrt(1 - a) * EPS[i][0],
              Math.sqrt(a) * p[1] + Math.sqrt(1 - a) * EPS[i][1]];
    });
  }
  /* q(x_{t-1} | x_t, x_0)：不需要训练网络，用真实后验即可演示去噪 */
  function back(x, t){
    if(t <= 0) return x;
    const a = abar[t], ap = abar[t - 1], bt = 1 - a / ap;
    const coe1 = Math.sqrt(ap) * bt / (1 - a);
    const coe2 = Math.sqrt(1 - bt) * (1 - ap) / (1 - a);
    const sd = Math.sqrt((1 - ap) / (1 - a) * bt);
    return x.map(function(p, i){
      return [coe1 * X0[i][0] + coe2 * p[0] + sd * 0.35 * EPS[i][1],
              coe1 * X0[i][1] + coe2 * p[1] + sd * 0.35 * EPS[i][0]];
    });
  }
  function cloud(cv, pts, col){
    const c = dclr(cv), W = cv.width, H = cv.height, R = 3.1;
    c.strokeStyle = '#eceae2'; c.lineWidth = 1; c.strokeRect(6, 6, W - 12, H - 12);
    pts.forEach(function(p){
      const x = W / 2 + p[0] / 3.2 * (W / 2 - 12);
      const y = H / 2 - p[1] / 3.2 * (H / 2 - 12);
      c.fillStyle = col;
      c.beginPath(); c.arc(x, y, 2.4, 0, 6.284); c.fill();
    });
  }
  function drawSched(){
    const cv = g('cvc'), c = dclr(cv), W = cv.width, H = cv.height;
    c.strokeStyle = '#eceae2'; c.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const yy = H - 22 - i / 4 * (H - 34);
      c.beginPath(); c.moveTo(26, yy); c.lineTo(W - 8, yy); c.stroke();
    }
    c.strokeStyle = '#185fa5'; c.lineWidth = 2.2; c.beginPath();
    abar.forEach(function(v, i){
      const px = 26 + i / T * (W - 34), py = H - 22 - v * (H - 34);
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    });
    c.stroke();
    const px = 26 + curT / T * (W - 34);
    c.strokeStyle = '#c0392b'; c.lineWidth = 1.6; c.setLineDash([4, 3]);
    c.beginPath(); c.moveTo(px, 8); c.lineTo(px, H - 22); c.stroke();
    c.setLineDash([]);
    c.fillStyle = '#888780'; c.font = '10px sans-serif';
    c.fillText('ᾱ=1', 2, 12); c.fillText('ᾱ=0', 2, H - 24);
    c.fillStyle = '#c0392b';
    c.fillText('t=' + curT, Math.min(W - 40, px + 4), 18);
  }
  function refresh(){
    g('sv').textContent = curT;
    g('tv').textContent = T;
    Xt = xAt(curT);
    cloud(g('cva'), X0, 'rgba(24,95,165,.85)');
    cloud(g('cvb'), Xt, curT > T * 0.7 ? 'rgba(180,178,169,.9)' : 'rgba(24,95,165,.8)');
    drawSched();
    const a = abar[curT];
    let num = 0, d1 = 0, d2 = 0;
    X0.forEach(function(p, i){
      num += Xt[i][0] * p[0] + Xt[i][1] * p[1];
      d1 += Xt[i][0] * Xt[i][0] + Xt[i][1] * Xt[i][1];
      d2 += p[0] * p[0] + p[1] * p[1];
    });
    const corr = num / (Math.sqrt(d1 * d2) || 1);
    const snr = 10 * Math.log10(Math.max(1e-9, a) / Math.max(1e-9, 1 - a));
    g('p').textContent = a.toFixed(3);
    g('q').textContent = snr.toFixed(1);
    g('r').textContent = corr.toFixed(3);
    g('z').textContent = ((1 - a) * 100).toFixed(0) + '%';
    g('rw').className = 'kpi' + (corr > 0.9 ? ' ok' : '');
    let note;
    if(curT === 0) note = 't=0：这就是<b>真实数据</b>。向右拖时间步 = 前向扩散（加噪）；'
      + '点「▶ 从噪声去噪」= 反向生成。';
    else if(curT >= T) note = 't=T：ᾱ=<b>' + a.toFixed(4) + '</b>，原数据只剩 <b>'
      + (corr * 100).toFixed(0) + '%</b> 的相关性，看起来就是<b>一团高斯噪声</b>。'
      + '反向过程要做的，就是从这里一步步把结构「捞」回来。';
    else if(corr < 0.35) note = 't=<b>' + curT + '</b>：噪声已占 <b>' + ((1 - a) * 100).toFixed(0)
      + '%</b>，双月形状几乎看不出来了。Diffusion 的关键设计是：'
      + '<b>每一步只去掉一点点噪声</b>，把「一步生成」拆成 T 个简单的小任务。';
    else note = 't=<b>' + curT + '</b>：ᾱ=<b>' + a.toFixed(3) + '</b>，SNR <b>' + snr.toFixed(1)
      + ' dB</b>，与原始数据还有 <b>' + (corr * 100).toFixed(0) + '%</b> 的相关。'
      + '训练时网络学的就是「给定 x_t，预测刚才加进去的噪声 ε」——'
      + '目标是一个<b>确定性的回归</b>任务，所以比 GAN 的对抗训练稳定得多。';
    g('n').innerHTML = note;
  }
  function loop(){
    const cv = g('cvb');
    if(!cv || !cv.isConnected){ playing = false; return; }
    if(playing){
      curT = Math.max(0, curT - 2);
      Xt = back(Xt, curT + 2);
      refresh();
      if(curT > 0) requestAnimationFrame(loop);
      else { playing = false; g('g1').textContent = '▶ 从噪声去噪'; }
    }
  }
  T = parseInt(g('t').value, 10);
  abar = schedule(T);
  g('s').max = T;
  g('s').oninput = function(){
    playing = false; g('g1').textContent = '▶ 从噪声去噪';
    curT = parseInt(g('s').value, 10);
    refresh();
  };
  g('t').oninput = function(){
    playing = false; g('g1').textContent = '▶ 从噪声去噪';
    T = parseInt(g('t').value, 10);
    abar = schedule(T);
    g('s').max = T;
    curT = Math.min(curT, T);
    g('s').value = curT;
    refresh();
  };
  g('g1').onclick = function(){
    if(curT === 0){ curT = T; }
    playing = !playing;
    g('g1').textContent = playing ? '⏸ 暂停' : '▶ 从噪声去噪';
    if(playing) requestAnimationFrame(loop);
  };
  g('g2').onclick = function(){
    playing = false; g('g1').textContent = '▶ 从噪声去噪';
    curT = T; g('s').value = T;
    refresh();
  };
  refresh();
});

/* ===== 30. 异常检测 ===== */
regDemo(['异常检测'], function(host){
  const id = 'd' + (++DEMOSEQ);
  host.innerHTML = dshell(id, {
    cards: [{k: 'a', t: '密度等高线 + 判定结果', w: 320, h: 260},
            {k: 'b', t: '异常分数排序（红＝真异常）', w: 320, h: 260}],
    legend: '<span><i style="background:#b4b2a9"></i>正常样本</span>'
          + '<span><i style="background:#c0392b"></i>真异常</span>'
          + '<span><i style="background:#185fa5"></i>判定阈值</span>',
    ctl: [{k: 's', label: '污染率 contamination', min: 0.02, max: 0.25, step: 0.01, val: 0.08},
          {k: 't', label: 'KDE 带宽 h', min: 0.08, max: 0.6, step: 0.02, val: 0.22},
          {k: 'u', label: '注入异常点数', min: 0, max: 30, step: 1, val: 12}],
    kpi: [{k: 'p', label: '精确率'}, {k: 'q', label: '召回率'},
          {k: 'r', label: 'F1'}, {k: 'z', label: '误报数'}]
  });
  const g = x => document.getElementById(id + x);
  const rnd = gaussRnd(20241210);
  const NOR = [];
  for(let i = 0; i < 180; i++){
    const t = i / 180 * 6.2832;
    const rr = 1 + 0.22 * rnd();
    NOR.push([rr * Math.cos(t) * 1.15, rr * Math.sin(t) * 0.72]);
  }
  const OUT = [];
  for(let i = 0; i < 30; i++){
    const a = (i / 30) * 6.2832 + 0.4, rr = 2.05 + 0.5 * ((i % 5) / 5);
    OUT.push([rr * Math.cos(a) * 1.15, rr * Math.sin(a) * 0.8]);
  }
  const EX = {x0: -3, x1: 3, y0: -2.2, y2: 2.2};
  function kde(x, y, P, h){
    let s = 0;
    for(let i = 0; i < P.length; i++){
      const dx = x - P[i][0], dy = y - P[i][1];
      s += Math.exp(-(dx * dx + dy * dy) / (2 * h * h));
    }
    return s / (P.length * h * h * 6.2832);
  }
  function up(){
    const cont = parseFloat(g('s').value), h = parseFloat(g('t').value), nOut = parseInt(g('u').value, 10);
    g('sv').textContent = (cont * 100).toFixed(0) + '%';
    g('tv').textContent = h.toFixed(2);
    g('uv').textContent = nOut;
    const P = NOR.concat(OUT.slice(0, nOut));
    const truth = NOR.map(() => 0).concat(OUT.slice(0, nOut).map(() => 1));
    const sc = P.map(p => -Math.log(Math.max(1e-9, kde(p[0], p[1], P, h))));
    const sorted = sc.slice().sort((a, b) => b - a);
    const kth = Math.max(1, Math.round(cont * P.length));
    const thr = sorted[kth - 1];
    const pred = sc.map(v => v > thr ? 1 : 0);
    let tp = 0, fp = 0, fn = 0;
    pred.forEach(function(v, i){
      if(v && truth[i]) tp++;
      else if(v && !truth[i]) fp++;
      else if(!v && truth[i]) fn++;
    });
    const prec = tp + fp ? tp / (tp + fp) : 0;
    const rec = tp + fn ? tp / (tp + fn) : 0;
    const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
    const cv = g('cva'), c = dclr(cv), W = cv.width, H = cv.height, G = 5;
    const sm = 26 / P.length;
    for(let py = 0; py < H; py += G){
      for(let px = 0; px < W; px += G){
        const wx = EX.x0 + (px + 0.5) / W * (EX.x1 - EX.x0);
        const wy = EX.y0 + (1 - (py + 0.5) / H) * (EX.y2 - EX.y0);
        const s = -Math.log(Math.max(1e-9, kde(wx, wy, P, h)));
        const t = Math.max(0, Math.min(1, (s - thr) / (Math.max.apply(null, sc) - thr || 1)));
        c.fillStyle = t > 0.5 ? 'rgba(192,57,43,' + (0.06 + 0.22 * t) + ')'
                              : 'rgba(24,95,165,' + (0.05 + 0.2 * (1 - t)) + ')';
        c.fillRect(px, py, G, G);
      }
    }
    P.forEach(function(p, i){
      const px = (p[0] - EX.x0) / (EX.x1 - EX.x0) * W;
      const py2 = H - (p[1] - EX.y0) / (EX.y2 - EX.y0) * H;
      const isOut = truth[i] === 1, hit = pred[i] === 1;
      c.beginPath(); c.arc(px, py2, 3.1, 0, 6.284);
      c.fillStyle = isOut ? '#c0392b' : '#5f5e5a';
      c.fill();
      if(hit){
        c.strokeStyle = isOut ? '#0f6e56' : '#185fa5'; c.lineWidth = 1.8;
        c.beginPath(); c.arc(px, py2, 6, 0, 6.284); c.stroke();
      }
    });
    const cv2 = g('cvb'), c2 = dclr(cv2), W2 = cv2.width, H2 = cv2.height;
    const smax = Math.max.apply(null, sc), smin = Math.min.apply(null, sc);
    const rg = (smax - smin) || 1;
    sc.forEach(function(v, i){
      const px = 8 + i / sc.length * (W2 - 16);
      const py2 = H2 - 24 - (v - smin) / rg * (H2 - 40);
      c2.fillStyle = truth[i] ? '#c0392b' : '#b4b2a9';
      c2.fillRect(px, py2, Math.max(1.4, (W2 - 16) / sc.length - 0.6), H2 - 24 - py2);
    });
    const ty = H2 - 24 - (thr - smin) / rg * (H2 - 40);
    c2.strokeStyle = '#185fa5'; c2.lineWidth = 1.8; c2.setLineDash([5, 3]);
    c2.beginPath(); c2.moveTo(4, ty); c2.lineTo(W2 - 4, ty); c2.stroke();
    c2.setLineDash([]);
    c2.fillStyle = '#185fa5'; c2.font = '10px sans-serif';
    c2.fillText('阈值（前 ' + (cont * 100).toFixed(0) + '% 为异常）', 8, Math.max(10, ty - 5));
    c2.fillStyle = '#888780';
    c2.fillText('样本按异常分数从高到低排列', 8, H2 - 8);
    g('p').textContent = (prec * 100).toFixed(0) + '%';
    g('q').textContent = (rec * 100).toFixed(0) + '%';
    g('r').textContent = f1.toFixed(2);
    g('z').textContent = fp;
    g('rw').className = 'kpi' + (f1 > 0.85 ? ' ok' : ' warn');
    let note;
    if(h < 0.12) note = '带宽 h=<b>' + h.toFixed(2) + '</b> 太小：密度估计<b>过拟合</b>，'
      + '每个正常点周围都出现「孤岛」，连正常点都被判成异常（误报 <b>' + fp + '</b> 个）。';
    else if(h > 0.45) note = '带宽 h=<b>' + h.toFixed(2) + '</b> 太大：密度被抹平，'
      + '正常区与异常区的分数差不开，召回掉到 <b>' + (rec * 100).toFixed(0) + '%</b>。'
      + '带宽是无监督异常检测<b>最敏感</b>的超参数。';
    else if(nOut === 0) note = '没有注入异常点：此时所有被判为异常的都是<b>误报</b>——'
      + '因为 contamination 强制划走前 <b>' + (cont * 100).toFixed(0)
      + '%</b>。这是该类方法最容易被误用的地方：<b>它总能找出「异常」，但不一定真的异常</b>。';
    else note = 'contamination=<b>' + (cont * 100).toFixed(0) + '%</b>、h=<b>' + h.toFixed(2)
      + '</b>：精确率 <b>' + (prec * 100).toFixed(0) + '%</b>、召回率 <b>' + (rec * 100).toFixed(0)
      + '%</b>、F1 <b>' + f1.toFixed(2) + '</b>，误报 <b>' + fp + '</b> 个。'
      + '思路是<b>密度即正常</b>：核密度估计（KDE）给出每个位置的密度，'
      + '密度越低分数越高。绿色圈=正确捕获，蓝色圈=误报。';
    g('n').innerHTML = note;
  }
  ['s', 't', 'u'].forEach(function(k){ g(k).oninput = up; });
  up();
});
/* DEMOS_B6:END */
