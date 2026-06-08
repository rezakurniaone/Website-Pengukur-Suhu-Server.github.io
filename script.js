/* ============================================================
   DATA SUHU SERVER (3 HARI)
============================================================ */
const JAM   = [8,9,10,11,12,13,14,15,16,17];
const JAM_LBL = ['08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00'];
const HARI1 = [25.2,26.4,26.8,27.4,27.5,28.0,28.1,28.0,27.8,27.3];
const HARI2 = [25.4,26.1,26.9,27.6,27.8,28.2,28.3,28.1,27.7,27.4];
const HARI3 = [25.0,26.2,26.7,27.2,27.7,27.9,28.0,27.8,27.5,27.1];
const AVG   = JAM.map((_,i)=>+((HARI1[i]+HARI2[i]+HARI3[i])/3).toFixed(4));

/* ============================================================
   METODE 1: INTERPOLASI LAGRANGE
   P(x) = Σ yᵢ * Lᵢ(x)
   Lᵢ(x) = Π (x-xⱼ)/(xᵢ-xⱼ) untuk j≠i
============================================================ */
function hitungSuhuLagrange(xs, ys, x) {
  let hasil = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    let Li = 1;
    for (let j = 0; j < n; j++) {
      if (j !== i) Li *= (x - xs[j]) / (xs[i] - xs[j]);
    }
    hasil += ys[i] * Li;
  }
  return hasil;
}

/* ============================================================
   METODE 2A: BUAT TABEL SELISIH TERBAGI
   tbl[i][j] = (tbl[i+1][j-1] - tbl[i][j-1]) / (x[i+j] - x[i])
============================================================ */
function buatTabelSelisih(xs, ys) {
  const n = xs.length;
  const tbl = Array.from({length:n}, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) tbl[i][0] = ys[i];
  for (let j = 1; j < n; j++)
    for (let i = 0; i < n - j; i++)
      tbl[i][j] = (tbl[i+1][j-1] - tbl[i][j-1]) / (xs[i+j] - xs[i]);
  return tbl;
}

/* ============================================================
   METODE 2B: HITUNG NEWTON DARI TABEL SELISIH
   P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
============================================================ */
function hitungSuhuNewton(xs, tbl, x) {
  let hasil = tbl[0][0];
  let prod  = 1;
  for (let i = 1; i < xs.length; i++) {
    prod  *= (x - xs[i-1]);
    hasil += tbl[0][i] * prod;
  }
  return hasil;
}

/* ============================================================
   METODE 3A: VALIDASI DOMINAN DIAGONAL
   |a_ii| >= Σ|a_ij| untuk semua j≠i, minimal satu ketat
============================================================ */
function isDiagDom(A) {
  let ketat = false;
  const n = A.length;
  for (let i = 0; i < n; i++) {
    const diag = Math.abs(A[i][i]);
    let sum = 0;
    for (let j = 0; j < n; j++) if (j !== i) sum += Math.abs(A[i][j]);
    if (diag < sum) return false;
    if (diag > sum) ketat = true;
  }
  return ketat;
}

/* ============================================================
   METODE 3B: GAUSS-SEIDEL
   Iterasi: x[i] = (b[i] - Σa[i][j]*x[j]) / a[i][i]
   Berhenti jika error < tol ATAU iterasi > maxIt
============================================================ */
function hitungSPLGaussSeidel(A, b, x0, tol, maxIt) {
  const n = b.length;
  let x = [...x0];
  const hist = [];

  if (!isDiagDom(A)) {
    return { error: 'Matriks TIDAK dominan diagonal! Iterasi mungkin tidak konvergen.' };
  }

  for (let iter = 1; iter <= maxIt; iter++) {
    const xLama = [...x];
    for (let i = 0; i < n; i++) {
      let sigma = 0;
      for (let j = 0; j < n; j++) if (j !== i) sigma += A[i][j] * x[j];
      x[i] = (b[i] - sigma) / A[i][i];
    }
    // Hitung Error Relatif Absolut maksimum
    let errMaks = 0;
    for (let i = 0; i < n; i++) {
      if (x[i] !== 0) {
        const e = Math.abs((x[i] - xLama[i]) / x[i]) * 100;
        if (e > errMaks) errMaks = e;
      }
    }
    hist.push(+errMaks.toFixed(8));
    if (errMaks < tol) return { sol: x, iter, konvergen: true, hist };
  }
  return { sol: x, iter: maxIt, konvergen: false, hist };
}

/* ============================================================
   PREDIKSI SUHU DENGAN GAUSS-SEIDEL (via polinomial kuadratik)
   Ambil 3 titik terdekat, susun SPL y = a0 + a1*x + a2*x²
   Selesaikan koefisien, evaluasi di waktuInput
============================================================ */
function prediksiSuhuGaussSeidel(xs, ys, x, deg) {
  deg = deg || 2;
  const nPt = deg + 1;
  // Urutkan titik berdasarkan jarak ke x
  const dist = xs.map((v,i) => ({d: Math.abs(v-x), i}))
               .sort((a,b) => a.d - b.d);
  const idx  = dist.slice(0, nPt).map(o => o.i).sort((a,b) => a-b);
  const xp   = idx.map(i => xs[i]);
  const yp   = idx.map(i => ys[i]);

  // Bangun matriks Vandermonde: A[i][j] = xp[i]^j
  const A = xp.map(xv => Array.from({length:nPt}, (_,j) => Math.pow(xv,j)));
  const b = [...yp];

  // Coba buat dominan diagonal dengan partial pivoting sederhana
  let Ac = A.map(r => [...r]), bc = [...b];
  for (let col = 0; col < nPt; col++) {
    let maxRow = col;
    for (let r = col+1; r < nPt; r++)
      if (Math.abs(Ac[r][col]) > Math.abs(Ac[maxRow][col])) maxRow = r;
    [Ac[col], Ac[maxRow]] = [Ac[maxRow], Ac[col]];
    [bc[col], bc[maxRow]] = [bc[maxRow], bc[col]];
  }

  let koef;
  if (isDiagDom(Ac)) {
    const r = hitungSPLGaussSeidel(Ac, bc, Array(nPt).fill(0), 0.0001, 200);
    koef = r.sol || bc; // fallback
  } else {
    // Gauss eliminasi sederhana sebagai fallback
    koef = solveGaussElim(Ac, bc);
  }
  // Evaluasi polinomial
  return koef.reduce((acc, c, j) => acc + c * Math.pow(x, j), 0);
}

// Gauss eliminasi (fallback jika tidak dominan diagonal)
function solveGaussElim(A, b) {
  const n = b.length;
  let M = A.map((r,i) => [...r, b[i]]);
  for (let col = 0; col < n; col++) {
    for (let row = col+1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n-1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i+1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

/* ============================================================
   KONVERSI DESIMAL → JAM:MENIT
============================================================ */
function desToJam(d) {
  const jam   = Math.floor(d);
  const menit = Math.round((d - jam) * 60);
  return {
    jam:   String(jam).padStart(2,'0'),
    menit: String(menit).padStart(2,'0'),
    label: String(jam).padStart(2,'0') + ':' + String(menit).padStart(2,'0')
  };
}

/* ============================================================
   STATE & CHART
============================================================ */
let chartObj = null;
let xSaat    = 8.5;

function updateDisplay(x) {
  xSaat = x;
  const jm = desToJam(x);
  document.getElementById('dispJam').textContent   = jm.jam;
  document.getElementById('dispMenit').textContent = jm.menit;
  document.getElementById('dispDes').textContent   = (+x).toFixed(1);
}

function onSlider(v) {
  document.getElementById('inputManual').value = v;
  updateDisplay(parseFloat(v));
}

function onManual(v) {
  const f = parseFloat(v);
  if (!isNaN(f)) {
    document.getElementById('sliderWaktu').value = f;
    updateDisplay(f);
  }
}

/* ============================================================
   COMPUTE — Dipanggil saat tombol ditekan
============================================================ */
function compute() {
  const x = parseFloat(document.getElementById('inputManual').value);
  const errBox = document.getElementById('errBox');
  if (isNaN(x) || x < 8 || x > 17) {
    errBox.style.display = 'block';
    return;
  }
  errBox.style.display = 'none';
  updateDisplay(x);

  // === Hitung ketiga metode ===
  const tbl = buatTabelSelisih(JAM, AVG);
  const vL  = hitungSuhuLagrange(JAM, AVG, x);
  const vN  = hitungSuhuNewton(JAM, tbl, x);
  const vG  = prediksiSuhuGaussSeidel(JAM, AVG, x, 2);

  // === Tampilkan hasil ===
  document.getElementById('resLag').textContent = vL.toFixed(4);
  document.getElementById('resNwt').textContent = vN.toFixed(4);
  document.getElementById('resGs').textContent  = vG.toFixed(4);

  const avg3 = (vL + vN + vG) / 3;
  document.getElementById('diffLag').textContent = 'Δ dari avg: ' + (vL - avg3).toFixed(4);
  document.getElementById('diffNwt').textContent = 'Δ dari avg: ' + (vN - avg3).toFixed(4);
  document.getElementById('diffGs').textContent  = 'Δ dari avg: ' + (vG - avg3).toFixed(4);

  // === Grafik ===
  drawChart(x, vL, vN, vG);

  // === Log Lagrange ===
  let logL = [`<span class="head">hitungSuhuLagrange(JAM, AVG, ${x})</span>`, ''];
  let totalL = 0;
  for (let i = 0; i < JAM.length; i++) {
    let Li = 1;
    for (let j = 0; j < JAM.length; j++) if (j !== i) Li *= (x - JAM[j]) / (JAM[i] - JAM[j]);
    const suku = AVG[i] * Li;
    totalL += suku;
    logL.push(`  <span class="kw">L</span><sub>${i}</sub>(${x}) = <span class="num">${Li.toFixed(6)}</span>  →  suku = <span class="num">${suku.toFixed(6)}</span>`);
  }
  logL.push('');
  logL.push(`  <span class="ok">Hasil = ${vL.toFixed(6)} °C</span>`);
  document.getElementById('logLag').innerHTML = logL.join('<br>');

  // === Log Newton ===
  let logN = [`<span class="head">hitungSuhuNewton(JAM, tabelSelisih, ${x})</span>`, ''];
  logN.push('  Koefisien f[x₀,...,xₖ] (baris-0 tabel selisih terbagi):');
  for (let j = 0; j < JAM.length; j++) {
    logN.push(`  &nbsp;&nbsp;orde-<span class="num">${j}</span>: <span class="num">${tbl[0][j].toFixed(8)}</span>`);
  }
  logN.push('');
  logN.push(`  <span class="ok">Hasil = ${vN.toFixed(6)} °C</span>`);
  document.getElementById('logNwt').innerHTML = logN.join('<br>');

  // === Log Ringkasan ===
  let logD = [
    `<span class="head">Ringkasan Prediksi Suhu — Jam ${desToJam(x).label}</span>`,
    '',
    `  <span class="hi">Lagrange    :</span> <span class="num">${vL.toFixed(4)}</span> °C`,
    `  <span class="ok">Newton      :</span> <span class="num">${vN.toFixed(4)}</span> °C`,
    `  <span class="warn">Gauss-Seidel:</span> <span class="num">${vG.toFixed(4)}</span> °C`,
    '',
    `  Rata-rata ketiga metode : <span class="num">${avg3.toFixed(4)}</span> °C`,
    `  Selisih Lagrange–Newton : <span class="num">${Math.abs(vL-vN).toExponential(3)}</span>`,
    '',
    `  <span class="kw">Data rata-rata</span> jam terdekat:`,
  ];
  JAM.forEach((j,i) => {
    const jarak = Math.abs(j - x);
    if (jarak <= 1.0) {
      logD.push(`  &nbsp;&nbsp;Jam ${JAM_LBL[i]} → <span class="num">${AVG[i].toFixed(4)}</span> °C  (Δ = ${jarak.toFixed(1)} jam)`);
    }
  });
  document.getElementById('logDetail').innerHTML = logD.join('<br>');
}

/* ============================================================
   CHART.JS
============================================================ */
function drawChart(x, vL, vN, vG) {
  const pts = [];
  for (let t = 8; t <= 17; t += 0.05) {
    pts.push({ x: +t.toFixed(2), y: +hitungSuhuLagrange(JAM, AVG, t).toFixed(4) });
  }
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (chartObj) chartObj.destroy();

  chartObj = new Chart(ctx, {
    data: {
      datasets: [
        {
          type: 'line',
          label: 'Kurva Lagrange',
          data: pts,
          borderColor: '#3b82f6',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          order: 3
        },
        {
          type: 'scatter',
          label: 'Data rata-rata',
          data: JAM.map((j,i) => ({x:j, y:AVG[i]})),
          backgroundColor: '#10b981',
          pointRadius: 5,
          pointHoverRadius: 7,
          order: 2
        },
        {
          type: 'scatter',
          label: `Titik x=${x} (Lag)`,
          data: [{x, y: +vL.toFixed(4)}],
          backgroundColor: '#3b82f6',
          pointRadius: 9,
          pointStyle: 'rectRot',
          order: 1
        },
        {
          type: 'scatter',
          label: `Titik x=${x} (Nwt)`,
          data: [{x, y: +vN.toFixed(4)}],
          backgroundColor: '#10b981',
          pointRadius: 7,
          pointStyle: 'triangle',
          order: 1
        },
        {
          type: 'scatter',
          label: `Titik x=${x} (GS)`,
          data: [{x, y: +vG.toFixed(4)}],
          backgroundColor: '#f59e0b',
          pointRadius: 7,
          pointStyle: 'cross',
          order: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#6b7280',
            font: { family: "'IBM Plex Mono', monospace", size: 11 },
            boxWidth: 12, boxHeight: 12, padding: 12
          }
        },
        tooltip: {
          backgroundColor: '#13161e',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#e8eaf0',
          bodyColor: '#6b7280',
          titleFont: { family: "'IBM Plex Mono', monospace", size: 12 },
          bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${(+ctx.parsed.y).toFixed(4)} °C`
          }
        }
      },
      scales: {
        x: {
          type: 'linear', min: 7.8, max: 17.2,
          title: { display: true, text: 'Waktu (jam desimal)', color: '#6b7280', font: { size: 11, family: "'IBM Plex Mono', monospace" } },
          ticks: { color: '#6b7280', font: { size: 11 } },
          grid: { color: 'rgba(255,255,255,0.04)' }
        },
        y: {
          title: { display: true, text: 'Suhu (°C)', color: '#6b7280', font: { size: 11, family: "'IBM Plex Mono', monospace" } },
          ticks: { color: '#6b7280', font: { size: 11 } },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      }
    }
  });
}

/* ============================================================
   GAUSS-SEIDEL PANEL
============================================================ */
function jalanGS() {
  const parseRow = id => document.getElementById(id).value.trim().split(/\s+/).map(Number);
  const A = [parseRow('gsA1'), parseRow('gsA2'), parseRow('gsA3')];
  const b = ['gsB1','gsB2','gsB3'].map(id => parseFloat(document.getElementById(id).value));
  const tol   = parseFloat(document.getElementById('gsTol').value);
  const maxIt = parseInt(document.getElementById('gsMaxIt').value);
  const el = document.getElementById('logGs');

  const r = hitungSPLGaussSeidel(A, b, [0,0,0], tol, maxIt);
  if (r.error) {
    el.innerHTML = `<span class="err">✗ ${r.error}</span><br><span class="warn">Pastikan |a_ii| ≥ Σ|a_ij| untuk setiap baris i.</span>`;
    return;
  }
  const lines = [
    `<span class="head">Hasil Gauss-Seidel — SPL 3×3</span>`,
    '',
    `  Solusi <span class="kw">x</span> = [${r.sol.map(v => v.toFixed(6)).join(', ')}]`,
    `  Iterasi   : <span class="num">${r.iter}</span> / ${maxIt}`,
    `  Konvergen : ${r.konvergen ? '<span class="ok">✓ YA</span>' : '<span class="err">✗ TIDAK (maxIterasi tercapai)</span>'}`,
    `  Error akhir : <span class="num">${r.hist[r.hist.length-1]}</span> %`,
    '',
    `  <span class="kw">Riwayat error (10 iterasi pertama):</span>`,
    ...r.hist.slice(0,10).map((e,i) =>
      `  &nbsp;&nbsp;iter <span class="num">${String(i+1).padStart(2)}</span>: ${e} %`
    )
  ];
  el.innerHTML = lines.join('<br>');
}

/* ============================================================
   TAB SWITCHING
============================================================ */
function switchTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

/* ============================================================
   DATA TABLE
============================================================ */
const ALL_DATA = [HARI1, HARI2, HARI3, AVG];
const DAY_LABELS = ['Hari 1', 'Hari 2', 'Hari 3', 'Rata-rata'];
let activeDay = 0;

function switchData(idx, btn) {
  activeDay = idx;
  document.querySelectorAll('.data-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTable(idx);
}

function suhuClass(s) {
  if (s >= 28.0) return 'td-hot';
  if (s >= 27.5) return 'td-warm';
  return 'td-cool';
}
function statusLabel(s) {
  if (s >= 28.0) return '⬆ Tinggi';
  if (s >= 27.5) return '→ Sedang';
  return '⬇ Normal';
}

function renderTable(dayIdx) {
  const ys = ALL_DATA[dayIdx];
  const tbody = document.getElementById('dataBody');
  tbody.innerHTML = JAM.map((j,i) => {
    const s = ys[i];
    const cls = suhuClass(s);
    return `<tr>
      <td class="td-time">${String(i+1).padStart(2,'0')}</td>
      <td>${JAM_LBL[i]}</td>
      <td class="td-time">${j.toFixed(1)}</td>
      <td class="${cls}" style="font-weight:500">${s.toFixed(4)}</td>
      <td class="${cls}">${statusLabel(s)}</td>
    </tr>`;
  }).join('');
}

/* ============================================================
   INIT
============================================================ */
renderTable(0);
updateDisplay(8.5);
// Gambar grafik awal tanpa titik prediksi dulu
(function initChart(){
  const pts = [];
  for (let t = 8; t <= 17; t += 0.05)
    pts.push({x:+t.toFixed(2), y:+hitungSuhuLagrange(JAM,AVG,t).toFixed(4)});
  const ctx = document.getElementById('mainChart').getContext('2d');
  chartObj = new Chart(ctx, {
    data: {
      datasets: [
        { type:'line', label:'Kurva Lagrange', data:pts, borderColor:'#3b82f6', borderWidth:2, pointRadius:0, fill:false, tension:0.3 },
        { type:'scatter', label:'Data rata-rata', data:JAM.map((j,i)=>({x:j,y:AVG[i]})), backgroundColor:'#10b981', pointRadius:5 }
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{labels:{color:'#6b7280',font:{family:"'IBM Plex Mono',monospace",size:11},boxWidth:12,boxHeight:12,padding:12}},
        tooltip:{backgroundColor:'#13161e',borderColor:'rgba(255,255,255,0.1)',borderWidth:1,titleColor:'#e8eaf0',bodyColor:'#6b7280',titleFont:{family:"'IBM Plex Mono',monospace",size:12},bodyFont:{family:"'IBM Plex Mono',monospace",size:11}}
      },
      scales:{
        x:{type:'linear',min:7.8,max:17.2,title:{display:true,text:'Waktu (jam desimal)',color:'#6b7280',font:{size:11}},ticks:{color:'#6b7280',font:{size:11}},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{title:{display:true,text:'Suhu (°C)',color:'#6b7280',font:{size:11}},ticks:{color:'#6b7280',font:{size:11}},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
})();
