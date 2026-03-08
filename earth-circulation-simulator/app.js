/**
 * Earth Circulation Simulator — browser implementation
 * Physical model: heating gradient → pressure → winds → Coriolis → ocean → gyres
 */

(function () {
  'use strict';

  // ----- Grid & constants (spec) -----
  const NLON = 60;
  const NLAT = 30;
  const LAT_MIN = -90;
  const LAT_MAX = 90;
  const LON_MIN = 0;
  const LON_MAX = 360;
  const OMEGA = 7.2921e-5;  // s⁻¹
  const DT = 0.1;
  const STEPS_PER_UPDATE = 350;
  const MAX_PARTICLES = 100;
  const PARTICLE_TRAIL_LENGTH = 80;

  // Physics scaling (tuned for qualitative patterns)
  const K_PRESSURE = 0.8;
  const R_WIND = 0.15;
  const ALPHA_STRESS = 0.12;
  const R_OCEAN_BASE = 0.08;
  const NU = 0.02;

  const dLon = (LON_MAX - LON_MIN) / NLON;
  const dLat = (LAT_MAX - LAT_MIN) / NLAT;

  function lonIndexToLon(i) {
    return LON_MIN + (i + 0.5) * dLon;
  }
  function latIndexToLat(j) {
    return LAT_MIN + (j + 0.5) * dLat;
  }

  // ----- Land mask (simplified: Americas, Eurasia/Africa, Antarctica) -----
  function isLand(i, j) {
    const lon = lonIndexToLon(i);
    const lat = latIndexToLat(j);
    // Antarctica band
    if (lat <= -60) return true;
    // Americas (approx -130° to -30° → 230° to 330°)
    if (lon >= 230 && lon <= 330 && lat >= -55 && lat <= 75) return true;
    // Eurasia / Africa (approx -20° to 150° → 340° to 150°)
    if (lon <= 150 && lat >= -35 && lat <= 75) return true;
    if (lon >= 340 && lat >= -35 && lat <= 75) return true;
    return false;
  }

  function buildLandMask() {
    const mask = [];
    for (let j = 0; j < NLAT; j++) {
      for (let i = 0; i < NLON; i++) {
        mask.push(isLand(i, j) ? 1 : 0);
      }
    }
    return mask;
  }

  function idx(i, j) {
    return j * NLON + i;
  }

  // ----- Coriolis parameter f = coriolis_scale * 2Ω sin(lat) -----
  function coriolisAt(j, coriolisScale) {
    const latRad = (latIndexToLat(j) * Math.PI) / 180;
    return coriolisScale * 2 * OMEGA * Math.sin(latRad);
  }

  // ----- Temperature T(lat) = cos(lat), ∂P/∂y ∝ -∂T/∂y = sin(lat) (in lat direction) -----
  function pressureGradientMeridional(j) {
    const latRad = (latIndexToLat(j) * Math.PI) / 180;
    const dTdLat = -Math.sin(latRad);
    return -K_PRESSURE * dTdLat * (1 / 180) * Math.PI;
  }

  // ----- Laplacian (periodic in lon, reflective at lat bounds) -----
  function laplacianU(u, i, j) {
    const im = (i - 1 + NLON) % NLON;
    const ip = (i + 1) % NLON;
    const jm = Math.max(0, j - 1);
    const jp = Math.min(NLAT - 1, j + 1);
    const u0 = u[idx(i, j)];
    return (u[idx(ip, j)] + u[idx(im, j)] + u[idx(i, jp)] + u[idx(i, jm)] - 4 * u0);
  }

  function laplacianV(v, i, j) {
    const im = (i - 1 + NLON) % NLON;
    const ip = (i + 1) % NLON;
    const jm = Math.max(0, j - 1);
    const jp = Math.min(NLAT - 1, j + 1);
    const v0 = v[idx(i, j)];
    return (v[idx(ip, j)] + v[idx(im, j)] + v[idx(i, jp)] + v[idx(i, jm)] - 4 * v0);
  }

  // ----- Wind tendency: du/dt = -f v - r_w u, dv/dt = -∂P/∂y + f u - r_w v -----
  function windTendency(uWind, vWind, j, coriolisScale, windForcing) {
    const f = coriolisAt(j, coriolisScale);
    const dPdy = windForcing * pressureGradientMeridional(j);
    return (u, v) => ({
      du: -f * v - R_WIND * u,
      dv: dPdy + f * u - R_WIND * v
    });
  }

  // ----- Ocean tendency: du_o/dt = -f v_o + τx - r_o u_o + ν∇²u_o, etc. -----
  function oceanTendency(uOcean, vOcean, uWind, vWind, j, coriolisScale, oceanFriction) {
    const f = coriolisAt(j, coriolisScale);
    const r_o = R_OCEAN_BASE * oceanFriction;
    const tauX = ALPHA_STRESS * uWind;
    const tauY = ALPHA_STRESS * vWind;
    return (u, v, lapU, lapV) => ({
      du: -f * v + tauX - r_o * u + NU * lapU,
      dv: f * u + tauY - r_o * v + NU * lapV
    });
  }

  // ----- Single step: advance wind then ocean -----
  function step(uWind, vWind, uOcean, vOcean, landMask, coriolisScale, windForcing, oceanFriction) {
    const uW = uWind.slice();
    const vW = vWind.slice();
    const uO = uOcean.slice();
    const vO = vOcean.slice();

    for (let j = 0; j < NLAT; j++) {
      const getWind = windTendency(uW, vW, j, coriolisScale, windForcing);
      for (let i = 0; i < NLON; i++) {
        const n = idx(i, j);
        if (landMask[n]) continue;
        const { du, dv } = getWind(uW[n], vW[n]);
        uW[n] += DT * du;
        vW[n] += DT * dv;
      }
    }

    for (let j = 0; j < NLAT; j++) {
      for (let i = 0; i < NLON; i++) {
        const n = idx(i, j);
        if (landMask[n]) continue;
        const lapU = laplacianU(uO, i, j);
        const lapV = laplacianV(vO, i, j);
        const getOcean = oceanTendency(uO[n], vO[n], uW[n], vW[n], j, coriolisScale, oceanFriction);
        const { du, dv } = getOcean(uO[n], vO[n], lapU, lapV);
        uO[n] += DT * du;
        vO[n] += DT * dv;
      }
    }

    for (let n = 0; n < landMask.length; n++) {
      if (landMask[n]) {
        uW[n] = 0;
        vW[n] = 0;
        uO[n] = 0;
        vO[n] = 0;
      }
    }

    return { uWind: uW, vWind: vW, uOcean: uO, vOcean: vO };
  }

  // ----- Bilinear interpolation: (lon, lat) in [0,360] x [-90,90] -> u, v -----
  function interpolate(lon, lat, uField, vField) {
    const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * NLON - 0.5;
    const y = ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * NLAT - 0.5;
    const i0 = Math.floor(x);
    const j0 = Math.floor(y);
    const si = Math.max(0, Math.min(1, x - i0));
    const sj = Math.max(0, Math.min(1, y - j0));
    const i1 = (i0 + 1 + NLON) % NLON;
    const j1 = Math.min(NLAT - 1, j0 + 1);
    const j0c = Math.max(0, j0);

    const u00 = uField[idx((i0 + NLON) % NLON, j0c)];
    const u10 = uField[idx(i1, j0c)];
    const u01 = uField[idx((i0 + NLON) % NLON, j1)];
    const u11 = uField[idx(i1, j1)];
    const u = (1 - si) * (1 - sj) * u00 + si * (1 - sj) * u10 + (1 - si) * sj * u01 + si * sj * u11;

    const v00 = vField[idx((i0 + NLON) % NLON, j0c)];
    const v10 = vField[idx(i1, j0c)];
    const v01 = vField[idx((i0 + NLON) % NLON, j1)];
    const v11 = vField[idx(i1, j1)];
    const v = (1 - si) * (1 - sj) * v00 + si * (1 - sj) * v10 + (1 - si) * sj * v01 + si * sj * v11;

    return { u, v };
  }

  // ----- State -----
  let uWind, vWind, uOcean, vOcean;
  let landMask;
  let particles = [];
  let animationId = null;
  let running = false;
  let coriolisScale = 1;
  let windForcing = 1;
  let oceanFriction = 1;

  function initFields() {
    const n = NLON * NLAT;
    uWind = new Float64Array(n);
    vWind = new Float64Array(n);
    uOcean = new Float64Array(n);
    vOcean = new Float64Array(n);
    landMask = buildLandMask();
  }

  function runSteps(count, cScale, wForcing, oFriction) {
    let uW = uWind.slice();
    let vW = vWind.slice();
    let uO = uOcean.slice();
    let vO = vOcean.slice();
    for (let k = 0; k < count; k++) {
      const out = step(uW, vW, uO, vO, landMask, cScale, wForcing, oFriction);
      uW = out.uWind;
      vW = out.vWind;
      uO = out.uOcean;
      vO = out.vOcean;
    }
    uWind = uW;
    vWind = vW;
    uOcean = uO;
    vOcean = vO;
  }

  function reset() {
    initFields();
    runSteps(STEPS_PER_UPDATE, coriolisScale, windForcing, oceanFriction);
    particles = [];
    draw();
  }

  function releaseParticle() {
    if (particles.length >= MAX_PARTICLES) return;
    const lon = Math.random() * 360;
    const lat = (Math.random() - 0.5) * 160;
    particles.push({ lon, lat, trail: [] });
  }

  function advanceParticles() {
    const dt = 0.5;
    for (const p of particles) {
      const { u, v } = interpolate(p.lon, p.lat, uOcean, vWind);
      const dLon = (u * dt) / (111e3 * 1000 * Math.cos((p.lat * Math.PI) / 180)) * (360 / (2 * Math.PI * 6371e3 / 1000));
      const dLat = (v * dt) / (111e3 * 1000) * (180 / (Math.PI * 6371e3 / 1000));
      p.lon = (p.lon + dLon * 0.01 + 360) % 360;
      p.lat = Math.max(-89, Math.min(89, p.lat + dLat * 0.01));
      p.trail.push({ lon: p.lon, lat: p.lat });
      if (p.trail.length > PARTICLE_TRAIL_LENGTH) p.trail.shift();
    }
  }

  // Scale velocity to grid units per step for particle motion (simplified)
  function advanceParticlesSimple() {
    const scale = 0.15;
    for (const p of particles) {
      const { u, v } = interpolate(p.lon, p.lat, uOcean, vOcean);
      p.lon = (p.lon + u * scale + 360) % 360;
      p.lat = Math.max(-89, Math.min(89, p.lat + v * scale));
      p.trail.push({ lon: p.lon, lat: p.lat });
      if (p.trail.length > PARTICLE_TRAIL_LENGTH) p.trail.shift();
    }
  }

  // ----- Canvas -----
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const CW = 1000;
  const CH = 500;

  function lonLatToXY(lon, lat) {
    const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * CW;
    const y = CH - ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * CH;
    return { x, y };
  }

  function draw() {
    ctx.fillStyle = '#0d2137';
    ctx.fillRect(0, 0, CW, CH);

    const n = NLON * NLAT;
    const cellW = CW / NLON;
    const cellH = CH / NLAT;

    for (let j = 0; j < NLAT; j++) {
      for (let i = 0; i < NLON; i++) {
        const n_ = idx(i, j);
        if (landMask[n_]) {
          ctx.fillStyle = '#6b7280';
          ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
        } else {
          const mag = Math.sqrt(uOcean[n_] ** 2 + vOcean[n_] ** 2);
          const t = Math.min(1, mag / 0.5);
          const r = Math.floor(30 + t * 100);
          const g = Math.floor(80 + t * 80);
          const b = Math.floor(150 + t * 105);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    const windScale = 8;
    const oceanScale = 25;
    const arrowLen = 4;

    for (let j = 0; j < NLAT; j++) {
      for (let i = 0; i < NLON; i++) {
        const n_ = idx(i, j);
        if (landMask[n_]) continue;
        const cx = (i + 0.5) * cellW;
        const cy = (j + 0.5) * cellH;
        const uw = uWind[n_];
        const vw = vWind[n_];
        const uo = uOcean[n_];
        const vo = vOcean[n_];
        const drawArrow = (ux, uy, scale, color) => {
          const dx = ux * scale;
          const dy = -uy * scale;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const nx = dx / len;
          const ny = dy / len;
          const tipX = cx + dx;
          const tipY = cy + dy;
          const backX = cx + dx - nx * arrowLen - ny * 2;
          const backY = cy + dy - ny * arrowLen + nx * 2;
          const backX2 = cx + dx - nx * arrowLen + ny * 2;
          const backY2 = cy + dy - ny * arrowLen - nx * 2;
          ctx.strokeStyle = color;
          ctx.fillStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(tipX, tipY);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(tipX, tipY);
          ctx.lineTo(backX, backY);
          ctx.lineTo(backX2, backY2);
          ctx.closePath();
          ctx.fill();
        };
        drawArrow(uw, vw, windScale, 'rgba(255,255,255,0.9)');
        drawArrow(uo, vo, oceanScale, 'rgba(0,255,255,0.85)');
      }
    }

    for (const p of particles) {
      if (p.trail.length >= 2) {
        ctx.strokeStyle = 'rgba(255,235,59,0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const first = lonLatToXY(p.trail[0].lon, p.trail[0].lat);
        ctx.moveTo(first.x, first.y);
        for (let t = 1; t < p.trail.length; t++) {
          const pt = lonLatToXY(p.trail[t].lon, p.trail[t].lat);
          ctx.lineTo(pt.x, pt.y);
        }
        ctx.stroke();
      }
      const { x, y } = lonLatToXY(p.lon, p.lat);
      ctx.fillStyle = '#ffeb3b';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function tick() {
    advanceParticlesSimple();
    draw();
    if (running) animationId = requestAnimationFrame(tick);
  }

  function play() {
    if (running) return;
    running = true;
    tick();
  }

  function pause() {
    running = false;
    if (animationId) cancelAnimationFrame(animationId);
  }

  function recompute() {
    const c = parseFloat(document.getElementById('coriolis').value);
    const w = parseFloat(document.getElementById('wind-forcing').value);
    const o = parseFloat(document.getElementById('ocean-friction').value);
    coriolisScale = c;
    windForcing = w;
    oceanFriction = o;
    initFields();
    runSteps(STEPS_PER_UPDATE, coriolisScale, windForcing, oceanFriction);
    draw();
  }

  document.getElementById('coriolis').addEventListener('input', function () {
    document.getElementById('coriolis-value').textContent = parseFloat(this.value).toFixed(2);
    recompute();
  });
  document.getElementById('wind-forcing').addEventListener('input', function () {
    document.getElementById('wind-forcing-value').textContent = parseFloat(this.value).toFixed(1);
    recompute();
  });
  document.getElementById('ocean-friction').addEventListener('input', function () {
    document.getElementById('ocean-friction-value').textContent = parseFloat(this.value).toFixed(1);
    recompute();
  });
  document.getElementById('btn-play').addEventListener('click', play);
  document.getElementById('btn-pause').addEventListener('click', pause);
  document.getElementById('btn-reset').addEventListener('click', reset);
  document.getElementById('btn-particle').addEventListener('click', releaseParticle);

  initFields();
  runSteps(STEPS_PER_UPDATE, 1, 1, 1);
  draw();
})();
