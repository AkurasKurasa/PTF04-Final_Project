async function loadProjects() {
  const res = await fetch("pages/manifest.json");
  if (!res.ok) return [];
  const list = await res.json();
  return list;
}

function extractDescription(html) {
  const tmp = document.createElement("div");
  tmp.innerHTML = html;
  // first <p> in any markdown cell
  const p = tmp.querySelector(".text_cell_render p, .jp-MarkdownOutput p, p");
  if (p && p.textContent.trim()) {
    let txt = p.textContent.trim().replace(/\s+/g, " ");
    if (txt.length > 200) txt = txt.slice(0, 197).trimEnd() + "…";
    return txt;
  }
  return "";
}

function countCells(html) {
  const tmp = document.createElement("div");
  tmp.innerHTML = html;
  return tmp.querySelectorAll(".cell").length;
}

const state = {
  projects: [],
  current: -1,
  authorMode: (() => {
    const url = new URLSearchParams(location.search);
    if (url.get("author") === "1") {
      localStorage.setItem("authorMode", "1");
      return true;
    }
    if (url.get("author") === "0") {
      localStorage.removeItem("authorMode");
      return false;
    }
    return localStorage.getItem("authorMode") === "1";
  })(),
};

async function preloadProjectMeta() {
  for (const p of state.projects) {
    if (p._loaded) continue;
    p._html = await fetch(`pages/${p.file}`).then((r) => r.text());
    p._desc = extractDescription(p._html);
    p._cells = countCells(p._html);
    p._loaded = true;
  }
}

function renderStats() {
  const totalCells = state.projects.reduce((s, p) => s + (p._cells || 0), 0);
  animateCount(document.getElementById("stat-projects"), state.projects.length, 900, (n) => String(n).padStart(2, "0"));
  animateCount(document.getElementById("stat-cells"), totalCells, 1400);
}

function animateCount(el, target, duration, fmt) {
  if (!el) return;
  const start = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const v = Math.round(target * ease(t));
    el.textContent = fmt ? fmt(v) : String(v);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function duplicateMarquee() {
  const track = document.querySelector(".tools-track");
  if (!track) return;
  track.innerHTML += track.innerHTML;
}

function renderProjectList() {
  const list = document.getElementById("project-list");
  document.getElementById("proj-count").textContent =
    `${String(state.projects.length).padStart(2, "0")} projects`;

  list.innerHTML = state.projects
    .map(
      (p, i) => `
      <li class="project-item" data-idx="${i}">
        <a class="project-link" href="#${i + 1}">
          <span class="project-num">${String(i + 1).padStart(2, "0")}</span>
          <div class="project-text">
            <div class="project-title">${p.title}</div>
            <p class="project-desc">${p._desc || "Click to read the notebook."}</p>
          </div>
          <span class="project-arrow">→</span>
        </a>
      </li>
    `
    )
    .join("");

  list.querySelectorAll(".project-item").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      showDetail(parseInt(el.dataset.idx, 10));
    });
  });
}

function showView(viewId) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("view-active"));
  document.getElementById(viewId).classList.add("view-active");
  window.scrollTo({ top: 0, behavior: "instant" });
}

function showHome() {
  state.current = -1;
  history.replaceState(null, "", "#");
  showView("home");
}

function projectSlug(proj) {
  return proj.file.replace(/\.html$/i, "").toLowerCase();
}

async function reloadCurrent(force = false) {
  const proj = state.projects[state.current];
  if (!proj) return;
  const slug = projectSlug(proj);
  const stage = document.getElementById("notebook-stage");
  const reloadBtn = document.getElementById("reload-btn");

  if (reloadBtn) {
    reloadBtn.disabled = true;
    reloadBtn.querySelector("span:last-child").textContent = "Reloading…";
  }

  try {
    const res = await fetch(`/api/refresh/${slug}`, { method: "POST" });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "Reload failed");

    if (body.regenerated || force) {
      const html = await fetch(`/pages/${body.file}?t=${Date.now()}`).then((r) => r.text());
      proj._html = html;
      stage.innerHTML = html;
      enhanceCellsForEditing(stage, slug);
    }
  } catch (err) {
    console.error("Reload failed:", err);
  } finally {
    if (reloadBtn) {
      reloadBtn.disabled = false;
      reloadBtn.querySelector("span:last-child").textContent = "Reload";
    }
  }
}

function runAllCurrent() {
  const proj = state.projects[state.current];
  if (!proj) return;
  const btn = document.getElementById("run-all-btn");
  const stage = document.getElementById("notebook-stage");
  const slug = projectSlug(proj);

  btn.disabled = true;
  btn.classList.add("running");
  btn.querySelector("span:last-child").textContent = "Running…";

  // Mark each cell as queued
  const cells = stage.querySelectorAll(".cell");
  cells.forEach((c, i) => {
    c.classList.remove("running", "cell-ok", "cell-failed");
    c.classList.add("queued");
    c.dataset.cellIndex = i;
  });

  // Progress banner
  let done = 0, total = cells.length, codeTotal = 0, errCount = 0;
  const banner = document.createElement("div");
  banner.className = "run-banner running-banner";
  banner.innerHTML = `
    <div class="run-spinner"></div>
    <div class="run-banner-text">
      <strong class="run-status">Booting kernel…</strong>
      <span class="run-meta"></span>
    </div>
    <div class="run-bar"><div class="run-bar-fill" id="run-bar-fill"></div></div>
  `;
  stage.prepend(banner);

  const updateMeta = () => {
    const status = banner.querySelector(".run-status");
    const meta = banner.querySelector(".run-meta");
    const bar = banner.querySelector(".run-bar-fill");
    status.textContent = `Executing cell ${done} / ${codeTotal || "?"}`;
    meta.textContent = errCount > 0 ? `${errCount} error${errCount === 1 ? "" : "s"} so far` : "";
    if (codeTotal > 0) bar.style.width = `${(done / codeTotal) * 100}%`;
  };

  const es = new EventSource(`/api/run-stream/${slug}`);

  es.onmessage = async (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    if (data.event === "start") {
      total = data.total;
      codeTotal = data.code_total;
      updateMeta();
    } else if (data.event === "cell-skip") {
      const c = stage.querySelector(`.cell[data-cell-index="${data.index}"]`);
      c?.classList.remove("queued");
    } else if (data.event === "cell-start") {
      const c = stage.querySelector(`.cell[data-cell-index="${data.index}"]`);
      if (c) {
        c.classList.remove("queued");
        c.classList.add("running");
        c.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    } else if (data.event === "cell-done") {
      done = data.code_idx;
      if (data.error) errCount++;
      const c = stage.querySelector(`.cell[data-cell-index="${data.index}"]`);
      if (c) {
        c.classList.remove("running");
        c.classList.add(data.error ? "cell-failed" : "cell-ok");
      }
      updateMeta();
    } else if (data.event === "done") {
      es.close();
      const html = await fetch(`/pages/${data.file}?t=${Date.now()}`).then((r) => r.text());
      proj._html = html;
      stage.innerHTML = html;
      enhanceCellsForEditing(stage, slug);

      const final = document.createElement("div");
      if (errCount > 0) {
        final.className = "run-banner err";
        final.innerHTML = `⚠ Done with <strong>${errCount}</strong> error${errCount === 1 ? "" : "s"}. Tracebacks rendered in cell outputs.`;
      } else {
        final.className = "run-banner ok";
        final.textContent = `✓ All ${codeTotal} cells executed cleanly.`;
        setTimeout(() => final.remove(), 5000);
      }
      stage.prepend(final);

      btn.disabled = false;
      btn.classList.remove("running");
      btn.querySelector("span:last-child").textContent = "Run All Cells";
    } else if (data.event === "error") {
      es.close();
      banner.className = "run-banner err";
      banner.innerHTML = `⚠ ${data.msg}`;
      btn.disabled = false;
      btn.classList.remove("running");
      btn.querySelector("span:last-child").textContent = "Run All Cells";
    }
  };

  es.onerror = () => {
    es.close();
    banner.className = "run-banner err";
    banner.innerHTML = "⚠ Connection lost.";
    btn.disabled = false;
    btn.classList.remove("running");
    btn.querySelector("span:last-child").textContent = "Run All Cells";
  };
}

const LAB_BASE = "http://127.0.0.1:8888";

async function loadDemo(proj) {
  const stage = document.getElementById("demo-stage");
  stage.innerHTML = `<div class="demo-loading">Loading demo…</div>`;
  const slug = projectSlug(proj);

  // Try schema-driven custom GUI
  try {
    const res = await fetch(`/api/${slug}/schema`);
    if (res.ok) {
      const schema = await res.json();
      renderSchemaDemo(stage, slug, schema, proj);
      return;
    }
    if (res.status === 404) {
      // Schema not configured — fall through to JupyterLab iframe
    }
  } catch (e) {
    // Backend offline (e.g. on Vercel) — show static notice
    stage.innerHTML = `
      <div class="demo-empty">
        <h3 style="margin:0 0 8px;font-family:var(--serif);font-size:1.4rem;">Live demo unavailable on this host</h3>
        <p style="max-width:480px;margin:0 auto 12px;">
          The interactive demo requires a Python backend (Flask + TensorFlow).
          Run the project locally to use the live model:
        </p>
        <code style="display:block;max-width:540px;margin:8px auto;padding:10px 14px;background:var(--bg-soft);border-radius:8px;font-size:0.84rem;">
          python backend/app.py
        </code>
      </div>
    `;
    return;
  }

  // Fallback: JupyterLab iframe
  const nbName = proj.file.replace(/\.html$/i, ".ipynb");
  const url = `${LAB_BASE}/notebooks/${encodeURIComponent(nbName)}`;
  try {
    await fetch(`${LAB_BASE}/lab`, { mode: "no-cors" });
  } catch (e) {
    stage.innerHTML = `<p class="demo-empty">JupyterLab not running. Start:<br><code>jupyter lab --no-browser --port=8888 --ServerApp.token='' --notebook-dir=notebooks</code></p>`;
    return;
  }

  stage.innerHTML = `
    <div class="voila-wrap">
      <div class="voila-toolbar">
        <span class="voila-status"><span class="dot"></span>Live kernel · ${proj.title}</span>
        <span class="voila-hint">Use <kbd>▶</kbd> to run cell · <kbd>Run ▸ Run All Cells</kbd> for full notebook</span>
        <a href="${url}" target="_blank" rel="noopener" class="voila-open">Open in new tab ↗</a>
      </div>
      <iframe class="voila-frame" src="${url}" title="${proj.title}"></iframe>
    </div>
  `;
}

function fieldHTML(f, slug) {
  const id = `f-${slug}-${f.name}`;
  if (f.type === "textarea") {
    return `
      <div class="sd-field sd-field-textarea">
        <label for="${id}">
          <span class="sd-label">${f.label}</span>
          ${f.hint ? `<span class="sd-hint">${f.hint}</span>` : ""}
        </label>
        <textarea
          id="${id}"
          name="${f.name}"
          rows="${f.rows || 5}"
          ${f.required ? "required" : ""}
          placeholder="${f.placeholder || ""}"
          class="sd-textarea"
        >${f.default || ""}</textarea>
      </div>
    `;
  }
  if (f.type === "image") {
    return `
      <div class="sd-field sd-field-image">
        <div class="sd-field-head">
          <label for="${id}">
            <span class="sd-label">${f.label}</span>
            ${f.hint ? `<span class="sd-hint">${f.hint}</span>` : ""}
          </label>
        </div>
        <label class="sd-drop" id="${id}-drop" for="${id}">
          <input
            id="${id}"
            name="${f.name}"
            type="file"
            accept="${f.accept || "image/*"}"
            ${f.required ? "required" : ""}
            hidden
          />
          <div class="sd-drop-empty" id="${id}-empty">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div class="sd-drop-msg">Click or drop an image</div>
            <div class="sd-drop-sub">${f.accept || "Any image"}</div>
          </div>
          <img class="sd-drop-preview" id="${id}-preview" alt="" />
          <button type="button" class="sd-drop-clear" id="${id}-clear" aria-label="Remove">×</button>
        </label>
      </div>
    `;
  }
  if (f.type === "slider") {
    return `
      <div class="sd-field">
        <div class="sd-field-head">
          <label for="${id}">
            <span class="sd-label">${f.label}</span>
            ${f.hint ? `<span class="sd-hint">${f.hint}</span>` : ""}
          </label>
          <output class="sd-output" id="${id}-out">${f.default}<span class="sd-unit">${f.unit || ""}</span></output>
        </div>
        <input
          id="${id}"
          name="${f.name}"
          type="range"
          min="${f.min}"
          max="${f.max}"
          step="${f.step ?? 1}"
          value="${f.default}"
          class="sd-slider"
        />
        <div class="sd-scale">
          <span>${f.min}${f.unit || ""}</span>
          <span>${f.max}${f.unit || ""}</span>
        </div>
      </div>
    `;
  }
  // fallback: number / text
  const attrs = [
    f.type === "number" ? `type="number"` : `type="text"`,
    f.min !== undefined ? `min="${f.min}"` : "",
    f.max !== undefined ? `max="${f.max}"` : "",
    f.step !== undefined ? `step="${f.step}"` : "",
    f.default !== undefined ? `value="${f.default}"` : "",
  ].filter(Boolean).join(" ");
  return `
    <div class="sd-field">
      <label for="${id}"><span class="sd-label">${f.label}</span></label>
      <input id="${id}" name="${f.name}" ${attrs} />
    </div>
  `;
}

const BOSTON_HOUSES = [
  { url: "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=400&q=70", label: "Brownstone · Beacon Hill", tier: "mid" },
  { url: "https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=400&q=70", label: "Suburban · Cambridge", tier: "low" },
  { url: "https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=400&q=70", label: "Modern · Back Bay", tier: "high" },
  { url: "https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?auto=format&fit=crop&w=400&q=70", label: "Cottage · Roxbury", tier: "low" },
  { url: "https://images.unsplash.com/photo-1572120360610-d971b9d7767c?auto=format&fit=crop&w=400&q=70", label: "Townhouse · South End", tier: "mid" },
  { url: "https://images.unsplash.com/photo-1613490493576-7fde63acd811?auto=format&fit=crop&w=400&q=70", label: "Estate · Brookline", tier: "high" },
];

function pickHouseByPrice(usd) {
  // Tier thresholds (model output is in MEDV * 1000, very low for this dataset)
  const tier = usd > 30000 ? "high" : usd > 15000 ? "mid" : "low";
  const pool = BOSTON_HOUSES.filter((h) => h.tier === tier);
  return pool[Math.floor(Math.random() * pool.length)] || BOSTON_HOUSES[0];
}

function renderSchemaDemo(stage, slug, schema, proj) {
  const fields = (schema.fields || []).map((f) => fieldHTML(f, slug)).join("");
  const theme = THEMES[schema.theme] || THEMES.default;
  const layout = schema.layout || "side";

  const isBoston = slug === "boston_house_price";
  const galleryHtml = isBoston ? `
    <div class="sd-gallery">
      <div class="sd-gallery-label">Sampled Boston neighbourhoods</div>
      <div class="sd-gallery-grid">
        ${BOSTON_HOUSES.slice(0, 4).map((h, i) => `
          <figure class="sd-house" style="animation-delay:${i * 80}ms">
            <img src="${h.url}" alt="${h.label}" loading="lazy" />
            <figcaption>${h.label}</figcaption>
          </figure>
        `).join("")}
      </div>
    </div>
  ` : "";

  const iconFn = ICONS[schema.icon] || svgChip;
  const themeStyle = `--sd-bg1:${theme.bg1};--sd-bg2:${theme.bg2};--sd-glow1:${theme.glow1};--sd-glow2:${theme.glow2};--accent:${theme.accent}`;

  const sideHTML = `
    <aside class="sd-side">
      <div class="sd-icon">${iconFn()}</div>
      <h3 class="sd-title">${schema.title}</h3>
      ${schema.subtitle ? `<p class="sd-subtitle">${schema.subtitle}</p>` : ""}
      ${schema.description ? `<p class="sd-desc">${schema.description}</p>` : ""}
      ${galleryHtml}
      <div class="sd-tag-row">
        <span class="sd-tag">Live model</span>
        <span class="sd-tag">No mock</span>
        <span class="sd-tag">${(schema.fields || []).length} input${(schema.fields || []).length === 1 ? "" : "s"}</span>
      </div>
    </aside>
  `;

  const formHTML = `
    <form class="sd-form" id="sd-form-${slug}">
      <div class="sd-fields ${layout === "grid" ? "is-grid" : ""}">${fields}</div>
      <div class="sd-actions">
        <button type="submit" class="sd-submit">${schema.submit_label || "Predict"}</button>
        <button type="button" class="sd-reset">Reset</button>
      </div>
      <div class="sd-result" id="sd-result-${slug}">
        <div class="sd-result-empty">Adjust the inputs, then hit predict.</div>
      </div>
    </form>
  `;

  // Layouts: side (default), stacked (no side panel), duel (cat/dog dual pane)
  let bodyHTML;
  if (layout === "stacked") {
    bodyHTML = `
      <div class="sd-stacked-head">
        <div class="sd-icon">${iconFn()}</div>
        <h3 class="sd-title">${schema.title}</h3>
        ${schema.subtitle ? `<p class="sd-subtitle">${schema.subtitle}</p>` : ""}
        ${schema.description ? `<p class="sd-desc">${schema.description}</p>` : ""}
      </div>
      ${formHTML}
    `;
  } else {
    bodyHTML = `${sideHTML}${formHTML}`;
  }

  stage.innerHTML = `
    <div class="sd-wrap layout-${layout}" style="${themeStyle}">
      ${bodyHTML}
    </div>
  `;

  // Wire slider live output
  const form = document.getElementById(`sd-form-${slug}`);
  (schema.fields || []).forEach((f) => {
    if (f.type === "slider") {
      const inp = form.querySelector(`#f-${slug}-${f.name}`);
      const out = form.querySelector(`#f-${slug}-${f.name}-out`);
      inp.addEventListener("input", () => {
        const v = parseFloat(inp.value);
        const decimals = (f.step || 1).toString().split(".")[1]?.length || 0;
        out.innerHTML = `${v.toFixed(decimals)}<span class="sd-unit">${f.unit || ""}</span>`;
      });
    }
    if (f.type === "image") {
      const inp = form.querySelector(`#f-${slug}-${f.name}`);
      const drop = form.querySelector(`#f-${slug}-${f.name}-drop`);
      const empty = form.querySelector(`#f-${slug}-${f.name}-empty`);
      const preview = form.querySelector(`#f-${slug}-${f.name}-preview`);
      const clear = form.querySelector(`#f-${slug}-${f.name}-clear`);

      const showFile = (file) => {
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.classList.add("active");
        empty.classList.add("hidden");
        clear.classList.add("active");
      };
      const reset = () => {
        inp.value = "";
        preview.src = "";
        preview.classList.remove("active");
        empty.classList.remove("hidden");
        clear.classList.remove("active");
      };

      inp.addEventListener("change", () => {
        if (inp.files && inp.files[0]) showFile(inp.files[0]);
      });
      clear.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); reset(); });
      ["dragenter", "dragover"].forEach((evt) =>
        drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.add("hover"); })
      );
      ["dragleave", "drop"].forEach((evt) =>
        drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.remove("hover"); })
      );
      drop.addEventListener("drop", (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files?.[0];
        if (!file) return;
        const dt = new DataTransfer();
        dt.items.add(file);
        inp.files = dt.files;
        showFile(file);
      });
    }
  });

  // Reset
  form.querySelector(".sd-reset").addEventListener("click", () => {
    (schema.fields || []).forEach((f) => {
      const inp = form.querySelector(`#f-${slug}-${f.name}`);
      if (!inp) return;
      if (f.type === "image") {
        inp.value = "";
        const empty = form.querySelector(`#f-${slug}-${f.name}-empty`);
        const preview = form.querySelector(`#f-${slug}-${f.name}-preview`);
        const clear = form.querySelector(`#f-${slug}-${f.name}-clear`);
        if (preview) { preview.src = ""; preview.classList.remove("active"); }
        if (empty) empty.classList.remove("hidden");
        if (clear) clear.classList.remove("active");
      } else if (f.default !== undefined) {
        inp.value = f.default;
        inp.dispatchEvent(new Event("input"));
      } else {
        inp.value = "";
      }
    });
    const r = document.getElementById(`sd-result-${slug}`);
    r.innerHTML = `<div class="sd-result-empty">Adjust the inputs, then hit predict.</div>`;
  });

  // Submit
  const hasFile = (schema.fields || []).some((f) => f.type === "image" || f.type === "file");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const result = document.getElementById(`sd-result-${slug}`);
    result.innerHTML = `
      <div class="sd-result-loading">
        <div class="sd-bar"><div class="sd-bar-inner"></div></div>
        <div class="sd-loading-text">Running model…</div>
      </div>
    `;
    try {
      let res;
      if (hasFile) {
        // multipart/form-data — let browser set Content-Type
        const fd = new FormData(form);
        res = await fetch(`/api/${slug}/predict`, { method: "POST", body: fd });
      } else {
        const data = Object.fromEntries(new FormData(form).entries());
        res = await fetch(`/api/${slug}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
      }
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || "Prediction failed");
      renderPrediction(result, body, schema);
    } catch (err) {
      result.innerHTML = `<div class="sd-error">⚠ ${err.message}</div>`;
    }
  });
}

function animateValue(el, from, to, duration, fmt) {
  const start = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const v = from + (to - from) * ease(t);
    el.textContent = fmt(v);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderPrediction(el, body, schema) {
  // Sentiment gauge
  if (schema.output?.type === "sentiment" && body.score !== undefined) {
    const score = body.score;
    const polarity = body.polarity ?? (score - 0.5) * 2;
    const pct = Math.round(score * 100);
    const isPos = score > 0.5;
    const emoji = score > 0.85 ? "🤩" : score > 0.65 ? "😀" : score > 0.5 ? "🙂" :
                  score > 0.35 ? "😐" : score > 0.15 ? "😕" : "😡";
    el.innerHTML = `
      <div class="sd-sentiment-card">
        <div class="sd-sentiment-emoji">${emoji}</div>
        <div class="sd-sentiment-label ${isPos ? "pos" : "neg"}">${body.label.toUpperCase()}</div>
        <div class="sd-sentiment-meta">${pct}% confidence · ${body.tokens_used || 0} tokens</div>
        <div class="sd-sentiment-gauge">
          <div class="sd-gauge-track">
            <div class="sd-gauge-mid"></div>
            <div class="sd-gauge-fill" style="left: 50%; width: ${Math.abs(polarity) * 50}%; ${polarity < 0 ? "transform: translateX(-100%);" : ""}"></div>
            <div class="sd-gauge-marker" style="left: ${pct}%"></div>
          </div>
          <div class="sd-gauge-scale">
            <span>negative</span><span>neutral</span><span>positive</span>
          </div>
        </div>
      </div>
    `;
    return;
  }

  // Duel (cat vs dog)
  if (schema.output?.type === "duel" && body.scores) {
    const labels = schema.output.labels || Object.keys(body.scores);
    const [a, b] = labels;
    const sa = body.scores[a] ?? 0;
    const sb = body.scores[b] ?? 0;
    const winner = sa > sb ? a : b;
    el.innerHTML = `
      <div class="sd-duel">
        <div class="sd-duel-side ${winner === a ? "win" : ""}">
          <div class="sd-duel-emoji">${a === "cat" ? "🐱" : a === "dog" ? "🐶" : "A"}</div>
          <div class="sd-duel-name">${a}</div>
          <div class="sd-duel-score" id="sd-duel-a">0%</div>
        </div>
        <div class="sd-duel-vs">vs</div>
        <div class="sd-duel-side ${winner === b ? "win" : ""}">
          <div class="sd-duel-emoji">${b === "cat" ? "🐱" : b === "dog" ? "🐶" : "B"}</div>
          <div class="sd-duel-name">${b}</div>
          <div class="sd-duel-score" id="sd-duel-b">0%</div>
        </div>
        <div class="sd-duel-bar">
          <div class="sd-duel-bar-a" style="width: ${(sa * 100).toFixed(1)}%"></div>
          <div class="sd-duel-bar-b" style="width: ${(sb * 100).toFixed(1)}%"></div>
        </div>
        <div class="sd-duel-verdict">It's a <strong>${winner}</strong>.</div>
      </div>
    `;
    animateValue(document.getElementById("sd-duel-a"), 0, sa * 100, 700, (v) => v.toFixed(1) + "%");
    animateValue(document.getElementById("sd-duel-b"), 0, sb * 100, 700, (v) => v.toFixed(1) + "%");
    return;
  }

  // Simple currency (single line, no inputs grid)
  if (schema.output?.type === "currency_simple" && body.value !== undefined) {
    el.innerHTML = `
      <div class="sd-price-card sd-price-simple">
        <div class="sd-price-content">
          <div class="sd-price-eyebrow">Predicted price</div>
          <div class="sd-price-num"><span class="sd-currency">$</span><span id="sd-price-target">0</span></div>
          ${body.slope !== undefined ? `<div class="sd-price-meta">y = <code>${body.slope.toFixed(3)}</code> · x + <code>${body.intercept.toFixed(3)}</code> &middot; x = ${body.x}</div>` : ""}
        </div>
      </div>
    `;
    animateValue(document.getElementById("sd-price-target"), 0, body.value, 800,
                 (v) => Math.round(v).toLocaleString("en-US"));
    return;
  }

  if (schema.output?.type === "currency" || body.currency) {
    const usd = body.value;
    const house = pickHouseByPrice(usd);
    el.innerHTML = `
      <div class="sd-price-card">
        <div class="sd-price-bg" style="background-image:url('${house.url}')"></div>
        <div class="sd-price-content">
          <div class="sd-price-eyebrow">Predicted Median Price</div>
          <div class="sd-price-num"><span class="sd-currency">$</span><span id="sd-price-target">0</span></div>
          <div class="sd-price-tier">≈ ${house.label}</div>
          <div class="sd-price-meta">Model output (MEDV): <code>${body.raw_medv?.toFixed(3) ?? "—"}</code> &middot; ×1000</div>
          <div class="sd-price-inputs">
            ${Object.entries(body.inputs || {}).map(([k, v]) => `
              <div class="sd-input-pill"><span class="ip-name">${k.toUpperCase()}</span><span class="ip-val">${v}</span></div>
            `).join("")}
          </div>
        </div>
      </div>
    `;
    const target = document.getElementById("sd-price-target");
    animateValue(target, 0, usd, 900, (v) =>
      Math.round(v).toLocaleString("en-US")
    );
    return;
  }

  // Label + probabilities (Iris / CIFAR path)
  if (body.label !== undefined) {
    let probaHtml = "";
    if (body.probabilities) {
      const entries = Object.entries(body.probabilities).sort((a, b) => b[1] - a[1]);
      probaHtml = `
        <div class="proba-list">
          ${entries.map(([cls, p], i) => `
            <div class="proba-row ${i === 0 ? "top" : ""}">
              <div class="proba-name">${cls}</div>
              <div class="proba-bar"><div class="proba-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
              <div class="proba-val">${(p * 100).toFixed(1)}%</div>
            </div>
          `).join("")}
        </div>
      `;
    }
    const conf = body.confidence !== undefined ? ` · ${(body.confidence * 100).toFixed(1)}% confident` : "";
    el.innerHTML = `
      <div class="sd-price-card">
        <div class="sd-price-content">
          <div class="sd-price-eyebrow">Prediction${conf}</div>
          <div class="sd-label-big">${body.label}</div>
          ${probaHtml}
        </div>
      </div>
    `;
    return;
  }

  el.innerHTML = `<pre class="result-raw">${JSON.stringify(body, null, 2)}</pre>`;
}

function svgHouse() {
  return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path d="M3 11.5L12 4l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1V11.5Z"
      stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" fill="rgba(185,74,42,0.12)"/>
  </svg>`;
}
function svgChip() {
  return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <rect x="6" y="6" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.6" fill="rgba(185,74,42,0.10)"/>
    <path d="M9 3v3M12 3v3M15 3v3M9 18v3M12 18v3M15 18v3M3 9h3M3 12h3M3 15h3M18 9h3M18 12h3M18 15h3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
  </svg>`;
}
function svgShirt() {
  return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path d="M4 7 L9 3 L11 5 H13 L15 3 L20 7 L18 10 L16 9 V21 H8 V9 L6 10 Z"
      stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" fill="rgba(185,74,42,0.12)"/>
  </svg>`;
}
function svgPaw() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
    <ellipse cx="6" cy="11" rx="2" ry="2.6"/><ellipse cx="10" cy="7" rx="2" ry="2.6"/>
    <ellipse cx="14" cy="7" rx="2" ry="2.6"/><ellipse cx="18" cy="11" rx="2" ry="2.6"/>
    <path d="M9 13c-2 1-3 3-3 5 0 2 2 3 4 3h4c2 0 4-1 4-3 0-2-1-4-3-5-1.5-1-4.5-1-6 0z" fill="rgba(185,74,42,0.14)"/>
  </svg>`;
}
function svgWave() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
    <path d="M3 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/>
    <path d="M3 17c2-3 4-3 6 0s4 3 6 0 4-3 6 0" opacity="0.4"/>
  </svg>`;
}
function svgFlower() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
    <circle cx="12" cy="12" r="2.2" fill="rgba(185,74,42,0.4)"/>
    <ellipse cx="12" cy="6" rx="2.6" ry="3.5" fill="rgba(185,74,42,0.12)"/>
    <ellipse cx="12" cy="18" rx="2.6" ry="3.5" fill="rgba(185,74,42,0.12)"/>
    <ellipse cx="6" cy="12" rx="3.5" ry="2.6" fill="rgba(185,74,42,0.12)"/>
    <ellipse cx="18" cy="12" rx="3.5" ry="2.6" fill="rgba(185,74,42,0.12)"/>
  </svg>`;
}
function svgLine() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
    <path d="M3 20 L9 14 L13 17 L21 6"/>
    <circle cx="9" cy="14" r="1.5" fill="currentColor"/>
    <circle cx="13" cy="17" r="1.5" fill="currentColor"/>
  </svg>`;
}
function svgGrid() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
    <rect x="4" y="4" width="6" height="6" rx="1.5" fill="rgba(185,74,42,0.12)"/>
    <rect x="14" y="4" width="6" height="6" rx="1.5"/>
    <rect x="4" y="14" width="6" height="6" rx="1.5"/>
    <rect x="14" y="14" width="6" height="6" rx="1.5" fill="rgba(185,74,42,0.12)"/>
  </svg>`;
}
function svgCurve() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
    <path d="M3 19 C 7 19 9 4 13 4 C 17 4 18 19 21 19"/>
  </svg>`;
}
function svgTrophy() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
    <path d="M8 4 H16 V8 C 16 12 14 14 12 14 C 10 14 8 12 8 8 Z" fill="rgba(185,74,42,0.16)"/>
    <path d="M5 6 H8 M16 6 H19 M5 6 V8 C 5 10 7 10 8 10 M19 6 V8 C 19 10 17 10 16 10"/>
    <path d="M9 18 H15 M10 14 V18 M14 14 V18 M8 21 H16" stroke-linecap="round"/>
  </svg>`;
}

const ICONS = {
  house: svgHouse, chip: svgChip, shirt: svgShirt, paw: svgPaw,
  wave: svgWave, flower: svgFlower, line: svgLine, grid: svgGrid,
  curve: svgCurve, trophy: svgTrophy,
};

const THEMES = {
  default:  { bg1: "#fbf6ee", bg2: "#f3e9d6", glow1: "rgba(185, 74, 42, 0.18)", glow2: "rgba(132, 125, 108, 0.12)", accent: "#b94a2a" },
  ocean:    { bg1: "#e9f0f6", bg2: "#d0e1ec", glow1: "rgba(52, 122, 165, 0.20)", glow2: "rgba(40, 90, 120, 0.12)",  accent: "#2c6f8e" },
  forest:   { bg1: "#eef3e8", bg2: "#d8e2cb", glow1: "rgba(74, 138, 86, 0.22)",  glow2: "rgba(80, 100, 70, 0.12)",  accent: "#4a8a56" },
  lavender: { bg1: "#f1ecf5", bg2: "#dccfe6", glow1: "rgba(140, 95, 175, 0.22)", glow2: "rgba(110, 90, 130, 0.12)", accent: "#7a5ca8" },
  sunset:   { bg1: "#fdeede", bg2: "#f6c9a4", glow1: "rgba(232, 138, 76, 0.28)", glow2: "rgba(190, 80, 60, 0.18)",  accent: "#d96b3a" },
  navy:     { bg1: "#e6ebf2", bg2: "#c9d2e0", glow1: "rgba(60, 80, 140, 0.22)",  glow2: "rgba(40, 50, 90, 0.12)",   accent: "#3a4a8a" },
  charcoal: { bg1: "#e8e8ea", bg2: "#cbcbd0", glow1: "rgba(80, 80, 90, 0.22)",   glow2: "rgba(50, 50, 60, 0.16)",   accent: "#3c3c45" },
};

function renderDemoForm(stage, slug, schema) {
  const fields = (schema.fields || []).map((f) => {
    const id = `f-${slug}-${f.name}`;
    const attrs = [
      f.type === "number" ? `type="number"` : `type="text"`,
      f.min !== undefined ? `min="${f.min}"` : "",
      f.max !== undefined ? `max="${f.max}"` : "",
      f.step !== undefined ? `step="${f.step}"` : "",
      f.default !== undefined ? `value="${f.default}"` : "",
      f.placeholder ? `placeholder="${f.placeholder}"` : "",
    ].filter(Boolean).join(" ");

    return `
      <div class="demo-field">
        <label for="${id}">
          <span class="demo-label">${f.label}</span>
          ${f.unit ? `<span class="demo-unit">${f.unit}</span>` : ""}
        </label>
        <input id="${id}" name="${f.name}" ${attrs} required />
      </div>
    `;
  }).join("");

  stage.innerHTML = `
    <form class="demo-form" id="demo-form">
      <header class="demo-head">
        <h3>${schema.title || "Live demo"}</h3>
        ${schema.description ? `<p class="demo-sub">${schema.description}</p>` : ""}
      </header>
      <div class="demo-grid">${fields}</div>
      <div class="demo-actions">
        <button type="submit" class="demo-submit">${schema.submit_label || "Predict"}</button>
        <button type="button" class="demo-reset">Reset</button>
      </div>
      <div class="demo-result" id="demo-result"></div>
    </form>
  `;

  const form = document.getElementById("demo-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());
    const result = document.getElementById("demo-result");
    result.innerHTML = `<div class="demo-thinking">Predicting…</div>`;
    try {
      const res = await fetch(`/api/${slug}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || "Prediction failed");
      renderDemoResult(result, body, schema);
    } catch (err) {
      result.innerHTML = `<div class="demo-error">⚠ ${err.message}</div>`;
    }
  });

  form.querySelector(".demo-reset").addEventListener("click", () => {
    form.reset();
    document.getElementById("demo-result").innerHTML = "";
  });
}

function renderDemoResult(el, body, schema) {
  if (body.label !== undefined) {
    let probaHtml = "";
    if (body.probabilities) {
      const entries = Object.entries(body.probabilities).sort((a, b) => b[1] - a[1]);
      probaHtml = `
        <div class="proba-list">
          ${entries.map(([cls, p]) => `
            <div class="proba-row">
              <div class="proba-name">${cls}</div>
              <div class="proba-bar"><div class="proba-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
              <div class="proba-val">${(p * 100).toFixed(1)}%</div>
            </div>
          `).join("")}
        </div>
      `;
    }
    el.innerHTML = `
      <div class="result-card">
        <div class="result-eyebrow">Prediction</div>
        <div class="result-label">${body.label}</div>
        ${probaHtml}
      </div>
    `;
    return;
  }
  if (body.value !== undefined) {
    el.innerHTML = `
      <div class="result-card">
        <div class="result-eyebrow">Prediction</div>
        <div class="result-label">${body.value}</div>
      </div>
    `;
    return;
  }
  el.innerHTML = `<pre class="result-raw">${JSON.stringify(body, null, 2)}</pre>`;
}

function enhanceCellsForEditing(stage, slug) {
  if (!state.authorMode) return; // viewer mode: leave cells as static rendered HTML
  const codeCells = stage.querySelectorAll(".cell.code_cell, .cell.border-box-sizing.code_cell");
  codeCells.forEach((cell, i) => {
    const inputArea = cell.querySelector(".input_area");
    if (!inputArea || cell.dataset.editEnhanced) return;
    cell.dataset.editEnhanced = "1";

    const codeEl = inputArea.querySelector("pre, code");
    const codeText = codeEl ? codeEl.innerText : "";

    const ta = document.createElement("textarea");
    ta.className = "cell-editor";
    ta.spellcheck = false;
    ta.value = codeText;
    ta.rows = Math.max(2, codeText.split("\n").length);

    const actions = document.createElement("div");
    actions.className = "cell-actions";
    actions.innerHTML = `
      <button class="cell-run" title="Run cell (Shift+Enter)">▶ Run</button>
      <span class="cell-status"></span>
    `;

    inputArea.innerHTML = "";
    inputArea.appendChild(actions);
    inputArea.appendChild(ta);

    const autoSize = () => {
      ta.style.height = "auto";
      ta.style.height = ta.scrollHeight + "px";
    };
    ta.addEventListener("input", autoSize);
    setTimeout(autoSize, 0);

    const runBtn = actions.querySelector(".cell-run");
    const statusEl = actions.querySelector(".cell-status");

    const runThis = () => runCell(slug, cell, ta, runBtn, statusEl);
    runBtn.addEventListener("click", runThis);
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.shiftKey || e.ctrlKey)) {
        e.preventDefault();
        runThis();
      }
      if (e.key === "Tab") {
        e.preventDefault();
        const start = ta.selectionStart, end = ta.selectionEnd;
        ta.value = ta.value.slice(0, start) + "    " + ta.value.slice(end);
        ta.selectionStart = ta.selectionEnd = start + 4;
        autoSize();
      }
    });
  });
}

async function runCell(slug, cellEl, ta, runBtn, statusEl) {
  runBtn.disabled = true;
  runBtn.innerHTML = `<span class="cell-spin"></span> Running`;
  statusEl.innerHTML = `<span class="cell-status-running">executing<span class="dots"><span>.</span><span>.</span><span>.</span></span></span>`;
  cellEl.classList.remove("cell-ok", "cell-failed");
  cellEl.classList.add("running");

  // Hide original nbconvert output (so old errors don't linger)
  cellEl.querySelectorAll(".output_wrapper, .output").forEach((el) => {
    el.classList.add("output-hidden-by-rerun");
  });

  // Live output placeholder showing in-progress
  let outWrap = cellEl.querySelector(".cell-live-output");
  if (!outWrap) {
    outWrap = document.createElement("div");
    outWrap.className = "cell-live-output";
    cellEl.appendChild(outWrap);
  }
  outWrap.innerHTML = `
    <div class="cell-live-pending">
      <div class="cell-live-bar"><div class="cell-live-bar-inner"></div></div>
      <div class="cell-live-msg">Kernel busy — awaiting result…</div>
    </div>
  `;

  const t0 = performance.now();

  try {
    const res = await fetch(`/api/cell/${slug}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Author-Mode": state.authorMode ? "1" : "0",
      },
      body: JSON.stringify({ code: ta.value }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "Run failed");

    outWrap.innerHTML = renderOutputs(body.outputs || []);

    const errored = (body.outputs || []).some((o) => o.type === "error");
    cellEl.classList.toggle("cell-failed", errored);
    cellEl.classList.toggle("cell-ok", !errored);
    const ms = Math.round(performance.now() - t0);
    statusEl.innerHTML = errored
      ? `<span class="cell-status-err">✗ error · ${ms}ms</span>`
      : `<span class="cell-status-ok">✓ done · ${ms}ms</span>`;
  } catch (err) {
    outWrap.innerHTML = "";
    statusEl.innerHTML = `<span class="cell-status-err">⚠ ${err.message}</span>`;
    cellEl.classList.add("cell-failed");
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = "▶ Run";
    cellEl.classList.remove("running");
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function ansiToHtml(s) {
  // Strip ANSI escape codes (basic). For full color, library needed.
  return escapeHtml(s.replace(/\x1b\[[0-9;]*m/g, ""));
}

function renderOutputs(outputs) {
  const parts = outputs.map((o) => {
    if (o.type === "stream") {
      const cls = o.name === "stderr" ? "output_stderr" : "output_stream";
      return `<div class="${cls}"><pre>${ansiToHtml(o.text)}</pre></div>`;
    }
    if (o.type === "error") {
      const tb = (o.traceback || []).join("\n");
      return `<div class="output_error"><pre>${ansiToHtml(tb || `${o.ename}: ${o.evalue}`)}</pre></div>`;
    }
    if (o.type === "execute_result" || o.type === "display_data") {
      const data = o.data || {};
      if (data["image/png"]) {
        return `<div class="output_area"><img src="data:image/png;base64,${data["image/png"]}" /></div>`;
      }
      if (data["image/jpeg"]) {
        return `<div class="output_area"><img src="data:image/jpeg;base64,${data["image/jpeg"]}" /></div>`;
      }
      if (data["text/html"]) {
        const html = Array.isArray(data["text/html"]) ? data["text/html"].join("") : data["text/html"];
        return `<div class="output_area output_html">${html}</div>`;
      }
      if (data["text/plain"]) {
        const txt = Array.isArray(data["text/plain"]) ? data["text/plain"].join("") : data["text/plain"];
        return `<div class="output_area"><pre>${ansiToHtml(txt)}</pre></div>`;
      }
    }
    return "";
  });
  return parts.join("");
}

function showDetail(idx) {
  const proj = state.projects[idx];
  if (!proj) return;
  state.current = idx;

  document.getElementById("detail-num").textContent =
    `Project ${String(idx + 1).padStart(2, "0")} / ${String(state.projects.length).padStart(2, "0")}`;
  document.getElementById("detail-title").textContent = proj.title;
  document.getElementById("detail-desc").textContent = proj._desc || "";
  document.getElementById("notebook-stage").innerHTML = proj._html || "Loading…";

  enhanceCellsForEditing(document.getElementById("notebook-stage"), projectSlug(proj));

  // Auto-refresh only in author mode (viewer mode locks current state)
  if (state.authorMode) reloadCurrent(false);

  // Reset to notebook tab
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === "notebook"));
  document.getElementById("tab-notebook").classList.remove("hidden");
  document.getElementById("tab-demo").classList.add("hidden");
  document.getElementById("demo-stage").innerHTML = `<p class="demo-empty">Click <strong>Demo</strong> to load.</p>`;
  state._currentDemoLoaded = false;

  // Update nav
  const prev = idx - 1, next = idx + 1;
  const prevBtn = document.getElementById("detail-prev");
  const nextBtn = document.getElementById("detail-next");
  prevBtn.disabled = prev < 0;
  nextBtn.disabled = next >= state.projects.length;
  document.getElementById("detail-prev-name").textContent =
    prev >= 0 ? state.projects[prev].title : "—";
  document.getElementById("detail-next-name").textContent =
    next < state.projects.length ? state.projects[next].title : "—";

  history.replaceState(null, "", `#${idx + 1}`);
  showView("detail");
}

function bindEvents() {
  document.getElementById("back-btn").addEventListener("click", showHome);

  document.getElementById("run-all-btn").addEventListener("click", runAllCurrent);
  document.getElementById("reload-btn").addEventListener("click", () => reloadCurrent(true));

  // Mode badge: shows VIEWER / AUTHOR; click to flip
  const badge = document.getElementById("mode-badge");
  if (badge) {
    const sync = () => {
      badge.textContent = state.authorMode ? "AUTHOR" : "VIEWER";
      badge.classList.toggle("author", state.authorMode);
      document.body.classList.toggle("viewer-mode", !state.authorMode);
      document.getElementById("reload-btn").style.display = state.authorMode ? "" : "none";
    };
    sync();
    badge.addEventListener("click", () => {
      state.authorMode = !state.authorMode;
      if (state.authorMode) localStorage.setItem("authorMode", "1");
      else localStorage.removeItem("authorMode");
      sync();
      // Re-render current detail to apply
      if (state.current >= 0) showDetail(state.current);
    });
  }

  document.getElementById("detail-prev").addEventListener("click", () => {
    if (state.current > 0) showDetail(state.current - 1);
  });
  document.getElementById("detail-next").addEventListener("click", () => {
    if (state.current < state.projects.length - 1) showDetail(state.current + 1);
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.getElementById("tab-notebook").classList.toggle("hidden", tab !== "notebook");
      document.getElementById("tab-demo").classList.toggle("hidden", tab !== "demo");
      if (tab === "demo" && !state._currentDemoLoaded) {
        state._currentDemoLoaded = true;
        const proj = state.projects[state.current];
        if (proj) loadDemo(proj);
      }
    });
  });

  document.querySelectorAll('[data-action="goto-projects"]').forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelector(".projects").scrollIntoView({ behavior: "smooth" });
    })
  );
  document.querySelectorAll('[data-action="goto-about"]').forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      document.getElementById("about").scrollIntoView({ behavior: "smooth", block: "center" });
    })
  );

  // Reading progress in detail view
  window.addEventListener("scroll", () => {
    if (state.current < 0) return;
    const max = document.body.scrollHeight - window.innerHeight;
    const pct = max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0;
    document.getElementById("detail-progress")?.style.setProperty("--progress", pct + "%");
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (state.current < 0) return; // home view: skip arrow nav
    if (e.key === "ArrowLeft" && state.current > 0) showDetail(state.current - 1);
    if (e.key === "ArrowRight" && state.current < state.projects.length - 1) showDetail(state.current + 1);
    if (e.key === "Escape") showHome();
  });

  window.addEventListener("hashchange", () => {
    const n = parseInt(location.hash.replace("#", ""), 10);
    if (!isNaN(n) && n >= 1 && n <= state.projects.length) {
      showDetail(n - 1);
    } else {
      showHome();
    }
  });
}

async function init() {
  state.projects = await loadProjects();

  if (state.projects.length === 0) {
    document.getElementById("project-list").innerHTML = `
      <li class="project-item"><div class="project-link">
        <div class="project-text">
          <div class="project-title">No projects found</div>
          <p class="project-desc">Run convert.ps1 / convert.sh, then refresh.</p>
        </div>
      </div></li>
    `;
    return;
  }

  await preloadProjectMeta();
  duplicateMarquee();
  renderStats();
  renderProjectList();
  bindEvents();

  // Restore from hash
  const n = parseInt(location.hash.replace("#", ""), 10);
  if (!isNaN(n) && n >= 1 && n <= state.projects.length) {
    showDetail(n - 1);
  }
}

init();
