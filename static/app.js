"use strict";

const $ = (id) => document.getElementById(id);

const TINT = "rgb(230, 40, 80)";
const MAX_FILE_MB = 20;

const state = {
  items: [],
  current: null,
  nextId: 1,
  batchRunning: false,
};

// ---------------------------------------------------------------- options

function options() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const fill = document.querySelector('input[name="fill"]:checked').value;
  return {
    mode,
    fill,
    bg_color: $("bg-solid").checked ? $("bg-color").value : "",
    feather: $("feather").value,
    shift: $("shift").value,
    model: $("model").value,
  };
}

function optsKey(o, edited) {
  // A hand-edited mask doesn't depend on the detection model.
  return JSON.stringify([o.mode, o.fill, o.bg_color, o.feather, o.shift, edited ? "" : o.model]);
}

function bgMode() {
  return options().mode === "remove_background";
}

// Mask value that means "this pixel gets removed" in the current mode.
function removeValue() {
  return bgMode() ? 0 : 255;
}

function syncOptionUI() {
  const o = options();
  const fg = o.mode === "remove_foreground";
  $("fill-section").hidden = !fg;
  $("bg-section").hidden = fg && o.fill === "inpaint";
  $("bg-color").disabled = !$("bg-solid").checked;
  $("feather-out").textContent = o.feather;
  $("shift-out").textContent = o.shift;
}

// ---------------------------------------------------------------- api

async function api(path, fields) {
  const body = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined || v === null) continue;
    if (v instanceof Blob) body.append(k, v, v.name || `${k}.png`);
    else body.append(k, String(v));
  }
  const res = await fetch(path, { method: "POST", body });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.detail) msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return res.blob();
}

function composeFields(o) {
  return { mode: o.mode, fill: o.fill, bg_color: o.bg_color, feather: o.feather, shift: o.shift };
}

async function segmentItem(item, model) {
  item.aiMaskBlob = await api("/api/segment", { image: item.file, model });
  item.aiModel = model;
}

function setResult(item, blob, o, edited) {
  if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
  item.resultBlob = blob;
  item.resultUrl = URL.createObjectURL(blob);
  item.resultMode = o.mode;
  item.optsKey = optsKey(o, edited);
  item.status = "done";
  item.error = null;
}

// ---------------------------------------------------------------- items / queue

function addFiles(fileList) {
  const files = [...fileList].filter((f) => f.type.startsWith("image/"));
  if (!files.length) return toast("Please choose image files.");
  const added = [];
  for (const file of files) {
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      toast(`${file.name} is larger than ${MAX_FILE_MB} MB.`);
      continue;
    }
    const item = {
      id: state.nextId++,
      file,
      name: file.name || `pasted-${Date.now()}.png`,
      url: URL.createObjectURL(file),
      status: "queued",
    };
    state.items.push(item);
    added.push(item);
  }
  renderQueue();
  if (added.length && (!state.current || added.length === 1)) openItem(added[0]);
}

function removeItem(item) {
  state.items = state.items.filter((i) => i !== item);
  URL.revokeObjectURL(item.url);
  if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
  if (state.current === item) {
    state.current = null;
    if (state.items.length) openItem(state.items[0]);
    else showEmpty();
  }
  renderQueue();
}

function isStale(item) {
  return item.status === "done" && item.optsKey !== optsKey(options(), !!item.maskBlob);
}

function needsWork(item) {
  return item.status !== "working" && (item.status !== "done" || isStale(item));
}

function stateLabel(item) {
  if (item.status === "working") return ["working", "Processing…"];
  if (item.status === "error") return ["error", item.error || "Failed"];
  if (item.status === "done") {
    if (isStale(item)) return ["", "Options changed - re-run"];
    return ["done", item.maskBlob ? "Done · mask edited" : "Done"];
  }
  return ["", "Ready"];
}

function renderQueue() {
  const list = $("items");
  list.replaceChildren(
    ...state.items.map((item) => {
      const li = document.createElement("li");
      li.className = "item" + (item === state.current ? " active" : "");
      li.title = item.status === "error" ? `${item.name}: ${item.error}` : item.name;

      const thumb = document.createElement("img");
      thumb.className = "thumb";
      thumb.alt = "";
      thumb.src = item.resultUrl || item.url;

      const info = document.createElement("div");
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = item.name;
      const st = document.createElement("div");
      const [cls, label] = stateLabel(item);
      st.className = "state " + cls;
      st.textContent = label;
      info.append(name, st);

      const actions = document.createElement("div");
      actions.className = "actions";
      if (item.resultBlob) {
        const dl = document.createElement("button");
        dl.textContent = "↓";
        dl.title = "Download result";
        dl.onclick = (e) => { e.stopPropagation(); downloadItem(item); };
        actions.append(dl);
      }
      const rm = document.createElement("button");
      rm.textContent = "✕";
      rm.title = "Remove from list";
      rm.disabled = item.status === "working";
      rm.onclick = (e) => { e.stopPropagation(); removeItem(item); };
      actions.append(rm);

      li.append(thumb, info, actions);
      li.onclick = () => openItem(item);
      return li;
    })
  );

  const pending = state.items.filter(needsWork).length;
  const done = state.items.filter((i) => i.status === "done").length;
  $("count").textContent = state.items.length;
  $("process-all").disabled = state.batchRunning || pending === 0;
  $("process-all").textContent = state.batchRunning ? "Processing…" : pending ? `Process all (${pending})` : "Process all";
  $("process-all").classList.toggle("pulse", !state.batchRunning && pending > 1);
  $("zip").disabled = done === 0;
  $("clear").disabled = state.batchRunning || state.items.length === 0;
}

async function processAll() {
  if (state.batchRunning) return;
  const o = options();
  const todo = state.items.filter(needsWork);
  if (!todo.length) return;
  state.batchRunning = true;
  $("progress").hidden = false;
  let finished = 0;
  const progress = () => { $("progress-bar").style.width = `${(finished / todo.length) * 100}%`; };
  progress();
  renderQueue();

  for (const item of todo) {
    if (!state.items.includes(item)) { finished++; continue; } // removed meanwhile
    item.status = "working";
    renderQueue();
    try {
      const edited = !!item.maskBlob;
      if (!edited && (!item.aiMaskBlob || item.aiModel !== o.model)) await segmentItem(item, o.model);
      const blob = await api("/api/compose", {
        image: item.file,
        mask: item.maskBlob || item.aiMaskBlob,
        ...composeFields(o),
      });
      setResult(item, blob, o, edited);
      if (item === state.current) showResult(item);
    } catch (err) {
      item.status = "error";
      item.error = err.message;
    }
    finished++;
    progress();
    renderQueue();
  }

  state.batchRunning = false;
  setTimeout(() => { if (!state.batchRunning) $("progress").hidden = true; }, 800);
  renderQueue();
  const failed = todo.filter((i) => i.status === "error").length;
  if (failed) toast(`${failed} image(s) failed. Hover over them for details or click to retry in the editor.`);
}

function outputName(item, used) {
  const base = item.name.replace(/\.[^.]+$/, "") || "image";
  const suffix = item.resultMode === "remove_foreground" ? "no-foreground" : "no-background";
  let name = `${base}-${suffix}.png`;
  for (let n = 2; used && used.has(name); n++) name = `${base}-${suffix}-${n}.png`;
  if (used) used.add(name);
  return name;
}

function downloadBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

function downloadItem(item) {
  if (item.resultBlob) downloadBlob(item.resultBlob, outputName(item));
}

async function downloadZip() {
  const done = state.items.filter((i) => i.resultBlob);
  if (!done.length) return;
  const zip = new JSZip();
  const used = new Set();
  for (const item of done) zip.file(outputName(item, used), item.resultBlob);
  const blob = await zip.generateAsync({ type: "blob" });
  downloadBlob(blob, `cutouts-${new Date().toISOString().slice(0, 10)}.zip`);
}

function clearAll() {
  for (const item of state.items) {
    URL.revokeObjectURL(item.url);
    if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
  }
  state.items = [];
  state.current = null;
  showEmpty();
  renderQueue();
}

// ---------------------------------------------------------------- editor

const editor = {
  img: null,          // ImageBitmap of the photo
  w: 0,
  h: 0,
  base: document.createElement("canvas"),  // mask the brush edits start from
  mask: document.createElement("canvas"),  // current mask, white = subject
  strokes: [],
  edited: false,
  drawing: null,
  token: 0,
};
const maskCtx = editor.mask.getContext("2d", { willReadFrequently: true });
const photo = $("photo");
const overlay = $("overlay");
const overlayCtx = overlay.getContext("2d");

function showEmpty() {
  $("editor").hidden = true;
  $("dropzone").hidden = false;
  editor.img = null;
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function setBusy(text) {
  $("busy").hidden = !text;
  if (text) $("busy-text").textContent = text;
  $("apply").disabled = !!text;
  $("redetect").disabled = !!text;
}

function selectTab(name) {
  const mask = name === "mask";
  $("tab-mask").setAttribute("aria-selected", mask);
  $("tab-result").setAttribute("aria-selected", !mask);
  $("mask-view").hidden = !mask;
  $("result-view").hidden = mask;
  if (mask) layoutCanvas();
}

function markDirty(dirty = true) {
  $("apply").classList.toggle("pulse", dirty);
}

async function openItem(item) {
  const token = ++editor.token;
  state.current = item;
  renderQueue();
  $("dropzone").hidden = true;
  $("editor").hidden = false;

  const img = await createImageBitmap(item.file).catch(() => null);
  if (token !== editor.token) return;
  if (!img) {
    item.status = "error";
    item.error = "Your browser can't read this image";
    renderQueue();
    return toast(item.error);
  }
  editor.img = img;
  editor.w = img.width;
  editor.h = img.height;
  for (const c of [photo, overlay, editor.mask, editor.base]) {
    c.width = img.width;
    c.height = img.height;
  }
  photo.getContext("2d").drawImage(img, 0, 0);
  overlayCtx.clearRect(0, 0, editor.w, editor.h);
  $("before-img").src = item.url;
  setStatus(`${item.name} · ${img.width}×${img.height}`);
  layoutCanvas();

  if (item.resultUrl) showResult(item);

  try {
    if (!item.aiMaskBlob) {
      selectTab("mask");
      setBusy("Detecting subject…");
      await segmentItem(item, options().model);
      if (token !== editor.token) return;
    }
    await loadBaseMask(item.maskBlob || item.aiMaskBlob, !!item.maskBlob);
    if (token !== editor.token) return;
    setBusy(null);
    if (!item.resultUrl) await apply();
    else {
      markDirty(isStale(item));
      selectTab(isStale(item) ? "mask" : "result");
    }
  } catch (err) {
    if (token !== editor.token) return;
    setBusy(null);
    setStatus(err.message, true);
    toast(err.message);
  }
}

async function loadBaseMask(blob, edited) {
  const bmp = await createImageBitmap(blob);
  const ctx = editor.base.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, editor.w, editor.h);
  ctx.drawImage(bmp, 0, 0, editor.w, editor.h);
  editor.strokes = [];
  editor.edited = edited;
  rebuildMask();
}

function rebuildMask() {
  maskCtx.globalCompositeOperation = "copy";
  maskCtx.drawImage(editor.base, 0, 0);
  maskCtx.globalCompositeOperation = "source-over";
  for (const s of editor.strokes) replayStroke(s);
  rebuildOverlay();
}

function rebuildOverlay() {
  if (!editor.w) return;
  const src = maskCtx.getImageData(0, 0, editor.w, editor.h).data;
  const out = overlayCtx.createImageData(editor.w, editor.h);
  const d = out.data;
  const bg = bgMode();
  for (let i = 0; i < src.length; i += 4) {
    d[i] = 230;
    d[i + 1] = 40;
    d[i + 2] = 80;
    d[i + 3] = bg ? 255 - src[i] : src[i];
  }
  overlayCtx.putImageData(out, 0, 0);
}

function layoutCanvas() {
  if (!editor.img || $("mask-view").hidden) return;
  const stage = $("stage").getBoundingClientRect();
  const scale = Math.min((stage.width - 16) / editor.w, (stage.height - 16) / editor.h, 3);
  const wrap = $("canvas-wrap");
  wrap.style.width = `${Math.max(1, Math.floor(editor.w * scale))}px`;
  wrap.style.height = `${Math.max(1, Math.floor(editor.h * scale))}px`;
}

// --- brush

function brushValue() {
  const keep = document.querySelector('input[name="brush"]:checked').value === "keep";
  return keep === bgMode() ? 255 : 0;
}

function screenToImageScale() {
  return editor.w / $("canvas-wrap").getBoundingClientRect().width;
}

function toImage(e) {
  const r = $("canvas-wrap").getBoundingClientRect();
  return [((e.clientX - r.left) * editor.w) / r.width, ((e.clientY - r.top) * editor.h) / r.height];
}

function strokeOn(ctx, color, size, a, b, erase = false) {
  ctx.save();
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = size;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  if (a[0] === b[0] && a[1] === b[1]) {
    ctx.arc(a[0], a[1], size / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSegment(stroke, a, b) {
  strokeOn(maskCtx, stroke.value ? "#fff" : "#000", stroke.size, a, b);
  strokeOn(overlayCtx, TINT, stroke.size, a, b, stroke.value !== removeValue());
}

function replayStroke(s) {
  const pts = s.points;
  strokeOn(maskCtx, s.value ? "#fff" : "#000", s.size, pts[0], pts[0]);
  for (let i = 1; i < pts.length; i++) strokeOn(maskCtx, s.value ? "#fff" : "#000", s.size, pts[i - 1], pts[i]);
}

function updateCursor(e) {
  const cur = $("cursor");
  if (!e) return (cur.hidden = true);
  const r = $("canvas-wrap").getBoundingClientRect();
  const size = Number($("brush-size").value);
  cur.hidden = false;
  cur.style.width = cur.style.height = `${size}px`;
  cur.style.left = `${e.clientX - r.left}px`;
  cur.style.top = `${e.clientY - r.top}px`;
  cur.classList.toggle("remove", brushValue() === removeValue());
}

function onPointerDown(e) {
  if (!editor.img || !$("busy").hidden) return;
  if (e.pointerType === "mouse" && e.button !== 0) return;
  e.preventDefault();
  $("canvas-wrap").setPointerCapture(e.pointerId);
  const p = toImage(e);
  const stroke = { value: brushValue(), size: Number($("brush-size").value) * screenToImageScale(), points: [p] };
  editor.strokes.push(stroke);
  editor.drawing = stroke;
  editor.edited = true;
  drawSegment(stroke, p, p);
  markDirty();
}

function onPointerMove(e) {
  updateCursor(e);
  const stroke = editor.drawing;
  if (!stroke) return;
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of events.length ? events : [e]) {
    const p = toImage(ev);
    const last = stroke.points[stroke.points.length - 1];
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 0.5) continue;
    stroke.points.push(p);
    drawSegment(stroke, last, p);
  }
}

function onPointerUp() {
  editor.drawing = null;
}

function undo() {
  if (!editor.strokes.length) return;
  editor.strokes.pop();
  rebuildMask();
  markDirty();
}

function resetMask() {
  const item = state.current;
  if (!item || !item.aiMaskBlob) return;
  loadBaseMask(item.aiMaskBlob, false).then(() => markDirty());
}

async function redetect() {
  const item = state.current;
  if (!item) return;
  const token = editor.token;
  try {
    setBusy("Detecting subject…");
    await segmentItem(item, options().model);
    if (token !== editor.token) return;
    await loadBaseMask(item.aiMaskBlob, false);
    markDirty();
  } catch (err) {
    toast(err.message);
  } finally {
    if (token === editor.token) setBusy(null);
  }
}

function maskBlob() {
  return new Promise((resolve) => editor.mask.toBlob(resolve, "image/png"));
}

async function apply() {
  const item = state.current;
  if (!item || !editor.img) return;
  const token = editor.token;
  const o = options();
  const fg = o.mode === "remove_foreground";
  setBusy(fg && o.fill === "inpaint" ? "Filling the removed area…" : "Cutting out…");
  item.status = "working";
  renderQueue();
  try {
    const mask = await maskBlob();
    const blob = await api("/api/compose", { image: item.file, mask, ...composeFields(o) });
    item.maskBlob = editor.edited ? mask : null;
    setResult(item, blob, o, editor.edited);
    if (token === editor.token) {
      markDirty(false);
      showResult(item);
      selectTab("result");
    }
  } catch (err) {
    item.status = "error";
    item.error = err.message;
    toast(err.message);
  } finally {
    if (token === editor.token) setBusy(null);
    renderQueue();
  }
}

// --- result view

function showResult(item) {
  $("result-img").src = item.resultUrl;
  $("before-img").src = item.url;
  $("download").disabled = !item.resultUrl;
}

function onOptionsChanged() {
  syncOptionUI();
  rebuildOverlay();
  renderQueue();
  if (state.current && state.current.resultUrl) markDirty(isStale(state.current));
}

// ---------------------------------------------------------------- misc

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 5000);
}

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const sel = $("model");
    sel.replaceChildren(
      ...data.models.map((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        opt.selected = m.id === data.default;
        return opt;
      })
    );
  } catch (err) {
    toast("Could not reach the server: " + err.message);
  }
}

function wire() {
  // file input, drag & drop, paste
  $("browse").onclick = (e) => { e.stopPropagation(); $("file-input").click(); };
  $("dropzone").onclick = () => $("file-input").click();
  $("add-more").onclick = () => $("file-input").click();
  $("file-input").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };

  let dragDepth = 0;
  document.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragover"); });
  document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragover"); } });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove("dragover");
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  document.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) addFiles(files);
  });

  // options
  for (const el of document.querySelectorAll('input[name="mode"], input[name="fill"], #bg-solid, #bg-color, #feather, #shift')) {
    el.addEventListener("input", onOptionsChanged);
  }
  $("model").addEventListener("change", () => {
    renderQueue();
    if (state.current && !editor.edited) redetect();
  });

  // editor
  const wrap = $("canvas-wrap");
  wrap.addEventListener("pointerdown", onPointerDown);
  wrap.addEventListener("pointermove", onPointerMove);
  wrap.addEventListener("pointerup", onPointerUp);
  wrap.addEventListener("pointercancel", onPointerUp);
  wrap.addEventListener("pointerleave", () => updateCursor(null));
  $("undo").onclick = undo;
  $("reset").onclick = resetMask;
  $("redetect").onclick = redetect;
  $("apply").onclick = apply;
  $("show-overlay").onchange = (e) => wrap.classList.toggle("no-overlay", !e.target.checked);
  $("tab-mask").onclick = () => selectTab("mask");
  $("tab-result").onclick = () => selectTab("result");
  new ResizeObserver(layoutCanvas).observe($("stage"));

  // result
  $("compare-range").addEventListener("input", (e) => $("compare").style.setProperty("--pos", `${e.target.value}%`));
  $("download").onclick = () => state.current && downloadItem(state.current);

  // queue
  $("process-all").onclick = processAll;
  $("zip").onclick = downloadZip;
  $("clear").onclick = clearAll;

  // keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("select, input[type=text], input[type=color]")) return;
    const size = $("brush-size");
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
    else if (e.key === "[") size.value = Number(size.value) - 5;
    else if (e.key === "]") size.value = Number(size.value) + 5;
    else if (e.key.toLowerCase() === "k") document.querySelector('input[name="brush"][value="keep"]').checked = true;
    else if (e.key.toLowerCase() === "r" && !e.ctrlKey && !e.metaKey) document.querySelector('input[name="brush"][value="remove"]').checked = true;
  });
}

wire();
syncOptionUI();
loadModels();
renderQueue();
