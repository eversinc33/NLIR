"use strict";

const state = {
  prompt: "",
  result: null,
  rules: [],
  selectedRuleId: null,
};

const ruleDetailCache = new Map();
let cy = null;

const HUE_BANDS = {
  entity: [205, 255],
  operation: [268, 320],
  relationship: [15, 45],
};

function hashCode(text) {
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function colorForType(kind, type) {
  const band = HUE_BANDS[kind] || [210, 210];
  const span = Math.max(1, band[1] - band[0]);
  const hue = band[0] + (hashCode(`${kind}:${type}`) % span);
  // Cytoscape's color parser only accepts the comma-separated hsl() form.
  return {
    line: `hsl(${hue}, 50%, 48%)`, // edges and legend swatches — needs contrast on white
    fill: `hsl(${hue}, 55%, 95%)`, // node background — a light tint, not a solid fill
    ring: `hsl(${hue}, 38%, 66%)`, // node border — soft, not heavy
    text: `hsl(${hue}, 42%, 32%)`, // node label — muted, same hue family as the fill
  };
}

function nodeWidth(ele) {
  const label = ele.data("label") || "";
  return Math.max(42, Math.min(150, label.length * 6.4 + 30));
}

const GRAPH_STYLE = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "font-family": "Inter, ui-sans-serif, system-ui, sans-serif",
      "font-weight": 500,
      "font-size": 10,
      "text-valign": "center",
      "text-halign": "center",
      "text-wrap": "wrap",
      "text-max-width": "88px",
      width: nodeWidth,
      height: 30,
      "border-width": 1.5,
    },
  },
  {
    selector: 'node[kind = "entity"]',
    style: {
      shape: "ellipse",
      "background-color": (ele) => colorForType("entity", ele.data("type")).fill,
      "border-color": (ele) => colorForType("entity", ele.data("type")).ring,
      color: (ele) => colorForType("entity", ele.data("type")).text,
    },
  },
  {
    selector: 'node[kind = "operation"]',
    style: {
      shape: "round-rectangle",
      "corner-radius": "auto",
      height: 26,
      "background-color": (ele) => colorForType("operation", ele.data("type")).fill,
      "border-color": (ele) => colorForType("operation", ele.data("type")).ring,
      color: (ele) => colorForType("operation", ele.data("type")).text,
    },
  },
  {
    selector: "node.hovered",
    style: { "border-width": 2.5, "border-color": "#7c6cf0" },
  },
  {
    selector: "node.node-selected",
    style: { "border-width": 3, "border-color": "#7c6cf0", "border-style": "double" },
  },
  {
    selector: "node.matched",
    style: {
      "border-width": 3,
      "border-color": "#c0392b",
      "background-color": "#fdecea",
      color: "#96322a",
    },
  },
  {
    selector: 'edge[kind = "relationship"]',
    style: {
      width: 2,
      "line-color": (ele) => colorForType("relationship", ele.data("type")).line,
      "target-arrow-color": (ele) => colorForType("relationship", ele.data("type")).line,
      "target-arrow-shape": "triangle",
      "arrow-scale": 1,
      "curve-style": "bezier",
      label: "data(label)",
      "font-size": 8,
      "font-weight": 500,
      color: "#5a5a63",
      "text-background-color": "#ffffff",
      "text-background-opacity": 0.9,
      "text-background-shape": "roundrectangle",
      "text-background-padding": "2px",
      "text-rotation": "autorotate",
    },
  },
  {
    selector: 'edge[kind = "role"]',
    style: {
      width: 1.2,
      "line-color": "#c7c7ce",
      "target-arrow-color": "#c7c7ce",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.9,
      "curve-style": "bezier",
      "line-style": "dashed",
      label: "data(label)",
      "font-size": 7,
      color: "#a3a3ab",
      "text-background-color": "#ffffff",
      "text-background-opacity": 0.85,
      "text-background-shape": "roundrectangle",
      "text-background-padding": "2px",
      "text-rotation": "autorotate",
    },
  },
  {
    selector: "edge.matched",
    style: { "line-color": "#c0392b", "target-arrow-color": "#c0392b", width: 3.2 },
  },
];

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function setStatus(message) {
  document.getElementById("lift-status").textContent = message;
}

function initGraph() {
  cy = cytoscape({
    container: document.getElementById("graph"),
    elements: [],
    style: GRAPH_STYLE,
    layout: { name: "grid" },
    wheelSensitivity: 0.25,
  });
  const container = document.getElementById("graph");
  cy.on("mouseover", "node", (event) => {
    event.target.addClass("hovered");
    container.style.cursor = "pointer";
  });
  cy.on("mouseout", "node", (event) => {
    event.target.removeClass("hovered");
    container.style.cursor = "default";
  });
  cy.on("tap", "node", (event) => selectGraphNode(event.target.id()));
  cy.on("tap", (event) => {
    if (event.target === cy) selectGraphNode(null);
  });
}

function selectGraphNode(nodeId) {
  document.querySelectorAll(".token-mark.node-selected").forEach((el) => el.classList.remove("node-selected"));
  cy.elements(".node-selected").removeClass("node-selected");
  if (!nodeId) return;
  cy.getElementById(nodeId).addClass("node-selected");
  let firstMark = null;
  document.querySelectorAll(".token-mark").forEach((el) => {
    const ids = (el.dataset.nodeIds || "").split(",").filter(Boolean);
    if (ids.includes(nodeId)) {
      el.classList.add("node-selected");
      if (!firstMark) firstMark = el;
    }
  });
  if (firstMark) firstMark.scrollIntoView({ block: "center", behavior: "smooth" });
}

function renderGraph(nodes, edges) {
  cy.elements().remove();
  cy.add([...nodes, ...edges]);
  cy.layout({
    name: "cose",
    animate: true,
    animationDuration: 350,
    padding: 40,
    nodeRepulsion: 9000,
    idealEdgeLength: 90,
    edgeElasticity: 120,
    gravity: 45,
    numIter: 1500,
  }).run();
  const empty = document.getElementById("graph-empty");
  const legend = document.getElementById("graph-legend");
  if (nodes.length) {
    empty.classList.add("hidden");
    legend.classList.remove("hidden");
  } else {
    empty.classList.remove("hidden");
    legend.classList.add("hidden");
  }
}

function renderLegend() {
  const legend = document.getElementById("graph-legend");
  const items = [
    ["Entity", colorForType("entity", "x").line],
    ["Operation", colorForType("operation", "x").line],
    ["Relationship", colorForType("relationship", "x").line],
  ];
  legend.replaceChildren();
  for (const [label, color] of items) {
    const row = document.createElement("div");
    row.className = "flex items-center gap-2";
    const swatch = document.createElement("span");
    swatch.style.background = color;
    swatch.className = "inline-block w-2.5 h-2.5 rounded-full shrink-0";
    row.append(swatch, document.createTextNode(label));
    legend.append(row);
  }
}

function renderLiftMap(text, tokens) {
  const points = new Set([0, text.length]);
  for (const token of tokens) {
    points.add(token.start);
    points.add(token.end);
  }
  const bounds = [...points].filter((point) => point >= 0 && point <= text.length).sort((a, b) => a - b);
  const container = document.getElementById("lift-map");
  container.replaceChildren();
  for (let index = 0; index < bounds.length - 1; index += 1) {
    const start = bounds[index];
    const end = bounds[index + 1];
    if (start === end) continue;
    const segment = text.slice(start, end);
    const covering = tokens.filter((token) => token.start < end && token.end > start);
    appendSegment(container, segment, covering);
  }
}

function appendSegment(container, text, covering) {
  if (!covering.length) {
    container.append(document.createTextNode(text));
    return;
  }
  const mark = document.createElement("mark");
  mark.className = "token-mark";
  const color = colorForType(covering[0].kind, covering[0].type);
  mark.style.setProperty("--token-bg", color.fill);
  mark.style.setProperty("--token-border", color.ring);
  mark.title = covering.map((token) => `${token.kind}: ${token.type}`).join("; ");
  mark.dataset.nodeIds = covering.map((token) => token.node_id).join(",");
  mark.append(document.createTextNode(text));
  container.append(mark);
}

function applyMatchHighlight(ruleId) {
  document.querySelectorAll(".token-mark.matched").forEach((el) => el.classList.remove("matched"));
  if (cy) cy.elements().removeClass("matched");
  if (!ruleId || !state.result) return;
  const rule = state.result.rules.find((item) => item.id === ruleId);
  if (!rule || !rule.matches.length) return;
  const matchedIds = new Set(rule.matches.flatMap((match) => match.matched_node_ids));
  document.querySelectorAll(".token-mark").forEach((el) => {
    const ids = (el.dataset.nodeIds || "").split(",").filter(Boolean);
    if (ids.some((id) => matchedIds.has(id))) el.classList.add("matched");
  });
  cy.elements().forEach((ele) => {
    if (matchedIds.has(ele.id())) ele.addClass("matched");
  });
  const matchedEles = cy.elements(".matched");
  if (matchedEles.length) {
    cy.animate({ fit: { eles: matchedEles, padding: 60 } }, { duration: 200 });
  }
}

function renderRuleList() {
  const box = document.getElementById("rule-list");
  box.replaceChildren();
  const results = new Map((state.result ? state.result.rules : []).map((item) => [item.id, item]));
  for (const rule of state.rules) {
    const row = document.createElement("button");
    row.className = "rule-row" + (state.selectedRuleId === rule.id ? " selected" : "");
    const status = results.get(rule.id) ? results.get(rule.id).status : null;
    const badge = document.createElement("span");
    badge.className = "rule-badge" + (status === "HIT" ? " hit" : "");
    badge.textContent = status || "—";
    row.append(badge, document.createTextNode(rule.id));
    if (rule.description) {
      const small = document.createElement("div");
      small.className = "text-obs-muted text-[11px] mt-1 font-normal";
      small.textContent = rule.description;
      row.append(small);
    }
    row.onclick = () => selectRule(rule.id);
    box.append(row);
  }
}

async function selectRule(id) {
  state.selectedRuleId = id;
  renderRuleList();
  const wrap = document.getElementById("rule-detail-wrap");
  const body = document.getElementById("rule-detail-body");
  wrap.classList.remove("hidden");
  body.classList.remove("hidden");
  document.getElementById("rule-detail-chevron").textContent = "▲";
  document.getElementById("rule-detail-title").textContent = id;
  try {
    let detail = ruleDetailCache.get(id);
    if (!detail) {
      detail = await api("/api/rules/" + encodeURIComponent(id));
      ruleDetailCache.set(id, detail);
    }
    document.getElementById("rule-detail-meta").textContent = [
      detail.description,
      detail.author ? "Author: " + detail.author : null,
    ]
      .filter(Boolean)
      .join(" — ");
    const code = document.getElementById("rule-detail-code");
    code.textContent = detail.text;
    code.removeAttribute("data-highlighted");
    hljs.highlightElement(code);
  } catch (error) {
    setStatus(error.message);
  }
  applyMatchHighlight(id);
}

function showLiftMap() {
  const textarea = document.getElementById("prompt-input");
  textarea.classList.remove("flex-1", "min-h-[140px]");
  textarea.classList.add("h-24", "shrink-0");
  document.getElementById("lift-map-wrap").classList.remove("hidden");
  document.getElementById("edit-button").classList.remove("hidden");
}

function hideLiftMap() {
  document.getElementById("lift-map-wrap").classList.add("hidden");
  const textarea = document.getElementById("prompt-input");
  textarea.classList.add("flex-1", "min-h-[140px]");
  textarea.classList.remove("h-24", "shrink-0");
  textarea.focus();
}

function analysisStatus(result) {
  const rejected = result.attempts.filter((attempt) => attempt.state !== "accepted");
  if (!rejected.length) return "Lift complete.";
  const codes = [...new Set(rejected.flatMap((attempt) => attempt.diagnostics))];
  const anyAccepted = result.attempts.some((attempt) => attempt.state === "accepted");
  const detail = codes.length ? codes.join(", ") : "no accepted IR";
  return anyAccepted ? `Lift completed with diagnostics: ${detail}.` : `Lift did not complete: ${detail}.`;
}

async function runLift() {
  const text = document.getElementById("prompt-input").value;
  if (!text.trim()) {
    setStatus("Enter a prompt first.");
    return;
  }
  state.prompt = text;
  state.selectedRuleId = null;
  setStatus("Lifting through the live model…");
  document.getElementById("lift-button").disabled = true;
  try {
    const result = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text }),
    });
    state.result = result;
    renderLiftMap(text, result.tokens);
    renderGraph(result.graph.nodes, result.graph.edges);
    renderRuleList();
    document.getElementById("rule-detail-wrap").classList.add("hidden");
    showLiftMap();
    setStatus(analysisStatus(result));
  } catch (error) {
    setStatus(error.message);
  } finally {
    document.getElementById("lift-button").disabled = false;
  }
}

async function loadRules() {
  try {
    const data = await api("/api/rules");
    state.rules = data.rules;
    renderRuleList();
  } catch (error) {
    setStatus(error.message);
  }
}

document.getElementById("lift-button").onclick = runLift;
document.getElementById("edit-button").onclick = hideLiftMap;
document.getElementById("left-toggle").onclick = () =>
  document.getElementById("left-panel").classList.toggle("collapsed");
document.getElementById("right-toggle").onclick = () =>
  document.getElementById("right-panel").classList.toggle("collapsed");
document.getElementById("rule-detail-toggle").onclick = () => {
  const body = document.getElementById("rule-detail-body");
  const collapsed = body.classList.toggle("hidden");
  document.getElementById("rule-detail-chevron").textContent = collapsed ? "▼" : "▲";
};

initGraph();
renderLegend();
loadRules();
