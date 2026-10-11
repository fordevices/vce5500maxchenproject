#!/usr/bin/env python3
"""
ESAM SE Cockpit — Backend API  v2.0
Adds: ingestion_run_log table, /api/ingestion/runs endpoint,
      Ingestion History tab in the UI.

Changes from v1:
  - GET /api/ingestion/runs  → returns last 50 ingestion runs
  - GET /api/ingestion/runs/{run_id}  → returns one run in full detail
  - HTML UI gains a fourth tab: Ingestion History
"""

import os, json, subprocess, hashlib, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('/opt/cockpit/.env')

import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

app = FastAPI(title="ESAM SE Cockpit", version="2.0.0")

DB_URL       = os.environ['DATABASE_URL']
BASELINE_PATH = os.environ['YAML_BASELINE_PATH']
BASELINE_REPO = os.environ['YAML_BASELINE_REPO']

gpt_client = OpenAI(
    api_key=os.environ['GPT51_API_KEY'],
    base_url=os.environ['GPT51_API_BASE']
)
GPT_MODEL = os.environ['GPT51_MODEL']

def get_db():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)


# ── SCHEMA MIGRATION (run once on startup) ────────────────────────────────────

def ensure_ingestion_log_table():
    """Idempotent: creates ingestion_run_log if it doesn't exist yet."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_run_log (
            id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            run_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            mode             TEXT NOT NULL,
            triggered_by     TEXT NOT NULL DEFAULT 'cli',
            source_files     JSONB NOT NULL DEFAULT '[]',
            vertices_added   INTEGER NOT NULL DEFAULT 0,
            edges_added      INTEGER NOT NULL DEFAULT 0,
            chunks_added     INTEGER NOT NULL DEFAULT 0,
            duration_sec     NUMERIC(8,2),
            status           TEXT NOT NULL DEFAULT 'success',
            error_detail     TEXT,
            graph_json_hash  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_irl_run_at
            ON ingestion_run_log (run_at DESC);
    """)
    conn.commit()
    cur.close()
    conn.close()

ensure_ingestion_log_table()


# ── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        conn = get_db(); conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GRAPH STATS ───────────────────────────────────────────────────────────────

@app.get("/api/graph/stats")
def graph_stats():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT label, count(*) as n FROM graph_vertices GROUP BY label ORDER BY label")
    vertices = cur.fetchall()
    cur.execute("SELECT type, count(*) as n FROM graph_edges GROUP BY type ORDER BY type")
    edges = cur.fetchall()
    cur.execute("SELECT COUNT(*) AS n FROM document_chunks")
    chunks = cur.fetchone()["n"]
    conn.close()
    return {"vertices": vertices, "edges": edges, "chunks": chunks}


# ── FULL GRAPH TREE ───────────────────────────────────────────────────────────

@app.get("/api/graph/tree")
def graph_tree():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        WITH RECURSIVE wbs_tree AS (
            SELECT v.id, v.label, v.properties,
                   NULL::uuid AS parent_id, 0 AS depth
            FROM graph_vertices v
            WHERE v.label = 'Vehicle'

            UNION ALL

            SELECT child.id, child.label, child.properties,
                   e.from_id AS parent_id, wbs_tree.depth + 1
            FROM graph_vertices child
            JOIN graph_edges e ON e.from_id = child.id AND e.type = 'PART_OF'
            JOIN wbs_tree ON wbs_tree.id = e.to_id
            WHERE wbs_tree.depth < 6
        )
        SELECT * FROM wbs_tree ORDER BY depth, properties->>'wbs_code'
    """)
    rows = cur.fetchall()
    conn.close()
    return {"tree": rows}


# ── COVERAGE DASHBOARD ────────────────────────────────────────────────────────

@app.get("/api/coverage")
def coverage_dashboard():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT
            sys.properties->>'name'     AS system_name,
            sys.properties->>'wbs_code' AS wbs_code,
            COUNT(DISTINCT srs.id)      AS total_srs,
            COUNT(DISTINCT CASE WHEN t.properties->>'status' = 'PASSED'
                                THEN srs.id END) AS green,
            COUNT(DISTINCT CASE WHEN t.properties->>'status' = 'FAILED'
                                THEN srs.id END) AS red,
            COUNT(DISTINCT CASE
                WHEN NOT EXISTS (
                    SELECT 1 FROM graph_edges ve
                    WHERE ve.from_id = srs.id AND ve.type = 'VERIFIES'
                ) THEN srs.id END) AS orange
        FROM graph_vertices sys
        JOIN graph_edges pe  ON pe.to_id  = sys.id  AND pe.type = 'PART_OF'
        JOIN graph_vertices sub ON pe.from_id = sub.id AND sub.label = 'Subsystem'
        JOIN graph_edges se  ON se.to_id  = sub.id
        JOIN graph_vertices srs ON se.from_id = srs.id AND srs.label = 'SRS'
        LEFT JOIN graph_edges ve ON ve.from_id = srs.id AND ve.type = 'VERIFIES'
        LEFT JOIN graph_vertices t ON ve.to_id = t.id  AND t.label  = 'TestCase'
        WHERE sys.label = 'System'
        GROUP BY sys.id, sys.properties->>'name', sys.properties->>'wbs_code'
        ORDER BY sys.properties->>'wbs_code'
    """)
    rows = cur.fetchall()
    conn.close()
    return {"coverage": rows}


# ── INGESTION RUN LOG ─────────────────────────────────────────────────────────

@app.get("/api/ingestion/runs")
def ingestion_runs(limit: int = 50):
    """Return the most recent ingestion runs, newest first."""
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT id, run_at, mode, triggered_by,
               source_files, vertices_added, edges_added, chunks_added,
               duration_sec, status, graph_json_hash
        FROM ingestion_run_log
        ORDER BY run_at DESC
        LIMIT %s
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    # Convert UUIDs and datetimes to strings for JSON
    for r in rows:
        r["id"]     = str(r["id"])
        r["run_at"] = r["run_at"].isoformat() if r["run_at"] else None
    conn.close()
    return {"runs": rows, "count": len(rows)}


@app.get("/api/ingestion/runs/{run_id}")
def ingestion_run_detail(run_id: str):
    """Return one run including error_detail."""
    conn = get_db(); cur = conn.cursor()
    cur.execute(
        "SELECT * FROM ingestion_run_log WHERE id = %s::uuid",
        (run_id,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    r = dict(row)
    r["id"]     = str(r["id"])
    r["run_at"] = r["run_at"].isoformat() if r["run_at"] else None
    return r


# ── YAML DIFF MATRIX ──────────────────────────────────────────────────────────

@app.post("/api/baseline/diff")
async def baseline_diff(file: UploadFile = File(...)):
    content = await file.read()
    try:
        new_data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT properties->>'id' as eid, label FROM graph_vertices")
    existing = {row['eid']: row['label'] for row in cur.fetchall()}
    conn.close()

    new_entities = {}
    vehicle = new_data.get('vehicle', {})
    new_entities[vehicle.get('id')] = 'Vehicle'
    for family in vehicle.get('families', []):
        new_entities[family['id']] = 'VehicleFamily'
        for variant in family.get('variants', []):
            new_entities[variant['id']] = 'Variant'
            for system in variant.get('systems', []):
                new_entities[system['id']] = 'System'
                for sub in system.get('subsystems', []):
                    new_entities[sub['id']] = 'Subsystem'

    added   = {k: v for k, v in new_entities.items() if k not in existing}
    removed = {k: v for k, v in existing.items()     if k not in new_entities}
    retained = {k: v for k, v in new_entities.items() if k in existing}

    return {
        "diff": {
            "added":            added,
            "removed":          removed,
            "retained_count":   len(retained),
            "interface_count":  len(new_data.get('interfaces', [])),
            "artifact_count":   len(new_data.get('artifacts', []))
        },
        "summary": (
            f"{len(added)} nodes added, {len(removed)} nodes removed, "
            f"{len(retained)} unchanged"
        )
    }


# ── APPROVE AND LOCK BASELINE ─────────────────────────────────────────────────

@app.post("/api/baseline/approve")
async def approve_baseline(
    approved_by: str = Form(...),
    file: UploadFile = File(...)
):
    content = await file.read()
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    baseline_file = Path(BASELINE_PATH) / 'vehicle_decomposition.yaml'
    if not Path(BASELINE_PATH).exists():
        subprocess.run(['git', 'clone', BASELINE_REPO, BASELINE_PATH], check=True)

    baseline_file.write_bytes(content)
    subprocess.run(['git', '-C', BASELINE_PATH, 'add', '.'], check=True)
    subprocess.run(['git', '-C', BASELINE_PATH, 'commit',
                    '-m', f'SE approved baseline — {approved_by}'], check=True)
    subprocess.run(['git', '-C', BASELINE_PATH, 'push'], check=True)

    git_hash = subprocess.check_output(
        ['git', '-C', BASELINE_PATH, 'rev-parse', 'HEAD']
    ).decode().strip()

    t0 = time.time()
    result = subprocess.run(
        ['python3', '/opt/esam/ingest.py'],
        capture_output=True, text=True
    )
    duration = round(time.time() - t0, 2)

    if result.returncode != 0:
        raise HTTPException(status_code=500,
                            detail=f"Ingestion failed: {result.stderr}")

    return {
        "status":           "approved",
        "git_hash":         git_hash,
        "approved_by":      approved_by,
        "duration_sec":     duration,
        "ingestion_output": result.stdout
    }


# ── NATURAL LANGUAGE QUERY ────────────────────────────────────────────────────

@app.post("/api/query")
async def natural_language_query(body: dict):
    question = body.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    classification_prompt = f"""
You are a systems engineering assistant with access to a vehicle engineering database.

The database has these tables:
- graph_vertices(id UUID, label TEXT, properties JSONB)
  Labels: Vehicle, VehicleFamily, Variant, System, Subsystem, SRS, TestCase, DesignArtifact
- graph_edges(id UUID, from_id UUID, to_id UUID, type TEXT, properties JSONB)
  Edge types: PART_OF, HAS_ARTIFACT, INTERFACES_WITH, DERIVED_FROM, VERIFIES
- document_chunks(id UUID, artifact_id UUID, content TEXT, embedding vector(768))

Classify this question as either:
- STRUCTURAL: asks about relationships, interfaces, WBS decomposition, dependencies, coverage
- CONTEXTUAL: asks about content within documents (specifications, limits, descriptions, values)

Then write a PostgreSQL query to answer it.

Question: {question}

Respond in JSON only:
{{
  "type": "STRUCTURAL" or "CONTEXTUAL",
  "sql": "your SQL query here",
  "explanation": "one sentence explaining what this query does"
}}
"""

    response = gpt_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0,
        max_tokens=800
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except Exception:
        return {"answer": response.choices[0].message.content, "type": "direct"}

    query_type  = result.get("type", "STRUCTURAL")
    sql         = result.get("sql", "")
    explanation = result.get("explanation", "")

    conn = get_db(); cur = conn.cursor()
    try:
        if query_type == "STRUCTURAL":
            cur.execute(sql)
            raw_results = [dict(r) for r in cur.fetchall()]
        else:
            raw_results = [{"note": "Contextual search — run semantic query via CLI: python3 esam_ingest_standalone.py --mode query"}]
    except Exception as e:
        conn.close()
        return {"error": str(e), "sql": sql, "type": query_type}
    conn.close()

    format_response = gpt_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": f"""
The user asked: {question}
Database results:
{json.dumps(raw_results, indent=2, default=str)}
Provide a clear, concise answer in plain English for a systems engineer.
"""}],
        temperature=0.3,
        max_tokens=500
    )

    return {
        "question":         question,
        "type":             query_type,
        "explanation":      explanation,
        "answer":           format_response.choices[0].message.content,
        "raw_result_count": len(raw_results)
    }


# ── HTML UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def cockpit_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESAM SE Cockpit</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; }
  header { background: #1a1f2e; padding: 16px 32px; border-bottom: 2px solid #2e75b6;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 20px; color: #ffffff; }
  header span { font-size: 12px; color: #7a8ba0; }
  .tabs { display: flex; gap: 2px; padding: 16px 32px 0; background: #13161f; }
  .tab { padding: 10px 24px; cursor: pointer; border-radius: 6px 6px 0 0;
         font-size: 13px; background: #1a1f2e; color: #8a9ab0; border: 1px solid #2a3040; }
  .tab.active { background: #2e75b6; color: white; border-color: #2e75b6; }
  .panel { display: none; padding: 24px 32px 80px; }
  .panel.active { display: block; }
  .card { background: #1a1f2e; border: 1px solid #2a3040; border-radius: 8px;
          padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 14px; color: #7a9fc0; margin-bottom: 12px; text-transform: uppercase;
             letter-spacing: 0.5px; }
  input[type=file], input[type=text] { background: #0f1117; border: 1px solid #2a3040;
    color: #e0e0e0; padding: 8px 12px; border-radius: 4px; width: 100%; margin-bottom: 8px; }
  button { background: #2e75b6; color: white; border: none; padding: 10px 20px;
           border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:hover { background: #3a8fd6; }
  button.success { background: #27ae60; }
  pre { background: #0f1117; padding: 12px; border-radius: 4px; font-size: 12px;
        overflow-x: auto; max-height: 400px; overflow-y: auto; border: 1px solid #2a3040; }
  .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
  .stat { background: #0f1117; border-radius: 6px; padding: 12px 20px; flex: 1;
          min-width: 120px; text-align: center; border: 1px solid #2a3040; }
  .stat .n { font-size: 28px; font-weight: bold; color: #2e75b6; }
  .stat .label { font-size: 11px; color: #7a8ba0; margin-top: 4px; }
  .green { color: #27ae60; } .orange { color: #e67e22; } .red { color: #c0392b; }
  table.data { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.data th { text-align: left; padding: 8px 12px; background: #13161f;
                  color: #7a9fc0; border-bottom: 1px solid #2a3040; white-space: nowrap; }
  table.data td { padding: 8px 12px; border-bottom: 1px solid #1e2535;
                  vertical-align: top; word-break: break-word; }
  table.data tr:hover td { background: #1e253a; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: bold; }
  .badge-success { background: #1a4a2a; color: #27ae60; }
  .badge-failed  { background: #4a1a1a; color: #c0392b; }
  .badge-partial { background: #4a3a1a; color: #e67e22; }
  textarea { width: 100%; background: #0f1117; border: 1px solid #2a3040; color: #e0e0e0;
             padding: 10px; border-radius: 4px; font-size: 13px; min-height: 60px; resize: vertical; }
  .answer-box { background: #0f1117; border-left: 3px solid #2e75b6; padding: 12px 16px;
                border-radius: 0 4px 4px 0; margin-top: 12px; font-size: 13px; line-height: 1.6; }
  label { font-size: 12px; color: #7a8ba0; display: block; margin-bottom: 4px; }
  .detail-row { display: none; }
  .detail-row td { background: #13161f; padding: 12px 16px; }
  .detail-row pre { max-height: 200px; }
  #status-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #1a1f2e;
                padding: 8px 32px; font-size: 12px; color: #7a8ba0;
                border-top: 1px solid #2a3040; z-index: 100; }
</style>
</head>
<body>

<header>
  <div>
    <h1>ESAM SE Cockpit</h1>
    <span>Embedded Systems AI-Assisted Development Methodology &middot; Pilot v2.0</span>
  </div>
</header>

<div class="tabs">
  <div class="tab active"  onclick="showTab('dashboard')">Dashboard</div>
  <div class="tab"         onclick="showTab('baseline')">Baseline Management</div>
  <div class="tab"         onclick="showTab('query')">Query Interface</div>
  <div class="tab"         onclick="showTab('ingest-history')">Ingestion History</div>
</div>

<!-- ── DASHBOARD TAB ─────────────────────────────────────────────────────── -->
<div id="tab-dashboard" class="panel active">
  <div class="card">
    <h2>Graph statistics</h2>
    <div id="graph-stats" class="stat-row">Loading...</div>
  </div>
  <div class="card">
    <h2>Coverage dashboard — live RTM</h2>
    <p style="font-size:12px;color:#7a8ba0;margin-bottom:12px;">
      Green = test passed &nbsp;&middot;&nbsp; Orange = not yet tested &nbsp;&middot;&nbsp; Red = test failed
    </p>
    <table class="data">
      <thead>
        <tr><th>System</th><th>WBS</th><th>Total SRS</th>
            <th class="green">Green</th><th class="orange">Orange</th><th class="red">Red</th></tr>
      </thead>
      <tbody id="coverage-rows"><tr><td colspan="6">Loading...</td></tr></tbody>
    </table>
    <br><button onclick="loadDashboard()">Refresh</button>
  </div>
</div>

<!-- ── BASELINE MANAGEMENT TAB ───────────────────────────────────────────── -->
<div id="tab-baseline" class="panel">
  <div class="card">
    <h2>Step 1 — Upload YAML and check diff</h2>
    <p style="font-size:12px;color:#7a8ba0;margin-bottom:12px;">
      Upload your updated vehicle_decomposition.yaml to see exactly what will change
      before approving. No changes are written until you hit Approve and Lock Baseline.
    </p>
    <label>Select YAML file:</label>
    <input type="file" id="diff-file" accept=".yaml,.yml">
    <button onclick="checkDiff()">Check Diff</button>
    <pre id="diff-result" style="margin-top:12px;display:none"></pre>
  </div>
  <div class="card">
    <h2>Step 2 — Approve and lock baseline</h2>
    <p style="font-size:12px;color:#c0392b;margin-bottom:12px;">
      &#9888; This action commits the YAML to Git, runs ingestion, and updates the live graph.
      Only proceed after reviewing the diff above and confirming with the SE team lead.
    </p>
    <label>SE Engineer name (approver):</label>
    <input type="text" id="approver-name" placeholder="First Last">
    <label>Select the same YAML file:</label>
    <input type="file" id="approve-file" accept=".yaml,.yml">
    <br>
    <button class="success" onclick="approveBaseline()">Approve and Lock Baseline</button>
    <pre id="approve-result" style="margin-top:12px;display:none"></pre>
  </div>
</div>

<!-- ── QUERY INTERFACE TAB ───────────────────────────────────────────────── -->
<div id="tab-query" class="panel">
  <div class="card">
    <h2>Natural language query</h2>
    <p style="font-size:12px;color:#7a8ba0;margin-bottom:12px;">
      Ask structural questions ("Which subsystems interface with the TSN switch?") or
      contextual questions about document content. The system routes automatically.
    </p>
    <textarea id="query-input"
      placeholder="e.g. Which safety-critical subsystems do not have an ICD attached?"></textarea>
    <br><br>
    <button onclick="runQuery()">Ask</button>
    <div id="query-result" style="display:none">
      <div style="font-size:11px;color:#7a8ba0;margin-top:8px" id="query-meta"></div>
      <div class="answer-box" id="query-answer"></div>
    </div>
  </div>
  <div class="card">
    <h2>Example questions</h2>
    <div style="display:flex;flex-direction:column;gap:6px">
      <button style="text-align:left;background:#1a2a3a;font-size:12px"
        onclick="setQ('Which subsystems are safety-critical?')">
        Which subsystems are safety-critical?</button>
      <button style="text-align:left;background:#1a2a3a;font-size:12px"
        onclick="setQ('What interfaces exist on the TSN switch subsystem?')">
        What interfaces exist on the TSN switch subsystem?</button>
      <button style="text-align:left;background:#1a2a3a;font-size:12px"
        onclick="setQ('Which systems have no ICD artifact attached?')">
        Which systems have no ICD artifact attached?</button>
      <button style="text-align:left;background:#1a2a3a;font-size:12px"
        onclick="setQ('How many SRS items are untested across all systems?')">
        How many SRS items are untested across all systems?</button>
    </div>
  </div>
</div>

<!-- ── INGESTION HISTORY TAB ─────────────────────────────────────────────── -->
<div id="tab-ingest-history" class="panel">
  <div class="card">
    <h2>Ingestion run history</h2>
    <p style="font-size:12px;color:#7a8ba0;margin-bottom:16px;">
      Every ingestion run is recorded automatically — whether triggered from the
      command line or via the Approve and Lock Baseline gate above.
      Click a row to see which files were ingested.
    </p>
    <button onclick="loadIngestHistory()" style="margin-bottom:16px">Refresh</button>
    <div id="ingest-summary" class="stat-row" style="margin-bottom:16px"></div>
    <table class="data" id="ingest-table">
      <thead>
        <tr>
          <th></th>
          <th>Timestamp (UTC)</th>
          <th>Mode</th>
          <th>Triggered by</th>
          <th>+Vertices</th>
          <th>+Edges</th>
          <th>+Chunks</th>
          <th>Duration</th>
          <th>Graph SHA</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="ingest-rows">
        <tr><td colspan="10" style="color:#7a8ba0;padding:20px">
          Loading — click Refresh if this persists
        </td></tr>
      </tbody>
    </table>
  </div>
</div>

<div id="status-bar">Ready</div>

<script>
const setStatus = (msg) => document.getElementById('status-bar').textContent = msg;
const setQ = (q) => { document.getElementById('query-input').value = q; };

const TAB_NAMES = ['dashboard', 'baseline', 'query', 'ingest-history'];

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', TAB_NAMES[i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'dashboard')       loadDashboard();
  if (name === 'ingest-history')  loadIngestHistory();
}

// ── DASHBOARD ──────────────────────────────────────────────────────────────

async function loadDashboard() {
  setStatus('Loading dashboard...');

  const stats = await fetch('/api/graph/stats').then(r => r.json());
  const statsEl = document.getElementById('graph-stats');

  // vertex stats
  let html = stats.vertices.map(v =>
    `<div class="stat"><div class="n">${v.n}</div><div class="label">${v.label}</div></div>`
  ).join('');
  // chunk count
  html += `<div class="stat"><div class="n">${stats.chunks}</div>
             <div class="label">Doc Chunks</div></div>`;
  statsEl.innerHTML = html;

  const cov = await fetch('/api/coverage').then(r => r.json());
  const tbody = document.getElementById('coverage-rows');
  if (cov.coverage.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:#7a8ba0">No SRS items yet — begin Sprint 1</td></tr>';
  } else {
    tbody.innerHTML = cov.coverage.map(r => `
      <tr>
        <td>${r.system_name}</td>
        <td style="font-family:monospace">${r.wbs_code}</td>
        <td>${r.total_srs}</td>
        <td class="green">${r.green}</td>
        <td class="orange">${r.orange}</td>
        <td class="red">${r.red}</td>
      </tr>`).join('');
  }
  setStatus('Dashboard loaded');
}

// ── BASELINE ───────────────────────────────────────────────────────────────

async function checkDiff() {
  const file = document.getElementById('diff-file').files[0];
  if (!file) { alert('Select a YAML file first'); return; }
  setStatus('Computing diff...');
  const fd = new FormData(); fd.append('file', file);
  const result = await fetch('/api/baseline/diff', {method:'POST', body:fd}).then(r=>r.json());
  const el = document.getElementById('diff-result');
  el.style.display = 'block';
  el.textContent = JSON.stringify(result, null, 2);
  setStatus('Diff complete: ' + result.summary);
}

async function approveBaseline() {
  const approver = document.getElementById('approver-name').value.trim();
  const file = document.getElementById('approve-file').files[0];
  if (!approver) { alert('Enter approver name'); return; }
  if (!file) { alert('Select the YAML file to approve'); return; }
  if (!confirm(`Approve and lock baseline as "${approver}"?\\nThis commits to Git and updates the live graph.`)) return;
  setStatus('Approving baseline...');
  const fd = new FormData();
  fd.append('approved_by', approver);
  fd.append('file', file);
  const result = await fetch('/api/baseline/approve', {method:'POST', body:fd}).then(r=>r.json());
  const el = document.getElementById('approve-result');
  el.style.display = 'block';
  el.textContent = JSON.stringify(result, null, 2);
  setStatus(result.status === 'approved'
    ? 'Baseline approved — git: ' + result.git_hash
    : 'Error during approval');
}

// ── QUERY ──────────────────────────────────────────────────────────────────

async function runQuery() {
  const q = document.getElementById('query-input').value.trim();
  if (!q) return;
  setStatus('Querying...');
  document.getElementById('query-result').style.display = 'none';
  const result = await fetch('/api/query', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({question: q})
  }).then(r => r.json());
  document.getElementById('query-meta').textContent =
    `Type: ${result.type || 'unknown'} · ${result.explanation || ''} · ${result.raw_result_count || 0} rows`;
  document.getElementById('query-answer').textContent = result.answer || result.error || 'No answer';
  document.getElementById('query-result').style.display = 'block';
  setStatus('Query complete');
}

// ── INGESTION HISTORY ──────────────────────────────────────────────────────

async function loadIngestHistory() {
  setStatus('Loading ingestion history...');
  const data = await fetch('/api/ingestion/runs?limit=50').then(r => r.json());
  const runs  = data.runs || [];

  // Summary stats
  const total    = runs.length;
  const success  = runs.filter(r => r.status === 'success').length;
  const failed   = runs.filter(r => r.status === 'failed').length;
  const vAdded   = runs.reduce((s, r) => s + (r.vertices_added || 0), 0);
  const eAdded   = runs.reduce((s, r) => s + (r.edges_added   || 0), 0);
  const cAdded   = runs.reduce((s, r) => s + (r.chunks_added  || 0), 0);

  document.getElementById('ingest-summary').innerHTML = `
    <div class="stat"><div class="n">${total}</div><div class="label">Total Runs</div></div>
    <div class="stat"><div class="n green">${success}</div><div class="label">Successful</div></div>
    <div class="stat"><div class="n red">${failed}</div><div class="label">Failed</div></div>
    <div class="stat"><div class="n">${vAdded}</div><div class="label">Vertices Added</div></div>
    <div class="stat"><div class="n">${eAdded}</div><div class="label">Edges Added</div></div>
    <div class="stat"><div class="n">${cAdded}</div><div class="label">Chunks Added</div></div>
  `;

  if (runs.length === 0) {
    document.getElementById('ingest-rows').innerHTML =
      '<tr><td colspan="10" style="color:#7a8ba0;padding:20px">No ingestion runs recorded yet. ' +
      'Run the ingestion script on VM-1 to populate this table.</td></tr>';
    setStatus('No runs found');
    return;
  }

  const tbody = document.getElementById('ingest-rows');
  tbody.innerHTML = runs.map((r, i) => {
    const ts   = r.run_at ? r.run_at.replace('T', ' ').slice(0, 19) : '—';
    const dur  = r.duration_sec != null ? r.duration_sec + 's' : '—';
    const sha  = r.graph_json_hash || '—';
    const badge = r.status === 'success'
      ? '<span class="badge badge-success">success</span>'
      : r.status === 'failed'
        ? '<span class="badge badge-failed">failed</span>'
        : `<span class="badge badge-partial">${r.status}</span>`;

    const files = Array.isArray(r.source_files) ? r.source_files : [];
    const detailId = 'detail-' + i;

    return `
      <tr onclick="toggleDetail('${detailId}')" style="cursor:pointer">
        <td style="color:#7a9fc0;font-size:11px">&#9654;</td>
        <td style="font-family:monospace;font-size:12px">${ts}</td>
        <td><code>${r.mode}</code></td>
        <td>${r.triggered_by || '—'}</td>
        <td class="green">+${r.vertices_added}</td>
        <td class="green">+${r.edges_added}</td>
        <td class="green">+${r.chunks_added}</td>
        <td>${dur}</td>
        <td style="font-family:monospace;font-size:11px">${sha}</td>
        <td>${badge}</td>
      </tr>
      <tr id="${detailId}" class="detail-row">
        <td colspan="10">
          <strong style="font-size:12px;color:#7a9fc0">Files ingested this run:</strong><br>
          ${files.length === 0
            ? '<span style="color:#7a8ba0;font-size:12px">none recorded</span>'
            : files.map(f => `<code style="font-size:12px;display:block;margin:2px 0">${f}</code>`).join('')
          }
          ${r.error_detail
            ? `<br><strong style="color:#c0392b;font-size:12px">Error:</strong>
               <pre style="color:#c0392b;border-color:#4a1a1a">${escHtml(r.error_detail)}</pre>`
            : ''
          }
        </td>
      </tr>`;
  }).join('');

  setStatus(`Loaded ${runs.length} ingestion run(s)`);
}

function toggleDetail(id) {
  const row = document.getElementById(id);
  if (row) row.style.display = row.style.display === 'table-row' ? 'none' : 'table-row';
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── INIT ───────────────────────────────────────────────────────────────────
loadDashboard();
</script>
</body>
</html>
"""
