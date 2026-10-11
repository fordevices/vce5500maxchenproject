#!/usr/bin/env python3
"""
ESAM Standalone Document Ingestion Script
==========================================
Works with PostgreSQL + pgvector only.
Generates embeddings locally using sentence-transformers (CPU, no API needed).
Also writes a local JSON graph file as interim graph layer.

Requirements:
    pip install sentence-transformers psycopg2-binary pgvector pyyaml PyPDF2

Usage:
    python3 esam_ingest_standalone.py --mode graph    # ingest YAML baseline -> graph
    python3 esam_ingest_standalone.py --mode docs     # ingest PDF/text docs -> pgvector
    python3 esam_ingest_standalone.py --mode both     # do both in sequence
    python3 esam_ingest_standalone.py --mode query    # interactive query test
    python3 esam_ingest_standalone.py --mode summary  # print graph stats only

Configuration: edit the CONFIG block below or set environment variables.
"""

import os, sys, json, uuid, argparse, hashlib, re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "esam_pilot")
DB_USER     = os.getenv("DB_USER",     "esam_user")
DB_PASS     = os.getenv("DB_PASS",     "your_password_here")

YAML_PATH   = os.getenv("YAML_PATH",   "vehicle_decomposition.yaml")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "documents")
GRAPH_JSON  = os.getenv("GRAPH_JSON",  "esam_graph.json")

# Embedding model - runs entirely on CPU, no API needed.
# First run downloads weights and caches them locally (~400MB for bge-base).
# Options ranked by quality vs RAM:
#   "BAAI/bge-base-en-v1.5"   best quality  ~440MB  <- RECOMMENDED
#   "BAAI/bge-small-en-v1.5"  good          ~130MB
#   "all-MiniLM-L6-v2"        fast          ~90MB
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

# ── DATABASE ──────────────────────────────────────────────────────────────────

def get_db():
    import psycopg2
    from psycopg2.extras import RealDictCursor
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        cursor_factory=RealDictCursor
    )

def ensure_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

        CREATE TABLE IF NOT EXISTS graph_vertices (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            label      TEXT NOT NULL,
            properties JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_gv_label ON graph_vertices (label);
        CREATE INDEX IF NOT EXISTS idx_gv_props ON graph_vertices USING GIN (properties);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_gv_entity_id
            ON graph_vertices ((properties->>'id'));

        CREATE TABLE IF NOT EXISTS graph_edges (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            from_id    UUID NOT NULL REFERENCES graph_vertices(id) ON DELETE CASCADE,
            to_id      UUID NOT NULL REFERENCES graph_vertices(id) ON DELETE CASCADE,
            type       TEXT NOT NULL,
            properties JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_ge_from ON graph_edges (from_id);
        CREATE INDEX IF NOT EXISTS idx_ge_to   ON graph_edges (to_id);
        CREATE INDEX IF NOT EXISTS idx_ge_type ON graph_edges (type);

        CREATE TABLE IF NOT EXISTS document_chunks (
            id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            artifact_id   UUID REFERENCES graph_vertices(id) ON DELETE SET NULL,
            artifact_name TEXT,
            chunk_index   INTEGER NOT NULL,
            content       TEXT NOT NULL,
            embedding     vector(768),
            source_file   TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_dc_artifact  ON document_chunks (artifact_id);
        CREATE INDEX IF NOT EXISTS idx_dc_source    ON document_chunks (source_file);
        CREATE INDEX IF NOT EXISTS idx_dc_embedding
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50);

        CREATE TABLE IF NOT EXISTS agent_task_summaries (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            crew_name  TEXT,
            agent_name TEXT,
            sprint_id  TEXT,
            task_type  TEXT NOT NULL,
            summary    TEXT NOT NULL,
            outcome    TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS entity_registry (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            entity_type TEXT NOT NULL,
            entity_id   TEXT NOT NULL UNIQUE,
            vertex_id   UUID REFERENCES graph_vertices(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_er_entity_id ON entity_registry (entity_id);

        CREATE TABLE IF NOT EXISTS baseline_audit_log (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            approved_by TEXT NOT NULL DEFAULT 'ingest_script',
            yaml_hash   TEXT NOT NULL,
            note        TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );

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
        CREATE INDEX IF NOT EXISTS idx_irl_run_at ON ingestion_run_log (run_at DESC);
    """)
    conn.commit()
    cur.close()
    print("Schema ready.")

# ── EMBEDDING — LOCAL CPU, NO API ─────────────────────────────────────────────

_embed_model = None

def get_embedder():
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        print("(First run downloads weights and caches locally — subsequent runs are instant)")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
        dim = _embed_model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {dim}")
        if dim != 768:
            print(f"WARNING: schema expects 768-dim vectors but model produces {dim}-dim.")
            print("Either update the schema or switch to a 768-dim model.")
    return _embed_model

def embed_texts(texts):
    model = get_embedder()
    # BGE models use a passage prefix for stored documents
    prefixed = [f"passage: {t}" for t in texts]
    vecs = model.encode(prefixed, show_progress_bar=True, batch_size=16)
    return vecs.tolist()

def embed_query(text):
    model = get_embedder()
    vec = model.encode([f"query: {text}"])
    return vec[0].tolist()

# ── GRAPH HELPERS ─────────────────────────────────────────────────────────────

def upsert_vertex(cur, label, props):
    entity_id = props.get("id")
    if not entity_id:
        raise ValueError(f"Vertex missing 'id': {props}")
    cur.execute("""
        INSERT INTO graph_vertices (label, properties)
        VALUES (%s, %s::jsonb)
        ON CONFLICT ((properties->>'id'))
        DO UPDATE SET
            label      = EXCLUDED.label,
            properties = EXCLUDED.properties,
            updated_at = now()
        RETURNING id
    """, (label, json.dumps(props)))
    row_id = str(cur.fetchone()["id"])
    # Keep entity_registry in sync
    cur.execute("""
        INSERT INTO entity_registry (entity_type, entity_id, vertex_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (entity_id) DO UPDATE SET vertex_id = EXCLUDED.vertex_id
    """, (label, entity_id, row_id))
    return row_id

def upsert_edge(cur, from_uuid, to_uuid, etype, props=None):
    if props is None:
        props = {}
    cur.execute("""
        INSERT INTO graph_edges (from_id, to_id, type, properties)
        VALUES (%s, %s, %s, %s::jsonb)
        ON CONFLICT DO NOTHING
    """, (from_uuid, to_uuid, etype, json.dumps(props)))

def get_uuid_by_entity_id(cur, entity_id):
    cur.execute(
        "SELECT id FROM graph_vertices WHERE properties->>'id' = %s",
        (entity_id,)
    )
    row = cur.fetchone()
    return str(row["id"]) if row else None

# ── YAML GRAPH INGESTION ──────────────────────────────────────────────────────

def ingest_yaml(conn, yaml_path):
    import yaml as pyyaml
    with open(yaml_path) as f:
        data = pyyaml.safe_load(f)

    yaml_hash = hashlib.sha256(open(yaml_path, "rb").read()).hexdigest()[:16]
    cur = conn.cursor()

    # Count rows before so we can report delta
    cur.execute("SELECT COUNT(*) AS n FROM graph_vertices")
    v_before = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM graph_edges")
    e_before = cur.fetchone()["n"]

    vehicle = data["vehicle"]
    v_uuid = upsert_vertex(cur, "Vehicle", {
        "id": vehicle["id"],
        "name": vehicle["name"],
        "wbs_code": vehicle.get("wbs_code", "1.0"),
        "programme": data.get("metadata", {}).get("programme", ""),
        "mil_std_ref": data.get("metadata", {}).get("mil_std_ref", "MIL-STD-881F Appendix G"),
    })
    print(f"  Vehicle: {vehicle['name']}")

    for family in vehicle.get("families", []):
        fam_uuid = upsert_vertex(cur, "VehicleFamily", {
            "id": family["id"], "name": family["name"],
            "wbs_code": family["wbs_code"],
        })
        upsert_edge(cur, fam_uuid, v_uuid, "PART_OF", {"wbs_code": family["wbs_code"]})

        for variant in family.get("variants", []):
            var_uuid = upsert_vertex(cur, "Variant", {
                "id": variant["id"], "name": variant["name"],
                "wbs_code": variant["wbs_code"],
                "role": variant.get("role", "variant"),
            })
            upsert_edge(cur, var_uuid, fam_uuid, "PART_OF", {"wbs_code": variant["wbs_code"]})
            print(f"  Variant: {variant['name']}")

            for system in variant.get("systems", []):
                sys_uuid = upsert_vertex(cur, "System", {
                    "id": system["id"], "name": system["name"],
                    "wbs_code": system["wbs_code"],
                    "universality": system.get("universality", "universal"),
                })
                upsert_edge(cur, sys_uuid, var_uuid, "PART_OF", {"wbs_code": system["wbs_code"]})
                print(f"    System: {system['name']} ({system['wbs_code']})")

                for sub in system.get("subsystems", []):
                    sub_props = {
                        "id": sub["id"], "name": sub["name"],
                        "safety_critical": sub.get("safety_critical", False),
                        "wbs_code": system["wbs_code"] + ".sub",
                    }
                    for field in ("tsn_profile", "safety_classification",
                                  "repo_url", "jira_component"):
                        if field in sub:
                            sub_props[field] = sub[field]
                    sub_uuid = upsert_vertex(cur, "Subsystem", sub_props)
                    upsert_edge(cur, sub_uuid, sys_uuid, "PART_OF")
                    print(f"      Subsystem: {sub['name']}")

                    for fn in sub.get("functions", []):
                        fn_props = {
                            "id": fn["id"], "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parent_subsystem": sub["id"],
                        }
                        if "safety_classification" in fn:
                            fn_props["safety_classification"] = fn["safety_classification"]
                        fn_uuid = upsert_vertex(cur, "Function", fn_props)
                        upsert_edge(cur, fn_uuid, sub_uuid, "PART_OF")
                        print(f"        Function: {fn['name']}")

    # Interfaces
    for iface in data.get("interfaces", []):
        from_uuid = get_uuid_by_entity_id(cur, iface["from"])
        to_uuid   = get_uuid_by_entity_id(cur, iface["to"])
        if not from_uuid or not to_uuid:
            print(f"  WARNING: interface {iface['from']} <-> {iface['to']} — node not found, skipping")
            continue
        props = {
            "interface_type":  iface["type"],
            "status":          iface.get("status", "candidate"),
            "source_artifact": iface.get("source_artifact", ""),
            "notes":           iface.get("notes", ""),
        }
        if "latency_budget_ns" in iface:
            props["latency_budget_ns"] = iface["latency_budget_ns"]
        if "ieee_profile" in iface:
            props["ieee_profile"] = iface["ieee_profile"]
        upsert_edge(cur, from_uuid, to_uuid, "INTERFACES_WITH", props)
        status = iface.get("status", "candidate")
        print(f"  Interface: {iface['from']} <-> {iface['to']} ({iface['type']}, {status})")

    # Artifacts
    for art in data.get("artifacts", []):
        art_props = {
            "id": art["id"], "name": art["name"],
            "artifact_type": art["type"],
            "revision": art.get("revision", "A"),
            "file_path": art.get("file_path", ""),
        }
        if "jira_ticket" in art:
            art_props["jira_ticket"] = art["jira_ticket"]
        art_uuid = upsert_vertex(cur, "DesignArtifact", art_props)
        parent_uuid = get_uuid_by_entity_id(cur, art["attached_to"])
        if parent_uuid:
            upsert_edge(cur, parent_uuid, art_uuid, "HAS_ARTIFACT",
                        {"artifact_type": art["type"]})
        print(f"  Artifact: {art['name']} -> {art['attached_to']}")

    cur.execute(
        "INSERT INTO baseline_audit_log (yaml_hash, note) VALUES (%s, %s)",
        (yaml_hash, f"Ingested from {yaml_path}")
    )
    conn.commit()

    # Compute deltas
    cur.execute("SELECT COUNT(*) AS n FROM graph_vertices")
    v_after = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM graph_edges")
    e_after = cur.fetchone()["n"]
    cur.close()

    vertices_added = v_after - v_before
    edges_added    = e_after - e_before
    print(f"\nGraph ingestion complete. YAML hash: {yaml_hash}")
    print(f"  +{vertices_added} vertices, +{edges_added} edges")
    write_json_graph(conn)
    return vertices_added, edges_added

def write_json_graph(conn):
    """Write full graph to local JSON — the interim graph layer for queries."""
    cur = conn.cursor()
    cur.execute("SELECT id, label, properties FROM graph_vertices ORDER BY label")
    vertices = [{"id": str(r["id"]), "label": r["label"],
                 "properties": dict(r["properties"])} for r in cur.fetchall()]
    cur.execute("SELECT id, from_id, to_id, type, properties FROM graph_edges")
    edges = [{"id": str(r["id"]), "from_id": str(r["from_id"]),
              "to_id": str(r["to_id"]), "type": r["type"],
              "properties": dict(r["properties"])} for r in cur.fetchall()]
    cur.close()

    graph = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vertex_count": len(vertices),
        "edge_count":   len(edges),
        "vertices":     vertices,
        "edges":        edges,
    }
    with open(GRAPH_JSON, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"JSON graph cache written: {GRAPH_JSON} "
          f"({len(vertices)} vertices, {len(edges)} edges)")

# ── DOCUMENT INGESTION ────────────────────────────────────────────────────────

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def extract_text(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            try:
                from pdfminer.high_level import extract_text as pm
                return pm(file_path)
            except ImportError:
                print("  Install PyPDF2: pip install PyPDF2")
                return ""
    elif suffix in (".txt", ".md", ".rst", ".html", ".xml"):
        return path.read_text(errors="ignore")
    else:
        try:
            return path.read_text(errors="ignore")
        except Exception:
            print(f"  Cannot read {path.name} — skipping")
            return ""

def get_artifact_uuid_by_name(cur, name):
    cur.execute("""
        SELECT id FROM graph_vertices
        WHERE label = 'DesignArtifact' AND properties->>'name' = %s
    """, (name,))
    row = cur.fetchone()
    return str(row["id"]) if row else None

def ingest_documents(conn, docs_folder):
    folder = Path(docs_folder)
    if not folder.exists():
        print(f"Documents folder not found: {docs_folder}")
        print("Create the folder, place your PDFs/text files in it, then run again.")
        return

    files = [f for f in folder.rglob("*") if f.is_file() and not f.name.startswith(".")]
    if not files:
        print(f"No files found in {docs_folder}")
        return

    print(f"Found {len(files)} file(s) to ingest\n")
    cur = conn.cursor()
    total_chunks = 0
    ingested_files = []

    for file_path in files:
        print(f"Processing: {file_path.name}")
        # Try to match filename stem to a graph artifact name
        stem = file_path.stem
        clean_name = re.sub(r"[-_](Rev|rev|v|V)\w*$", "", stem)
        artifact_uuid = (get_artifact_uuid_by_name(cur, clean_name)
                         or get_artifact_uuid_by_name(cur, stem))
        if artifact_uuid:
            print(f"  Matched artifact: {clean_name}")
        else:
            print(f"  No artifact match for '{clean_name}' — storing unlinked")

        text = extract_text(str(file_path))
        if not text.strip():
            print(f"  No text extracted — skipping\n")
            continue

        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks — embedding locally on CPU...")

        # Clear old chunks for clean re-ingestion
        cur.execute("DELETE FROM document_chunks WHERE source_file = %s",
                    (file_path.name,))

        BATCH = 32
        all_embeddings = []
        for i in range(0, len(chunks), BATCH):
            all_embeddings.extend(embed_texts(chunks[i:i+BATCH]))

        for idx, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
            cur.execute("""
                INSERT INTO document_chunks
                    (artifact_id, artifact_name, chunk_index, content, embedding, source_file)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
            """, (artifact_uuid, clean_name, idx, chunk, emb, file_path.name))

        conn.commit()
        print(f"  Stored {len(chunks)} chunks\n")
        total_chunks += len(chunks)
        ingested_files.append(file_path.name)

    cur.close()
    print("Document ingestion complete.")
    return total_chunks, ingested_files

# ── LOCAL JSON GRAPH QUERIES (no graph DB needed) ─────────────────────────────

def load_graph():
    if not Path(GRAPH_JSON).exists():
        print(f"Graph cache not found: {GRAPH_JSON}")
        print("Run --mode graph first.")
        return {"vertices": [], "edges": []}
    with open(GRAPH_JSON) as f:
        return json.load(f)

def v_by_id(graph, vid):
    for v in graph["vertices"]:
        if v["id"] == vid:
            return v
    return None

def v_by_entity(graph, eid):
    for v in graph["vertices"]:
        if v["properties"].get("id") == eid:
            return v
    return None

def edges_from(graph, vid, etype=None):
    return [e for e in graph["edges"]
            if e["from_id"] == vid and (etype is None or e["type"] == etype)]

def edges_to(graph, vid, etype=None):
    return [e for e in graph["edges"]
            if e["to_id"] == vid and (etype is None or e["type"] == etype)]

def print_graph_summary(graph):
    labels = Counter(v["label"] for v in graph["vertices"])
    etypes = Counter(e["type"]  for e in graph["edges"])
    print("\n── Graph summary " + "─" * 45)
    print("Vertices:")
    for lbl, n in sorted(labels.items()):
        print(f"  {lbl:25s} {n}")
    print("Edges:")
    for et, n in sorted(etypes.items()):
        print(f"  {et:25s} {n}")
    print(f"\nTotal: {graph['vertex_count']} vertices, {graph['edge_count']} edges")
    print(f"Generated: {graph.get('generated_at', '?')}\n")

def q_safety_critical(graph):
    return [v for v in graph["vertices"]
            if v["label"] == "Subsystem"
            and v["properties"].get("safety_critical") is True]

def q_missing_artifact(graph, artifact_type):
    covered = set()
    for v in graph["vertices"]:
        if v["label"] != "DesignArtifact":
            continue
        if v["properties"].get("artifact_type") != artifact_type:
            continue
        for e in edges_to(graph, v["id"], "HAS_ARTIFACT"):
            parent = v_by_id(graph, e["from_id"])
            if parent and parent["label"] in ("Subsystem", "System"):
                covered.add(parent["id"])
    return [v for v in graph["vertices"]
            if v["label"] == "Subsystem" and v["id"] not in covered]

def q_interfaces(graph, entity_id):
    v = v_by_entity(graph, entity_id)
    if not v:
        return []
    results = []
    for e in graph["edges"]:
        if e["type"] != "INTERFACES_WITH":
            continue
        if e["from_id"] == v["id"] or e["to_id"] == v["id"]:
            other_id = e["to_id"] if e["from_id"] == v["id"] else e["from_id"]
            other = v_by_id(graph, other_id)
            results.append({
                "other":    other["properties"].get("name", "?") if other else "?",
                "other_id": other["properties"].get("id",   "?") if other else "?",
                "props":    e["properties"],
            })
    return results

def q_candidates(graph):
    results = []
    for e in graph["edges"]:
        if e["type"] == "INTERFACES_WITH" \
                and e["properties"].get("status") == "candidate":
            frm = v_by_id(graph, e["from_id"])
            to  = v_by_id(graph, e["to_id"])
            results.append({
                "from": frm["properties"].get("name", "?") if frm else "?",
                "to":   to["properties"].get("name",  "?") if to  else "?",
                "type": e["properties"].get("interface_type", "?"),
            })
    return results

def q_functions(graph, entity_id):
    v = v_by_entity(graph, entity_id)
    if not v:
        return []
    return [v_by_id(graph, e["from_id"])["properties"]
            for e in edges_to(graph, v["id"], "PART_OF")
            if v_by_id(graph, e["from_id"]) and
               v_by_id(graph, e["from_id"])["label"] == "Function"]

def q_no_repo(graph):
    return [v for v in graph["vertices"]
            if v["label"] == "Subsystem"
            and not v["properties"].get("repo_url")]

# ── SEMANTIC SEARCH ───────────────────────────────────────────────────────────

def semantic_search(conn, question, top_k=5):
    q_vec = embed_query(question)
    cur = conn.cursor()
    cur.execute("""
        SELECT artifact_name, source_file, chunk_index, content,
               1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (q_vec, q_vec, top_k))
    results = [dict(r) for r in cur.fetchall()]
    cur.close()
    return results

# ── INTERACTIVE QUERY SHELL ───────────────────────────────────────────────────

HELP = """
Commands:
  summary                      graph node/edge counts
  safety                       list safety-critical subsystems
  missing ssdd                 subsystems with no SSDD
  missing icd                  subsystems with no ICD
  missing ssr                  subsystems with no SSR
  missing repo                 subsystems with no repo_url set
  interfaces <subsystem-id>    all interfaces for a subsystem
  candidates                   unconfirmed (candidate) interfaces
  functions <subsystem-id>     functions defined under a subsystem
  search <question>            semantic search over ingested documents
  quit                         exit
"""

def interactive_query(conn):
    graph = load_graph()
    print_graph_summary(graph)
    print("Interactive query shell. Type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            cmd = input("esam> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        lc = cmd.lower()
        if lc in ("quit", "exit", "q"):
            break
        elif lc == "help":
            print(HELP)
        elif lc == "summary":
            print_graph_summary(graph)
        elif lc == "safety":
            items = q_safety_critical(graph)
            print(f"\n{len(items)} safety-critical subsystem(s):")
            for v in items:
                p = v["properties"]
                sc = p.get("safety_classification", "")
                print(f"  {p.get('id','?'):20s}  {p.get('name','?'):40s}  {sc}")
        elif lc.startswith("missing "):
            atype = lc.split(" ", 1)[1].strip()
            if atype == "repo":
                items = q_no_repo(graph)
                print(f"\n{len(items)} subsystem(s) missing repo_url:")
                for v in items:
                    p = v["properties"]
                    flag = "  *** SAFETY CRITICAL" if p.get("safety_critical") else ""
                    print(f"  {p.get('id','?'):20s}  {p.get('name','?')}{flag}")
            else:
                items = q_missing_artifact(graph, atype)
                print(f"\n{len(items)} subsystem(s) missing {atype.upper()}:")
                for v in items:
                    p = v["properties"]
                    flag = "  *** SAFETY CRITICAL" if p.get("safety_critical") else ""
                    print(f"  {p.get('id','?'):20s}  {p.get('name','?')}{flag}")
        elif lc.startswith("interfaces "):
            eid = cmd.split(" ", 1)[1].strip()
            items = q_interfaces(graph, eid)
            print(f"\n{len(items)} interface(s) for {eid}:")
            for i in items:
                p = i["props"]
                print(f"  <-> {i['other_id']:20s}  {i['other']:35s}"
                      f"  {p.get('interface_type','?'):10s}"
                      f"  {p.get('status','?'):10s}"
                      f"  {p.get('source_artifact','none')}")
        elif lc == "candidates":
            items = q_candidates(graph)
            print(f"\n{len(items)} candidate (unconfirmed) interface(s):")
            for i in items:
                print(f"  {i['from']:35s} <-> {i['to']:35s}  ({i['type']})")
        elif lc.startswith("functions "):
            eid = cmd.split(" ", 1)[1].strip()
            items = q_functions(graph, eid)
            print(f"\n{len(items)} function(s) for {eid}:")
            for fn in items:
                print(f"  {fn.get('id','?'):20s}  {fn.get('name','?')}")
                if fn.get("description"):
                    print(f"  {'':20s}  {fn['description']}")
        elif lc.startswith("search "):
            question = cmd[7:].strip()
            if not question:
                print("Usage: search <your question>")
                continue
            print(f"\nSearching documents for: '{question}'")
            results = semantic_search(conn, question)
            if not results:
                print("No chunks found — run --mode docs first.")
            for i, r in enumerate(results, 1):
                sim = round(r["similarity"] * 100, 1)
                src = r["artifact_name"] or r["source_file"]
                print(f"\n[{i}] {src} — chunk {r['chunk_index']}  (similarity {sim}%)")
                print(f"    {r['content'][:220].strip()}...")
        else:
            print(f"Unknown command. Type 'help'.")

    print("Bye.")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def write_run_log(conn, mode, triggered_by, source_files,
                  vertices_added, edges_added, chunks_added,
                  duration_sec, status, error_detail=None):
    """Write one row to ingestion_run_log so every run is auditable."""
    graph_json_hash = None
    if Path(GRAPH_JSON).exists():
        graph_json_hash = hashlib.sha256(
            open(GRAPH_JSON, "rb").read()
        ).hexdigest()[:16]

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ingestion_run_log
            (mode, triggered_by, source_files,
             vertices_added, edges_added, chunks_added,
             duration_sec, status, error_detail, graph_json_hash)
        VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, run_at
    """, (
        mode,
        triggered_by,
        json.dumps(source_files),
        vertices_added,
        edges_added,
        chunks_added,
        duration_sec,
        status,
        error_detail,
        graph_json_hash,
    ))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    run_id = str(row["id"])
    run_at = row["run_at"]
    print(f"\n── Ingestion run recorded ──────────────────────────────────")
    print(f"  Run ID   : {run_id}")
    print(f"  Timestamp: {run_at}")
    print(f"  Mode     : {mode}")
    print(f"  Status   : {status}")
    print(f"  Vertices : +{vertices_added}   Edges: +{edges_added}   Chunks: +{chunks_added}")
    print(f"  Duration : {duration_sec:.1f}s")
    if graph_json_hash:
        print(f"  Graph SHA: {graph_json_hash}")
    print(f"────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="ESAM standalone ingestion")
    parser.add_argument("--mode",
        choices=["graph", "docs", "both", "query", "summary"],
        default="both")
    parser.add_argument("--triggered-by", default="cli",
        help="Name or identifier of who/what triggered this run (default: cli)")
    args = parser.parse_args()

    print(f"\nESAM Standalone Ingestion — mode: {args.mode}")
    print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}\n")

    conn = get_db()
    ensure_schema(conn)

    # query and summary modes don't write data — skip run logging
    if args.mode in ("query", "summary"):
        if args.mode == "query":
            interactive_query(conn)
        else:
            graph = load_graph()
            print_graph_summary(graph)
        conn.close()
        print("\nDone.")
        return

    # Ingestion modes — track everything for the run log
    import time
    t_start = time.time()
    status        = "success"
    error_detail  = None
    vertices_added = 0
    edges_added    = 0
    chunks_added   = 0
    source_files   = []

    try:
        if args.mode in ("graph", "both"):
            if not Path(YAML_PATH).exists():
                print(f"YAML not found: {YAML_PATH}")
                print("Set YAML_PATH or place vehicle_decomposition.yaml in the current directory.")
                sys.exit(1)
            print(f"── Ingesting graph from: {YAML_PATH}")
            va, ea = ingest_yaml(conn, YAML_PATH)
            vertices_added += va
            edges_added    += ea
            source_files.append(str(YAML_PATH))

        if args.mode in ("docs", "both"):
            print(f"\n── Ingesting documents from: {DOCS_FOLDER}/")
            ca, doc_files = ingest_documents(conn, DOCS_FOLDER)
            chunks_added += ca
            source_files.extend(doc_files)

    except Exception as exc:
        status       = "failed"
        error_detail = str(exc)
        print(f"\nERROR during ingestion: {exc}")

    duration_sec = round(time.time() - t_start, 2)

    write_run_log(
        conn,
        mode           = args.mode,
        triggered_by   = args.triggered_by,
        source_files   = source_files,
        vertices_added = vertices_added,
        edges_added    = edges_added,
        chunks_added   = chunks_added,
        duration_sec   = duration_sec,
        status         = status,
        error_detail   = error_detail,
    )

    conn.close()
    if status == "failed":
        sys.exit(1)
    print("\nDone.")

if __name__ == "__main__":
    main()
