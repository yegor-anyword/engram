-- ============================================================================
-- Engram SQLite Schema  v0.1
-- ============================================================================
-- This file is the canonical database schema. The server runs it automatically
-- on first start (via `IF NOT EXISTS`), so there is no manual setup.
--
-- engram.db is gitignored because it contains runtime data. This file IS
-- tracked in git so every developer and CI pipeline can recreate the database
-- from scratch.
--
-- Tables are ordered by dependency: contexts first, then everything that
-- references a context_id.
-- ============================================================================

PRAGMA journal_mode = WAL;            -- better concurrent read performance
PRAGMA foreign_keys = ON;             -- enforce referential integrity

-- ── Contexts ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS contexts (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    owner           TEXT NOT NULL DEFAULT 'default',

    -- health metrics (denormalized, updated by triggers / application code)
    total_bullets       INTEGER NOT NULL DEFAULT 0,
    avg_salience        REAL    NOT NULL DEFAULT 0.0,
    stale_bullet_count  INTEGER NOT NULL DEFAULT 0,
    schema_count        INTEGER NOT NULL DEFAULT 0,
    last_consolidation  TEXT,                         -- ISO-8601 timestamp

    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ── Intent Anchors ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS intent_anchors (
    id              TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,

    -- immutable core
    objective       TEXT NOT NULL,
    success_criteria TEXT NOT NULL DEFAULT '[]',      -- JSON array of strings
    constraints     TEXT NOT NULL DEFAULT '[]',       -- JSON array of strings

    -- mutable progress
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active','paused','completed','abandoned')),
    parent_intent_id TEXT REFERENCES intent_anchors(id),  -- for sub-intents
    progress_notes  TEXT NOT NULL DEFAULT '[]',       -- JSON array of strings

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_intents_context ON intent_anchors(context_id);

-- ── Bullets ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bullets (
    id              TEXT PRIMARY KEY,                 -- 8-char UUID prefix
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
    section         TEXT NOT NULL,                    -- grouping key
    content         TEXT NOT NULL,                    -- 1-2 sentences
    bullet_type     TEXT NOT NULL
                    CHECK (bullet_type IN (
                        'fact','decision','strategy','warning',
                        'procedure','exception','principle'
                    )),
    source_type     TEXT NOT NULL DEFAULT 'reflection'
                    CHECK (source_type IN (
                        'reflection','execution_feedback','user_input',
                        'consolidation','schema_derived'
                    )),

    -- embedding (JSON array of floats for SQLite; real vector column for pgvector)
    embedding       TEXT,                             -- JSON array, nullable until computed

    -- usage tracking
    hit_count       INTEGER NOT NULL DEFAULT 0,
    miss_count      INTEGER NOT NULL DEFAULT 0,
    recall_count    INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TEXT,

    -- salience & lifecycle
    salience        REAL NOT NULL DEFAULT 0.5,
    confidence      REAL NOT NULL DEFAULT 0.5,
    lifecycle_state TEXT NOT NULL DEFAULT 'active'
                    CHECK (lifecycle_state IN ('active','archived','purged')),
    archive_reason  TEXT,

    -- provenance
    source_session  TEXT,
    source_agent    TEXT,
    schema_id       TEXT REFERENCES schema_nodes(id) ON DELETE SET NULL,

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_bullets_context    ON bullets(context_id);
CREATE INDEX IF NOT EXISTS idx_bullets_section    ON bullets(context_id, section);
CREATE INDEX IF NOT EXISTS idx_bullets_type       ON bullets(context_id, bullet_type);
CREATE INDEX IF NOT EXISTS idx_bullets_salience   ON bullets(context_id, salience DESC);
CREATE INDEX IF NOT EXISTS idx_bullets_lifecycle  ON bullets(context_id, lifecycle_state);

-- ── Schema Nodes ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS schema_nodes (
    id              TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',

    slots           TEXT NOT NULL DEFAULT '{}',       -- JSON object
    exceptions      TEXT NOT NULL DEFAULT '[]',       -- JSON array of strings
    instance_count  INTEGER NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 0.0,
    bullet_ids      TEXT NOT NULL DEFAULT '[]',       -- JSON array of bullet IDs

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_schemas_context ON schema_nodes(context_id);

-- ── Concept Edges ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS concept_edges (
    id              TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
    from_id         TEXT NOT NULL,                    -- bullet or schema ID
    to_id           TEXT NOT NULL,                    -- bullet or schema ID
    edge_type       TEXT NOT NULL
                    CHECK (edge_type IN (
                        'caused_by','contradicts','depends_on','preferred_over',
                        'part_of','preceded_by','derived_from','supports',
                        'blocks','related_to','co_recalled'
                    )),

    weight          REAL NOT NULL DEFAULT 0.5,        -- 0.0 – 1.0
    rationale       TEXT,
    source_session  TEXT,

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_edges_context  ON concept_edges(context_id);
CREATE INDEX IF NOT EXISTS idx_edges_from     ON concept_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to       ON concept_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_edges_type     ON concept_edges(context_id, edge_type);

-- ── Delta Batches ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS delta_batches (
    id              TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
    trigger         TEXT NOT NULL,                    -- 'commit', 'consolidation', 'refine', 'user_edit'

    -- stats (populated after application)
    bullets_added   INTEGER NOT NULL DEFAULT 0,
    bullets_updated INTEGER NOT NULL DEFAULT 0,
    bullets_removed INTEGER NOT NULL DEFAULT 0,
    bullets_merged  INTEGER NOT NULL DEFAULT 0,

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_deltas_context ON delta_batches(context_id);

-- ── Delta Operations ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS delta_operations (
    id              TEXT PRIMARY KEY,
    batch_id        TEXT NOT NULL REFERENCES delta_batches(id) ON DELETE CASCADE,
    op_type         TEXT NOT NULL
                    CHECK (op_type IN (
                        'ADD_BULLET','UPDATE_BULLET','REMOVE_BULLET','MERGE_BULLETS',
                        'ADD_EDGE','REMOVE_EDGE','ADD_SCHEMA','UPDATE_SCHEMA'
                    )),

    target_id       TEXT,                             -- for UPDATE / REMOVE
    target_ids      TEXT,                             -- JSON array, for MERGE
    section         TEXT,
    content         TEXT,
    bullet_type     TEXT,

    reasoning       TEXT NOT NULL,
    source          TEXT NOT NULL
                    CHECK (source IN ('reflector','curator','user','consolidation')),
    confidence      REAL NOT NULL DEFAULT 0.5,

    session_id      TEXT,
    agent_id        TEXT,
    previous_state  TEXT,                             -- JSON snapshot for rollback

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_ops_batch ON delta_operations(batch_id);

-- ── Activity Ledger ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS activities (
    id              TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
    agent_id        TEXT NOT NULL,
    session_id      TEXT,

    action_type     TEXT NOT NULL
                    CHECK (action_type IN (
                        'task_started','task_completed','task_failed',
                        'decision_made','fact_learned','hypothesis_formed',
                        'tool_called','user_input_received',
                        'context_branched','context_merged',
                        'consolidation_ran','materialization_occurred'
                    )),

    summary         TEXT NOT NULL,
    raw_content     TEXT,                             -- original text submitted via commit()
    content_type    TEXT DEFAULT 'conversation'
                    CHECK (content_type IN ('conversation','tool_output','document')),

    -- links
    delta_batch_id      TEXT REFERENCES delta_batches(id),
    materialization_id  TEXT REFERENCES materializations(id),

    -- execution feedback (stored as JSON when provided)
    feedback        TEXT,                             -- JSON ExecutionFeedback object

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_activities_context ON activities(context_id);
CREATE INDEX IF NOT EXISTS idx_activities_agent   ON activities(context_id, agent_id);

-- ── Materializations ────────────────────────────────────────────────────────
-- Tracks every materialize() call so reconsolidation can link commit feedback
-- back to the specific bullets that were recalled.

CREATE TABLE IF NOT EXISTS materializations (
    id              TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
    agent_id        TEXT,

    query           TEXT NOT NULL,                    -- the materialization query
    target_model    TEXT,                             -- 'claude', 'gpt', 'gemini', 'generic'
    token_budget    INTEGER,

    rendered_text   TEXT NOT NULL,                    -- what was returned to the agent
    bullets_included TEXT NOT NULL DEFAULT '[]',      -- JSON array of bullet IDs
    bullets_excluded_reasons TEXT DEFAULT '{}',       -- JSON object {bullet_id: reason}
    token_count     INTEGER NOT NULL DEFAULT 0,
    coverage_score  REAL NOT NULL DEFAULT 0.0,

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_materializations_context ON materializations(context_id);

-- ── Ingestion Config ────────────────────────────────────────────────────────
-- Server-side configuration exposed via GET/PUT /config/ingestion.
-- Single row table — always id = 'default'.

CREATE TABLE IF NOT EXISTS ingestion_config (
    id                      TEXT PRIMARY KEY DEFAULT 'default',

    reflector_model         TEXT NOT NULL DEFAULT 'claude-haiku-4-5',
    reflector_prompt_version TEXT NOT NULL DEFAULT 'v1',
    max_reflection_rounds   INTEGER NOT NULL DEFAULT 2,

    curator_dedup_threshold REAL NOT NULL DEFAULT 0.92,
    curator_slow_path_model TEXT NOT NULL DEFAULT 'claude-haiku-4-5',

    embedding_model         TEXT NOT NULL DEFAULT 'text-embedding-3-small',

    consolidation_trigger   TEXT NOT NULL DEFAULT 'every_10_commits',
    fast_decay_rate         REAL NOT NULL DEFAULT 0.97,
    slow_decay_rate         REAL NOT NULL DEFAULT 0.995,

    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Seed the default config row if it doesn't exist
INSERT OR IGNORE INTO ingestion_config (id) VALUES ('default');

-- ── Context Locks ───────────────────────────────────────────────────────────
-- Advisory locks for serializing delta application per context.
-- SQLite uses this table + application-level locking.
-- PostgreSQL would use pg_advisory_lock() instead.

CREATE TABLE IF NOT EXISTS context_locks (
    context_id      TEXT PRIMARY KEY REFERENCES contexts(id) ON DELETE CASCADE,
    locked_by       TEXT,                             -- agent or process ID
    locked_at       TEXT,
    expires_at      TEXT                              -- auto-release after timeout
);

-- ── Schema Version ──────────────────────────────────────────────────────────
-- Tracks which migration version the database is at.
-- Future migrations check this before applying changes.

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema — contexts, bullets, schemas, edges, deltas, activities, materializations');
