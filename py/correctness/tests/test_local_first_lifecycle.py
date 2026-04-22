"""
Tests for database lifecycle in local-first scenarios.

A local-first database lives for months or years on a user's device,
accumulating schema migrations, data growth, and sync history. These tests
simulate that lifecycle: multiple schema changes over time, data that
survives migrations, sync across schema generations, and the general
wear-and-tear of a long-lived database.
"""

from crsql_correctness import connect, close, get_site_id, min_db_v
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_changes(c, since=0):
    return c.execute(
        "SELECT [table], pk, cid, val, col_version, db_version, site_id, cl, seq, ts "
        "FROM crsql_changes WHERE db_version > ? ORDER BY db_version, seq",
        (since,),
    ).fetchall()


def apply_changes(c, changes):
    for change in changes:
        c.execute(
            "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
        )
    c.commit()


def apply_changes_skipping_errors(c, changes):
    applied = []
    errors = []
    for change in changes:
        try:
            c.execute(
                "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
            )
            applied.append(change)
        except Exception as e:
            errors.append((change, e))
    if applied:
        c.commit()
    return applied, errors


def get_table_contents(c, table):
    return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def alter_add_column(c, table, col_name, col_type):
    """Non-destructive alter: begin, alter, commit."""
    c.execute(f"SELECT crsql_begin_alter('{table}')")
    c.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
    c.execute(f"SELECT crsql_commit_alter('main', '{table}', 1)")
    c.commit()


def get_crr_tables(c):
    """Return the set of user table names that are CRRs (have clock tables)."""
    rows = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%__crsql_clock'"
    ).fetchall()
    return {r[0].replace("__crsql_clock", "") for r in rows}


# ===========================================================================
# 1. MULTI-GENERATION SCHEMA EVOLUTION — successive migrations over time
# ===========================================================================


class TestSchemaEvolution:
    """
    Simulate an app that goes through several schema versions:
    v1: tasks(id, title)
    v2: tasks(id, title, priority)
    v3: tasks(id, title, priority, due_date) + tags(id, name)
    v4: tasks(id, title, priority, due_date, completed) + tags(id, name, color)
    """

    def test_four_generation_migration(self):
        """
        A single node goes through 4 schema versions. Data written at each
        version should survive all subsequent migrations. Changes from each
        era should be readable in crsql_changes.
        """
        c = connect(":memory:")

        # v1
        c.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        c.execute("SELECT crsql_as_crr('tasks')")
        c.commit()

        c.execute("INSERT INTO tasks VALUES (1, 'Buy groceries')")
        c.execute("INSERT INTO tasks VALUES (2, 'Walk the dog')")
        c.commit()

        v1_version = c.execute("SELECT crsql_db_version()").fetchone()[0]

        # v2: add priority
        alter_add_column(c, "tasks", "priority", "INTEGER")
        c.execute("UPDATE tasks SET priority = 1 WHERE id = 1")
        c.execute("UPDATE tasks SET priority = 2 WHERE id = 2")
        c.execute("INSERT INTO tasks VALUES (3, 'Read a book', 3)")
        c.commit()

        v2_version = c.execute("SELECT crsql_db_version()").fetchone()[0]

        # v3: add due_date + new table
        alter_add_column(c, "tasks", "due_date", "TEXT")
        c.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        c.execute("SELECT crsql_as_crr('tags')")
        c.commit()

        c.execute("UPDATE tasks SET due_date = '2024-01-15' WHERE id = 1")
        c.execute("INSERT INTO tags VALUES (1, 'personal')")
        c.execute("INSERT INTO tags VALUES (2, 'work')")
        c.commit()

        v3_version = c.execute("SELECT crsql_db_version()").fetchone()[0]

        # v4: add completed to tasks, color to tags
        alter_add_column(c, "tasks", "completed", "INTEGER")
        alter_add_column(c, "tags", "color", "TEXT")

        c.execute("UPDATE tasks SET completed = 0")
        c.execute("UPDATE tasks SET completed = 1 WHERE id = 2")
        c.execute("UPDATE tags SET color = 'blue' WHERE id = 1")
        c.execute("INSERT INTO tasks VALUES (4, 'Deploy app', 1, '2024-02-01', 0)")
        c.commit()

        # All data should be present with all columns
        tasks = get_table_contents(c, "tasks")
        assert len(tasks) == 4
        assert tasks[0] == (1, 'Buy groceries', 1, '2024-01-15', 0)
        assert tasks[1] == (2, 'Walk the dog', 2, None, 1)
        assert tasks[2] == (3, 'Read a book', 3, None, 0)
        assert tasks[3] == (4, 'Deploy app', 1, '2024-02-01', 0)

        tags = get_table_contents(c, "tags")
        assert len(tags) == 2
        assert tags[0] == (1, 'personal', 'blue')
        assert tags[1] == (2, 'work', None)

        # crsql_changes should span the full history
        all_changes = get_all_changes(c)
        assert len(all_changes) > 0

        # Changes include both tables
        tables_in_changes = set(ch[0] for ch in all_changes)
        assert "tasks" in tables_in_changes
        assert "tags" in tables_in_changes

        close(c)

    def test_sync_across_schema_generations(self):
        """
        Node A is at v3 (tasks with 3 columns + tags table). Node B is at
        v1 (tasks with 1 column, no tags). Sync A->B. B should get what it
        can handle (title column, sentinel rows). Unknown columns/tables
        should error.
        """
        # Node A at v3
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('tasks')")
        node_a.commit()
        alter_add_column(node_a, "tasks", "priority", "INTEGER")
        alter_add_column(node_a, "tasks", "due_date", "TEXT")
        node_a.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('tags')")
        node_a.commit()

        node_a.execute("INSERT INTO tasks VALUES (1, 'Task one', 1, '2024-01-01')")
        node_a.execute("INSERT INTO tags VALUES (1, 'urgent')")
        node_a.commit()

        # Node B at v1
        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('tasks')")
        node_b.commit()

        changes = get_all_changes(node_a)
        applied, errors = apply_changes_skipping_errors(node_b, changes)

        # B should have the task with just id and title
        rows = get_table_contents(node_b, "tasks")
        assert len(rows) == 1
        assert rows[0][0] == 1       # id
        assert rows[0][1] == 'Task one'  # title

        # Errors for unknown columns and unknown table
        error_cids = [c[2] for c, e in errors]
        error_tables = [c[0] for c, e in errors]
        assert 'priority' in error_cids or 'due_date' in error_cids
        assert 'tags' in error_tables

        close(node_a)
        close(node_b)

    def test_staggered_migration_eventual_convergence(self):
        """
        Two nodes start at v1. Node A migrates to v2, writes data, syncs to B.
        B migrates to v2, syncs from A again. Then A migrates to v3, writes,
        syncs. B migrates to v3, syncs again. Each step should converge on
        shared columns.
        """
        node_a = connect(":memory:")
        node_b = connect(":memory:")

        # Both at v1
        for n in [node_a, node_b]:
            n.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
            n.execute("SELECT crsql_as_crr('tasks')")
            n.commit()

        # Shared data at v1
        node_a.execute("INSERT INTO tasks VALUES (1, 'Shared task')")
        node_a.commit()
        apply_changes(node_b, get_all_changes(node_a))

        # A -> v2
        alter_add_column(node_a, "tasks", "priority", "INTEGER")
        node_a.execute("UPDATE tasks SET priority = 1 WHERE id = 1")
        node_a.execute("INSERT INTO tasks VALUES (2, 'A task', 2)")
        node_a.commit()

        # Sync A->B (B still v1)
        a_changes = get_all_changes(node_a)
        apply_changes_skipping_errors(node_b, a_changes)

        # B should have both rows with just the title column
        rows_b = get_table_contents(node_b, "tasks")
        assert len(rows_b) == 2

        # B -> v2
        alter_add_column(node_b, "tasks", "priority", "INTEGER")

        # Re-sync A->B — now B can apply priority column changes
        apply_changes(node_b, get_all_changes(node_a))

        rows_a = get_table_contents(node_a, "tasks")
        rows_b = get_table_contents(node_b, "tasks")
        assert rows_a == rows_b, f"After both at v2:\n  A: {rows_a}\n  B: {rows_b}"

        # A -> v3
        alter_add_column(node_a, "tasks", "status", "TEXT")
        node_a.execute("UPDATE tasks SET status = 'done' WHERE id = 1")
        node_a.commit()

        # Sync A->B (B at v2)
        apply_changes_skipping_errors(node_b, get_all_changes(node_a))

        # B -> v3
        alter_add_column(node_b, "tasks", "status", "TEXT")
        apply_changes(node_b, get_all_changes(node_a))

        rows_a = get_table_contents(node_a, "tasks")
        rows_b = get_table_contents(node_b, "tasks")
        assert rows_a == rows_b, f"After both at v3:\n  A: {rows_a}\n  B: {rows_b}"

        close(node_a)
        close(node_b)


# ===========================================================================
# 2. DATA THAT SPANS MIGRATIONS — old data works with new schema
# ===========================================================================


class TestDataAcrossMigrations:
    """
    Data written before a migration must continue to sync correctly after.
    """

    def test_pre_migration_data_syncs_post_migration(self):
        """
        Node A writes data at v1. Migrates to v2. The v1-era data should
        still be readable in crsql_changes and syncable to another v2 node.
        """
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('foo')")
        node_a.commit()

        # v1 data
        node_a.execute("INSERT INTO foo VALUES (1, 'old-data')")
        node_a.commit()
        v1_version = node_a.execute("SELECT crsql_db_version()").fetchone()[0]

        # Migrate
        alter_add_column(node_a, "foo", "score", "INTEGER")

        # v2 data
        node_a.execute("INSERT INTO foo VALUES (2, 'new-data', 100)")
        node_a.commit()

        # Fresh v2 node
        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT, score INTEGER) STRICT;")
        node_b.execute("SELECT crsql_as_crr('foo')")
        node_b.commit()

        # Sync everything — includes pre-migration row 1 and post-migration row 2
        apply_changes(node_b, get_all_changes(node_a))

        rows = get_table_contents(node_b, "foo")
        assert (1, 'old-data', None) in rows, "Pre-migration data should sync with NULL for new column"
        assert (2, 'new-data', 100) in rows, "Post-migration data should sync fully"

        close(node_a)
        close(node_b)

    def test_updates_to_pre_migration_rows_include_new_columns(self):
        """
        A row created before migration is updated after migration to set
        the new column. The change should be tracked and syncable.
        """
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('foo')")
        node_a.commit()

        node_a.execute("INSERT INTO foo VALUES (1, 'original')")
        node_a.commit()

        alter_add_column(node_a, "foo", "extra", "TEXT")

        node_a.execute("UPDATE foo SET extra = 'added-later' WHERE id = 1")
        node_a.commit()

        # Verify the change for 'extra' is in crsql_changes
        changes = get_all_changes(node_a)
        extra_changes = [c for c in changes if c[2] == 'extra']
        assert len(extra_changes) > 0, "Update to new column should produce a change"
        assert extra_changes[0][3] == 'added-later'

        # Sync to fresh node
        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT, extra TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('foo')")
        node_b.commit()

        apply_changes(node_b, changes)
        rows = get_table_contents(node_b, "foo")
        assert rows == [(1, 'original', 'added-later')]

        close(node_a)
        close(node_b)

    def test_delete_row_created_before_migration(self):
        """
        A row from v1 is deleted at v2. The delete should propagate
        correctly to a v2 peer.
        """
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, val TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('foo')")
        node_a.commit()

        node_a.execute("INSERT INTO foo VALUES (1, 'will-die')")
        node_a.execute("INSERT INTO foo VALUES (2, 'will-survive')")
        node_a.commit()

        alter_add_column(node_a, "foo", "extra", "TEXT")

        node_a.execute("DELETE FROM foo WHERE id = 1")
        node_a.commit()

        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, val TEXT, extra TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('foo')")
        node_b.commit()

        apply_changes(node_b, get_all_changes(node_a))

        rows = get_table_contents(node_b, "foo")
        assert len(rows) == 1
        assert rows[0][0] == 2

        close(node_a)
        close(node_b)


# ===========================================================================
# 3. ADDING AND REMOVING CRR TABLES — tables come and go over app versions
# ===========================================================================


class TestTableLifecycle:
    """
    Over an app's life, tables are added as CRRs, and sometimes tables are
    removed (or stop being CRRs).
    """

    def test_add_crr_table_in_later_version(self):
        """
        v1 has only 'tasks'. v2 adds 'comments' as a CRR. Data from both
        tables should sync to a v2 peer.
        """
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('tasks')")
        node_a.commit()

        node_a.execute("INSERT INTO tasks VALUES (1, 'First task')")
        node_a.commit()

        # v2: add comments table
        node_a.execute("CREATE TABLE comments (id INTEGER PRIMARY KEY NOT NULL, task_id INTEGER, body TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('comments')")
        node_a.commit()

        node_a.execute("INSERT INTO comments VALUES (1, 1, 'Great task!')")
        node_a.commit()

        # v2 peer
        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('tasks')")
        node_b.execute("CREATE TABLE comments (id INTEGER PRIMARY KEY NOT NULL, task_id INTEGER, body TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('comments')")
        node_b.commit()

        apply_changes(node_b, get_all_changes(node_a))

        assert get_table_contents(node_b, "tasks") == [(1, 'First task')]
        assert get_table_contents(node_b, "comments") == [(1, 1, 'Great task!')]

        close(node_a)
        close(node_b)

    def test_remove_crr_then_sync(self):
        """
        A table that was a CRR is dropped. Changes referencing it should
        error gracefully on a node that no longer has it.
        """
        # Node A still has the old table
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('tasks')")
        node_a.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY NOT NULL, data TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('legacy')")
        node_a.commit()

        node_a.execute("INSERT INTO tasks VALUES (1, 'Keep me')")
        node_a.execute("INSERT INTO legacy VALUES (1, 'Old data')")
        node_a.commit()

        # Node B has dropped legacy table
        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY NOT NULL, title TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('tasks')")
        node_b.commit()

        changes = get_all_changes(node_a)
        applied, errors = apply_changes_skipping_errors(node_b, changes)

        # tasks data should arrive
        assert get_table_contents(node_b, "tasks") == [(1, 'Keep me')]

        # legacy changes should error
        legacy_errors = [c for c, e in errors if c[0] == 'legacy']
        assert len(legacy_errors) > 0

        close(node_a)
        close(node_b)

    def test_multiple_tables_added_over_time(self):
        """
        Simulate 4 app versions each adding a new CRR table. Data from all
        tables should coexist and sync correctly.
        """
        c = connect(":memory:")

        tables = [
            ("users", "id INTEGER PRIMARY KEY NOT NULL, name TEXT"),
            ("posts", "id INTEGER PRIMARY KEY NOT NULL, user_id INTEGER, title TEXT"),
            ("comments", "id INTEGER PRIMARY KEY NOT NULL, post_id INTEGER, body TEXT"),
            ("likes", "id INTEGER PRIMARY KEY NOT NULL, post_id INTEGER, user_id INTEGER"),
        ]

        for i, (name, schema) in enumerate(tables):
            c.execute(f"CREATE TABLE {name} ({schema}) STRICT;")
            c.execute(f"SELECT crsql_as_crr('{name}')")
            c.commit()

            # Write some data at each "version"
            if name == "users":
                c.execute("INSERT INTO users VALUES (1, 'Alice')")
                c.execute("INSERT INTO users VALUES (2, 'Bob')")
            elif name == "posts":
                c.execute("INSERT INTO posts VALUES (1, 1, 'Hello World')")
            elif name == "comments":
                c.execute("INSERT INTO comments VALUES (1, 1, 'Nice post!')")
            elif name == "likes":
                c.execute("INSERT INTO likes VALUES (1, 1, 2)")
            c.commit()

        # Sync everything to a peer with all tables
        peer = connect(":memory:")
        for name, schema in tables:
            peer.execute(f"CREATE TABLE {name} ({schema}) STRICT;")
            peer.execute(f"SELECT crsql_as_crr('{name}')")
        peer.commit()

        apply_changes(peer, get_all_changes(c))

        assert get_table_contents(peer, "users") == [(1, 'Alice'), (2, 'Bob')]
        assert get_table_contents(peer, "posts") == [(1, 1, 'Hello World')]
        assert get_table_contents(peer, "comments") == [(1, 1, 'Nice post!')]
        assert get_table_contents(peer, "likes") == [(1, 1, 2)]

        close(c)
        close(peer)


# ===========================================================================
# 4. LONG-LIVED DATABASE CONSISTENCY — accumulated state over time
# ===========================================================================


class TestLongLivedConsistency:
    """
    After many operations, the internal bookkeeping (clock tables, db_versions,
    site_id ordinals) should remain consistent.
    """

    def test_db_version_monotonically_increases(self):
        """
        After many mixed operations across multiple migrations, db_version
        should only increase.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, val TEXT) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        versions = []
        # Inserts
        for i in range(1, 51):
            c.execute("INSERT INTO foo VALUES (?, ?)", (i, f"v{i}"))
            c.commit()
            versions.append(c.execute("SELECT crsql_db_version()").fetchone()[0])

        # Updates
        for i in range(1, 21):
            c.execute("UPDATE foo SET val = ? WHERE id = ?", (f"updated-{i}", i))
            c.commit()
            versions.append(c.execute("SELECT crsql_db_version()").fetchone()[0])

        # Migration
        alter_add_column(c, "foo", "extra", "INTEGER")
        versions.append(c.execute("SELECT crsql_db_version()").fetchone()[0])

        # More writes
        for i in range(1, 11):
            c.execute("UPDATE foo SET extra = ? WHERE id = ?", (i * 10, i))
            c.commit()
            versions.append(c.execute("SELECT crsql_db_version()").fetchone()[0])

        # Deletes
        for i in range(40, 51):
            c.execute("DELETE FROM foo WHERE id = ?", (i,))
            c.commit()
            versions.append(c.execute("SELECT crsql_db_version()").fetchone()[0])

        # Verify monotonic
        for i in range(1, len(versions)):
            assert versions[i] >= versions[i-1], (
                f"db_version went backward: {versions[i-1]} -> {versions[i]} at step {i}"
            )

        close(c)

    def test_clock_table_integrity_after_lifecycle(self):
        """
        After inserts, updates, deletes, and a migration, every row in the
        base table should have corresponding clock entries, and deleted rows
        should have sentinel-only entries.
        """
        c = connect(":memory:")
        c.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        c.execute("SELECT crsql_as_crr('foo')")
        c.commit()

        # Create 20 rows
        for i in range(1, 21):
            c.execute("INSERT INTO foo VALUES (?, ?)", (i, f"name-{i}"))
        c.commit()

        # Update some
        for i in range(1, 11):
            c.execute("UPDATE foo SET name = ? WHERE id = ?", (f"updated-{i}", i))
        c.commit()

        # Migrate
        alter_add_column(c, "foo", "score", "INTEGER")

        # Delete some
        for i in range(15, 21):
            c.execute("DELETE FROM foo WHERE id = ?", (i,))
        c.commit()

        # Check: every alive row has clock entries
        alive_rows = c.execute("SELECT id FROM foo ORDER BY id").fetchall()
        alive_ids = {r[0] for r in alive_rows}
        assert len(alive_ids) == 14  # 20 created - 6 deleted

        # Clock table should have entries for all rows (alive and deleted)
        clock_keys = c.execute(
            "SELECT DISTINCT key FROM foo__crsql_clock"
        ).fetchall()
        assert len(clock_keys) >= 20, "Clock table should have entries for all 20 rows"

        # Deleted rows should have sentinel entries
        for i in range(15, 21):
            sentinel = c.execute(
                "SELECT col_name FROM foo__crsql_clock WHERE key = (SELECT __crsql_key FROM foo__crsql_pks WHERE id = ?)",
                (i,),
            ).fetchall()
            col_names = [r[0] for r in sentinel]
            assert '-1' in col_names, f"Deleted row {i} should have sentinel in clock table"

        close(c)

    def test_full_lifecycle_sync_to_fresh_node(self):
        """
        A database goes through a full lifecycle (create, write, migrate,
        write more, delete, migrate again). Then syncs everything to a
        brand new node at the latest schema. The fresh node should match.
        """
        old = connect(":memory:")
        old.execute("CREATE TABLE items (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        old.execute("SELECT crsql_as_crr('items')")
        old.commit()

        # v1 writes
        old.execute("INSERT INTO items VALUES (1, 'alpha')")
        old.execute("INSERT INTO items VALUES (2, 'beta')")
        old.execute("INSERT INTO items VALUES (3, 'gamma')")
        old.commit()

        # v2 migration
        alter_add_column(old, "items", "priority", "INTEGER")
        old.execute("UPDATE items SET priority = 1 WHERE id = 1")
        old.execute("UPDATE items SET priority = 3 WHERE id = 3")
        old.commit()

        # More writes, a delete
        old.execute("DELETE FROM items WHERE id = 2")
        old.execute("INSERT INTO items VALUES (4, 'delta', 2)")
        old.commit()

        # v3 migration
        alter_add_column(old, "items", "done", "INTEGER")
        old.execute("UPDATE items SET done = 1 WHERE id = 1")
        old.execute("UPDATE items SET done = 0 WHERE id = 3")
        old.execute("UPDATE items SET done = 0 WHERE id = 4")
        old.commit()

        # Fresh node at v3
        fresh = connect(":memory:")
        fresh.execute("CREATE TABLE items (id INTEGER PRIMARY KEY NOT NULL, name TEXT, priority INTEGER, done INTEGER) STRICT;")
        fresh.execute("SELECT crsql_as_crr('items')")
        fresh.commit()

        apply_changes(fresh, get_all_changes(old))

        old_rows = get_table_contents(old, "items")
        fresh_rows = get_table_contents(fresh, "items")
        assert old_rows == fresh_rows, f"Fresh node should match:\n  old: {old_rows}\n  fresh: {fresh_rows}"

        # Row 2 should be deleted on both
        assert all(r[0] != 2 for r in fresh_rows), "Row 2 should be deleted"

        close(old)
        close(fresh)

    def test_bidirectional_sync_after_independent_lifecycles(self):
        """
        Two nodes go through similar but independent lifecycles (different
        data, same migrations at different times). Then they sync. Should
        converge.
        """
        a = connect(":memory:")
        b = connect(":memory:")

        # Both start at v1
        for n in [a, b]:
            n.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY NOT NULL, body TEXT) STRICT;")
            n.execute("SELECT crsql_as_crr('notes')")
            n.commit()

        # A writes at v1
        a.execute("INSERT INTO notes VALUES (1, 'Note from A')")
        a.commit()

        # B writes at v1
        b.execute("INSERT INTO notes VALUES (2, 'Note from B')")
        b.commit()

        # A migrates to v2
        alter_add_column(a, "notes", "tag", "TEXT")
        a.execute("UPDATE notes SET tag = 'work' WHERE id = 1")
        a.execute("INSERT INTO notes VALUES (3, 'Another from A', 'personal')")
        a.commit()

        # B migrates to v2 independently
        alter_add_column(b, "notes", "tag", "TEXT")
        b.execute("UPDATE notes SET tag = 'home' WHERE id = 2")
        b.execute("INSERT INTO notes VALUES (4, 'Another from B', 'urgent')")
        b.commit()

        # Full bidirectional sync
        apply_changes(b, get_all_changes(a))
        apply_changes(a, get_all_changes(b))

        rows_a = get_table_contents(a, "notes")
        rows_b = get_table_contents(b, "notes")
        assert rows_a == rows_b, f"Should converge:\n  A: {rows_a}\n  B: {rows_b}"
        assert len(rows_a) == 4, "Should have all 4 notes"

        close(a)
        close(b)
