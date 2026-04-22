"""
Tests for schema version mismatch in local-first scenarios.

In local-first apps, devices run different app versions. Device A might have
migrated (added columns, new tables) while Device B is still on the old schema.
When they sync, changes may reference columns or tables that don't exist on
the other side.

The core merge code (`find_non_pk_col`) returns an error for unknown columns,
and `merge_insert` errors for unknown tables. These tests document and verify
that behavior, and test the paths that do work (old schema -> new schema, shared
columns, non-destructive alter).
"""

from crsql_correctness import connect, close, get_site_id, min_db_v
import pytest
import sqlite3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_schema_v1():
    """Original schema: foo(a PK, b)"""
    c = connect(":memory:")
    c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
    c.execute("SELECT crsql_as_crr('foo')")
    c.commit()
    return c


def make_schema_v2():
    """Migrated schema: foo(a PK, b, c) — added column c"""
    c = connect(":memory:")
    c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER, c TEXT) STRICT;")
    c.execute("SELECT crsql_as_crr('foo')")
    c.commit()
    return c


def make_schema_v2_alt():
    """Alternative migration: foo(a PK, b, d) — added column d instead of c"""
    c = connect(":memory:")
    c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER, d TEXT) STRICT;")
    c.execute("SELECT crsql_as_crr('foo')")
    c.commit()
    return c


def make_two_table_schema():
    """Schema with two CRR tables"""
    c = connect(":memory:")
    c.execute("CREATE TABLE foo (a INTEGER PRIMARY KEY NOT NULL, b INTEGER) STRICT;")
    c.execute("SELECT crsql_as_crr('foo')")
    c.execute("CREATE TABLE bar (x INTEGER PRIMARY KEY NOT NULL, y TEXT) STRICT;")
    c.execute("SELECT crsql_as_crr('bar')")
    c.commit()
    return c


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


def apply_change_expecting_error(c, change):
    """Apply a single change and expect it to raise an error. Returns the error."""
    try:
        c.execute(
            "INSERT INTO crsql_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", change
        )
        c.commit()
        return None
    except Exception as e:
        c.rollback()
        return e


def apply_changes_skipping_errors(c, changes):
    """Apply changes one at a time, skipping any that error. Returns (applied, errors)."""
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


def get_table_contents(c, table="foo"):
    return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


# ===========================================================================
# 1. COLUMN MISMATCH — new columns on one side but not the other
# ===========================================================================


class TestColumnMismatch:
    """
    Tests for when one node has columns the other doesn't know about.
    """

    def test_new_column_changes_to_old_schema_errors(self):
        """
        Node A (v2 schema with column c) inserts data using column c.
        Node B (v1 schema without column c) tries to merge those changes.

        Expected: changes referencing column 'c' error (find_non_pk_col fails).
        Changes for shared column 'b' should succeed.
        """
        node_a = make_schema_v2()
        node_b = make_schema_v1()

        # Node A writes using both old and new columns
        node_a.execute("INSERT INTO foo VALUES (1, 10, 'hello')")
        node_a.commit()

        changes = get_all_changes(node_a)

        # Separate changes by column
        shared_col_changes = [c for c in changes if c[2] in ('b', '-1')]  # b and sentinel
        new_col_changes = [c for c in changes if c[2] == 'c']

        assert len(new_col_changes) > 0, "Expected changes for column c"

        # Shared column changes should work
        apply_changes(node_b, shared_col_changes)

        # New column changes should error on old schema
        for change in new_col_changes:
            err = apply_change_expecting_error(node_b, change)
            assert err is not None, f"Expected error for unknown column 'c', but change succeeded: {change}"

        # Row should exist on B with the shared column value, but no column c
        rows = get_table_contents(node_b)
        assert len(rows) == 1
        assert rows[0] == (1, 10)  # only a, b — no c column

        close(node_a)
        close(node_b)

    def test_old_schema_changes_to_new_schema_works(self):
        """
        Node B (v1 schema) sends changes with only column 'b'.
        Node A (v2 schema with column 'c') receives them.

        Expected: changes for column 'b' merge fine. Column 'c' stays NULL.
        """
        node_b = make_schema_v1()
        node_a = make_schema_v2()

        node_b.execute("INSERT INTO foo VALUES (1, 42)")
        node_b.commit()

        changes = get_all_changes(node_b)
        apply_changes(node_a, changes)

        rows = get_table_contents(node_a)
        assert rows == [(1, 42, None)], f"Expected [(1, 42, None)], got: {rows}"

        close(node_a)
        close(node_b)

    def test_old_schema_changes_roundtrip(self):
        """
        Node B (v1) sends to Node A (v2). Node A makes local changes to column c.
        Node A sends all changes back to B. B should handle shared columns and
        error on the unknown column.
        """
        node_b = make_schema_v1()
        node_a = make_schema_v2()

        # B inserts
        node_b.execute("INSERT INTO foo VALUES (1, 10)")
        node_b.commit()

        # Sync B -> A
        changes_b = get_all_changes(node_b)
        apply_changes(node_a, changes_b)

        # A adds data in the new column
        node_a.execute("UPDATE foo SET c = 'extra' WHERE a = 1")
        node_a.commit()

        # Sync A -> B (includes changes for column c)
        changes_a = get_all_changes(node_a)
        applied, errors = apply_changes_skipping_errors(node_b, changes_a)

        # Column 'b' changes should have been applied (or were no-ops)
        # Column 'c' changes should have errored
        c_errors = [e for c, e in errors if c[2] == 'c']
        assert len(c_errors) > 0, "Expected errors for column 'c' changes"

        # B's data should still be intact
        rows = get_table_contents(node_b)
        assert rows == [(1, 10)], f"Expected B unchanged, got: {rows}"

        close(node_a)
        close(node_b)

    def test_both_nodes_add_different_columns(self):
        """
        Node A adds column 'c', Node B adds column 'd' (independent migrations).
        They try to sync. Each should handle its own known columns and error
        on the unknown ones.
        """
        node_a = make_schema_v2()      # has column c
        node_b = make_schema_v2_alt()  # has column d

        node_a.execute("INSERT INTO foo VALUES (1, 10, 'from-a')")
        node_a.commit()

        node_b.execute("INSERT INTO foo VALUES (2, 20, 'from-b')")
        node_b.commit()

        changes_a = get_all_changes(node_a)
        changes_b = get_all_changes(node_b)

        # A -> B: column 'c' changes error, but sentinel + 'b' should work
        applied_on_b, errors_on_b = apply_changes_skipping_errors(node_b, changes_a)
        c_errors = [e for c, e in errors_on_b if c[2] == 'c']
        assert len(c_errors) > 0, "Expected errors for column 'c' on node B"

        # B -> A: column 'd' changes error, but sentinel + 'b' should work
        applied_on_a, errors_on_a = apply_changes_skipping_errors(node_a, changes_b)
        d_errors = [e for c, e in errors_on_a if c[2] == 'd']
        assert len(d_errors) > 0, "Expected errors for column 'd' on node A"

        # Both should have both rows for the shared columns
        rows_a = get_table_contents(node_a)
        rows_b = get_table_contents(node_b)

        # Node A should have row 2 (from B) with b=20, c=NULL
        assert any(r[0] == 2 and r[1] == 20 for r in rows_a), \
            f"Expected row (2, 20, ...) on A, got: {rows_a}"
        # Node B should have row 1 (from A) with b=10, d=NULL
        assert any(r[0] == 1 and r[1] == 10 for r in rows_b), \
            f"Expected row (1, 10, ...) on B, got: {rows_b}"

        close(node_a)
        close(node_b)

    def test_non_destructive_alter_then_sync(self):
        """
        Both nodes start with the same schema. Node A adds a column via
        crsql_commit_alter (non-destructive), writes to it, then syncs to B.
        B has not yet migrated.

        This is the expected migration path for local-first: alter + commit_alter.
        """
        node_a = make_schema_v1()
        node_b = make_schema_v1()

        # Sync some initial data
        node_a.execute("INSERT INTO foo VALUES (1, 10)")
        node_a.commit()
        apply_changes(node_b, get_all_changes(node_a))

        # Node A migrates: add column, commit alter
        node_a.execute("SELECT crsql_begin_alter('foo')")
        node_a.execute("ALTER TABLE foo ADD COLUMN c TEXT")
        node_a.execute("SELECT crsql_commit_alter('main', 'foo', 1)")  # 1 = non-destructive
        node_a.commit()

        # Node A writes to new column
        node_a.execute("UPDATE foo SET c = 'migrated' WHERE a = 1")
        node_a.commit()

        changes_a = get_all_changes(node_a, since=1)

        # B has NOT migrated — column 'c' doesn't exist
        applied, errors = apply_changes_skipping_errors(node_b, changes_a)

        # Changes for shared column 'b' should work, 'c' should error
        c_errors = [e for c, e in errors if c[2] == 'c']
        assert len(c_errors) > 0, "Expected errors for column 'c'"

        # B's existing data should be intact
        rows = get_table_contents(node_b)
        assert rows == [(1, 10)]

        close(node_a)
        close(node_b)

    def test_both_nodes_migrate_then_sync(self):
        """
        Both nodes start with v1. Both independently migrate to v2 (add column c).
        Then they sync. Everything should work since schemas now match.
        """
        node_a = make_schema_v1()
        node_b = make_schema_v1()

        # Both migrate independently
        node_a.execute("SELECT crsql_begin_alter('foo')")
        node_a.execute("ALTER TABLE foo ADD COLUMN c TEXT")
        node_a.execute("SELECT crsql_commit_alter('main', 'foo', 1)")
        node_a.commit()

        node_b.execute("SELECT crsql_begin_alter('foo')")
        node_b.execute("ALTER TABLE foo ADD COLUMN c TEXT")
        node_b.execute("SELECT crsql_commit_alter('main', 'foo', 1)")
        node_b.commit()

        # Both write using new column
        node_a.execute("INSERT INTO foo VALUES (1, 10, 'from-a')")
        node_a.commit()
        node_b.execute("INSERT INTO foo VALUES (2, 20, 'from-b')")
        node_b.commit()

        # Full sync both ways
        apply_changes(node_b, get_all_changes(node_a))
        apply_changes(node_a, get_all_changes(node_b))

        rows_a = get_table_contents(node_a)
        rows_b = get_table_contents(node_b)
        assert rows_a == rows_b, f"Diverged:\n  A: {rows_a}\n  B: {rows_b}"
        assert len(rows_a) == 2

        close(node_a)
        close(node_b)


# ===========================================================================
# 2. TABLE MISMATCH — tables that exist on one side but not the other
# ===========================================================================


class TestTableMismatch:
    """
    Tests for when one node has CRR tables the other doesn't have.
    """

    def test_unknown_table_changes_error(self):
        """
        Node A has tables foo and bar. Node B only has foo.
        Changes for table bar should error on B.
        """
        node_a = make_two_table_schema()
        node_b = make_schema_v1()  # only has foo

        node_a.execute("INSERT INTO foo VALUES (1, 10)")
        node_a.execute("INSERT INTO bar VALUES (1, 'hello')")
        node_a.commit()

        changes = get_all_changes(node_a)
        foo_changes = [c for c in changes if c[0] == 'foo']
        bar_changes = [c for c in changes if c[0] == 'bar']

        assert len(bar_changes) > 0, "Expected changes for table bar"

        # foo changes should work
        apply_changes(node_b, foo_changes)
        assert get_table_contents(node_b) == [(1, 10)]

        # bar changes should error
        for change in bar_changes:
            err = apply_change_expecting_error(node_b, change)
            assert err is not None, f"Expected error for unknown table 'bar': {change}"

        close(node_a)
        close(node_b)

    def test_selective_sync_skipping_unknown_tables(self):
        """
        A practical pattern: filter changes by table name before applying,
        skipping tables that don't exist locally.
        """
        node_a = make_two_table_schema()
        node_b = make_schema_v1()

        node_a.execute("INSERT INTO foo VALUES (1, 10)")
        node_a.execute("INSERT INTO bar VALUES (1, 'hello')")
        node_a.commit()

        changes = get_all_changes(node_a)

        # B knows which tables it has — only apply matching ones
        local_tables = set(
            row[0] for row in node_b.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'crsql_%' AND name NOT LIKE '%__crsql_%' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )

        safe_changes = [c for c in changes if c[0] in local_tables]
        apply_changes(node_b, safe_changes)

        assert get_table_contents(node_b) == [(1, 10)]

        close(node_a)
        close(node_b)

    def test_new_table_created_on_both_then_sync(self):
        """
        Both nodes start with foo. Both independently create bar as a CRR.
        Then they sync. Should converge.
        """
        node_a = make_schema_v1()
        node_b = make_schema_v1()

        # Sync initial data
        node_a.execute("INSERT INTO foo VALUES (1, 10)")
        node_a.commit()
        apply_changes(node_b, get_all_changes(node_a))

        # Both create bar independently
        node_a.execute("CREATE TABLE bar (x INTEGER PRIMARY KEY NOT NULL, y TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('bar')")
        node_a.commit()

        node_b.execute("CREATE TABLE bar (x INTEGER PRIMARY KEY NOT NULL, y TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('bar')")
        node_b.commit()

        # Write to bar on both
        node_a.execute("INSERT INTO bar VALUES (1, 'from-a')")
        node_a.commit()
        node_b.execute("INSERT INTO bar VALUES (2, 'from-b')")
        node_b.commit()

        # Full sync
        apply_changes(node_b, get_all_changes(node_a))
        apply_changes(node_a, get_all_changes(node_b))

        rows_a = get_table_contents(node_a, "bar")
        rows_b = get_table_contents(node_b, "bar")
        assert rows_a == rows_b
        assert len(rows_a) == 2

        close(node_a)
        close(node_b)


# ===========================================================================
# 3. MIXED SCENARIOS — column + table mismatch combined
# ===========================================================================


class TestMixedSchemaMismatch:
    """
    Realistic scenarios where multiple schema differences exist simultaneously.
    """

    def test_filter_and_apply_compatible_changes(self):
        """
        Demonstrates the practical approach: filter incoming changes to only
        those matching local schema, apply them, and skip the rest. The skipped
        changes can be retried after local migration.
        """
        # Node A: advanced schema
        node_a = connect(":memory:")
        node_a.execute("CREATE TABLE users (id INTEGER PRIMARY KEY NOT NULL, name TEXT, email TEXT) STRICT;")
        node_a.execute("SELECT crsql_as_crr('users')")
        node_a.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY NOT NULL, title TEXT, author_id INTEGER) STRICT;")
        node_a.execute("SELECT crsql_as_crr('posts')")
        node_a.commit()

        # Node B: basic schema (no posts table, no email column)
        node_b = connect(":memory:")
        node_b.execute("CREATE TABLE users (id INTEGER PRIMARY KEY NOT NULL, name TEXT) STRICT;")
        node_b.execute("SELECT crsql_as_crr('users')")
        node_b.commit()

        # Node A writes data
        node_a.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@test.com')")
        node_a.execute("INSERT INTO posts VALUES (1, 'Hello World', 1)")
        node_a.commit()

        changes = get_all_changes(node_a)

        # Filter to only compatible changes
        local_tables = {'users'}
        local_columns = {'name', '-1'}  # -1 is sentinel

        compatible = []
        incompatible = []
        for change in changes:
            tbl, pk, cid = change[0], change[1], change[2]
            if tbl in local_tables and cid in local_columns:
                compatible.append(change)
            else:
                incompatible.append(change)

        assert len(incompatible) > 0, "Expected some incompatible changes"

        apply_changes(node_b, compatible)

        rows = get_table_contents(node_b, "users")
        assert len(rows) == 1
        assert rows[0][0] == 1      # id
        assert rows[0][1] == 'Alice' # name

        close(node_a)
        close(node_b)

    def test_gradual_migration_convergence(self):
        """
        Simulates a real upgrade path:
        1. Both start with v1 schema
        2. Node A upgrades, writes new data
        3. Node A syncs to B (B applies what it can)
        4. Node B upgrades
        5. Node A syncs again — now B can apply everything
        """
        node_a = make_schema_v1()
        node_b = make_schema_v1()

        # Initial shared state
        node_a.execute("INSERT INTO foo VALUES (1, 10)")
        node_a.commit()
        apply_changes(node_b, get_all_changes(node_a))

        # Step 2: Node A upgrades
        node_a.execute("SELECT crsql_begin_alter('foo')")
        node_a.execute("ALTER TABLE foo ADD COLUMN c TEXT")
        node_a.execute("SELECT crsql_commit_alter('main', 'foo', 1)")
        node_a.commit()

        node_a.execute("INSERT INTO foo VALUES (2, 20, 'new-data')")
        node_a.execute("UPDATE foo SET c = 'updated' WHERE a = 1")
        node_a.commit()

        # Step 3: Sync A -> B (B still on v1)
        all_changes_a = get_all_changes(node_a)
        applied, errors = apply_changes_skipping_errors(node_b, all_changes_a)

        # B should have the rows but missing column c data
        rows_b = get_table_contents(node_b)
        assert any(r[0] == 2 for r in rows_b), "Row 2 should exist on B"

        # Step 4: Node B upgrades
        node_b.execute("SELECT crsql_begin_alter('foo')")
        node_b.execute("ALTER TABLE foo ADD COLUMN c TEXT")
        node_b.execute("SELECT crsql_commit_alter('main', 'foo', 1)")
        node_b.commit()

        # Step 5: Re-sync from A — now all changes should apply
        all_changes_a = get_all_changes(node_a)
        apply_changes(node_b, all_changes_a)

        rows_a = get_table_contents(node_a)
        rows_b = get_table_contents(node_b)
        assert rows_a == rows_b, f"Should converge after migration:\n  A: {rows_a}\n  B: {rows_b}"

        close(node_a)
        close(node_b)
