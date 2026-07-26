# /wiki-lint — Health-check the docs wiki

Audit the wiki for structural problems, stale content, and missed connections.

## Schema

Read `docs/OKF_ADOPTION.md` and `docs/WIKI_SCHEMA.md` for the OKF frontmatter profile, conventions, and the full lint checklist.

## Steps

0. **Run automated lint** (from repo root):
   - `python scripts/okf_lint.py --profile project --exclude-prefix archive/`
   - `python scripts/wiki_lint.py --exclude-prefix archive/`
   Use findings as the starting point for manual review below.

1. **Scan all INDEX.md files** — build a list of every page referenced in indexes.
2. **Scan all .md files in `docs/`** — note living docs that lack OKF frontmatter or a non-empty `type`.
3. **Compare** to find orphans, broken index entries, and broken relative links.
4. **Check cross-references** — for each page, verify it has at least one inbound link (redirect stubs may be thin hubs).
5. **Optional full lint** — sample pages for stale claims vs current code; flag contradictions.
6. **Present findings** as a prioritized report (Critical / Structural / Stale / Suggestions).
7. **Ask the user** which fixes to apply; apply approved fixes and append to `docs/log.md`.

## Scope

- **Quick lint** (default): Steps 0–4.
- **Full lint**: include content staleness and contradiction checks (`/wiki-lint full`).

## Done when

- Report presented to the user.
- Approved fixes applied.
- Log entry appended when the wiki changed.
