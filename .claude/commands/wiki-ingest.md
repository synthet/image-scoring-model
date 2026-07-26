# /wiki-ingest — Process a source into the wiki

Ingest a new source document (article, paper, report, code analysis, or user-provided file) into the docs wiki. The wiki is a persistent, compounding knowledge base — each ingest updates multiple pages.

## Inputs

- Source document: file path, URL, or pasted content from the user message.
- User guidance on what to emphasize (optional).

## Schema

Read `docs/OKF_ADOPTION.md` and `docs/WIKI_SCHEMA.md` for OKF metadata and conventions, page types, and linking rules before proceeding.

## Steps

1. **Read the source** in full. If it's a URL, fetch it. If it's a file, read it.
2. **Summarize key takeaways** to the user (3-5 bullets). Ask what to emphasize or de-emphasize before writing.
3. **Determine page type and location** per the schema (technical, guides, reports, architecture, reference).
4. **Write the summary page** in the appropriate `docs/` subfolder following naming conventions, with OKF frontmatter containing at least `type` and preferably `title`, `description`, `resource`, `tags`, `timestamp`, and `okf_version: 0.1`.
5. **Update the folder's INDEX.md** — add the new page with a one-line description.
6. **Update `docs/INDEX.md`** if the page belongs to a category not yet represented there.
7. **Cross-reference existing pages:**
   - Read `docs/INDEX.md` and `docs/CANONICAL_SOURCES.md` to identify related pages.
   - Read those pages and add bidirectional links where relevant.
   - If new content contradicts existing pages, update them and note the discrepancy.
8. **If the source is a file**, copy or move it to `docs/raw/` (immutable archive) when archiving is useful.
9. **Append to `docs/log.md`** under the current month: `- YYYY-MM-DD: ingested — <details and paths>`

## Done when

- Summary page exists in the correct location.
- All relevant INDEX.md files are updated.
- Cross-references added to/from related pages when they exist.
- Log entry appended.
- User has reviewed the summary.

## Mine vs wiki

- `/mine` → executable agent assets (skills, commands, rules) under `.cursor/`
- `/wiki-ingest` → human-readable docs pages under `docs/`
