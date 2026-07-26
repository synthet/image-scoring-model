# /wiki-query — Query the wiki and optionally file the answer

Answer a question by searching the docs wiki. Good answers can be filed back as new wiki pages so explorations compound in the knowledge base.

## Inputs

- A question or topic from the user message.
- Optional: `--file` flag to save the answer as a wiki page.

## Schema

Read `docs/OKF_ADOPTION.md` and `docs/WIKI_SCHEMA.md` for OKF metadata and conventions. Prefer routing via `docs/CANONICAL_SOURCES.md` for contract questions.

## Steps

1. **Read `docs/INDEX.md`** to identify relevant pages. Scan folder-level INDEX.md files if needed.
2. **Read the relevant pages** (typically 3–10 pages depending on scope).
3. **Synthesize an answer** with inline citations linking to source pages.
4. **Present the answer** to the user in the chat.
5. **Ask if the answer should be filed** as a wiki page (or file automatically if `--file` was specified).
6. If filing:
   - Write the answer under `docs/reports/` (create the folder + INDEX.md if missing) with OKF frontmatter.
   - Update indexes and add cross-references from cited pages.
   - Append to `docs/log.md`.

## Answer format

```markdown
## Answer

[Synthesized response with citations]

### Sources

- [PAGE.md](path) — what this page contributed
```

## Done when

- Question is answered with citations to specific wiki pages.
- If filed: page exists, indexes updated, log entry appended.

## When to file

File when the answer is **durable, reusable knowledge**. Do not file simple lookups or ephemeral answers.
