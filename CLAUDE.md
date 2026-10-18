# CLAUDE.md

INSY team repo: one folder per assignment (`NN-topic/`), each with its own `CLAUDE.md` and `README.md`.

## Context

- Global context and conventions: `_claude/context.md` in the main Obsidian vault (coding, python, c-family conventions).
- School vault: `_INDEX.md`; per assignment a project note `INSY/NN-topic/_overview.md` (goal, status, decisions, progress).
- Documentation: school vault `INSY/` (German, Obsidian Markdown), published as a Quarto website on GitHub Pages. Build: QuickAdd "INSY → GitHub Pages" on the PC (`_export/insy-site.py` in the school vault). Claude writes the pages but does not render or publish. Published texts stay neutral: no "wir", no school, class or school year (the subject is fine).

## Constraints

- Write only inside this working directory and the Obsidian vaults (via connector). No scratchpad, no NuGet packages (restore writes to the user profile), `pip --no-cache-dir`. Temporary files go to `.tmp/` and are removed afterwards.
- `site/` is a generated copy of the vault's `INSY/` folder — never edit it.
- Commits only after the team's OK; the team pushes.
- Team project: several people write through the same account. Note who said what ("per Samuel", "per Jonas"); ask if unclear.
