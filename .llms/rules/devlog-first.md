# Always scan devlog titles first

For **every** question — conceptual or code-oriented — **always** start
by listing files under `devlog/compile/` (and subdirectories) to check
for relevant entries.  This is a cheap operation and avoids missing
important design rationale.

Devlog filenames are date-prefixed (`YYYY-MM-DD-title.md`).  **Prefer
newer entries** — older posts may be outdated or superseded.  When
multiple posts cover the same topic, the newest one is authoritative.

- **Conceptual / design / "how/why" questions:** Read matching posts
  **before** searching code or answering from memory.  Supplement with
  code search if needed.
- **Code tasks** (bug fixes, edits, refactors, "change this line", or
  when the user points at specific code): Only read a post if its title
  is clearly relevant to the task — otherwise proceed directly to code.

The devlog contains design rationale, migration history, and performance
data written by the engineers who built these systems.

When answering from devlog posts:
- **Always cite the date and title** of the devlog entry used (e.g., "per the
  2025-03-15 *Guard-Free Dynamic Shapes* devlog entry").
- **Always state** that the answer is based on devlog entries.
- **Prompt the user** at the end: ask whether the devlog-based answer is
  sufficient or whether they'd like a code search to verify/supplement.
