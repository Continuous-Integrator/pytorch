# ghstack Review Skill

## Purpose
Standardized workflow for reviewing a ghstack PR stack before pushing. Ensures each PR is clean, self-contained, properly ordered, and passes CI checks.

## When to Use
- Before `ghstack submit` to push updates
- After making changes across multiple PRs in a stack
- When preparing a stack for code review

## Review Checklist (per PR, bottom to top)

### 1. Stack Integrity
- [ ] Verify stack order: `sl log -r 'stack()' --template '{node|short} {desc|firstline}\n'`
- [ ] HEAD is at the tip (topmost PR): `sl log -r '.' --template '{node|short} {desc|firstline}\n'`
- [ ] Each commit's ghstack-source-id matches the expected PR
- [ ] No interleaved unrelated commits in the stack

### 2. Per-PR File Isolation
- [ ] Check files changed per commit: `sl diff -c <hash> --stat`
- [ ] **No submodule contamination** (third_party/ changes that don't belong)
- [ ] **No unrelated data files** (benchmark CSVs, images, scripts from other projects)
- [ ] **No junk files** (temp files, scratch files named "1", binary images)
- [ ] Each file change belongs to the correct PR in the stack
- [ ] Changes don't "leak" into adjacent PRs

### 3. Commit Message Quality
- [ ] Title matches the actual change (not stale from previous iterations)
- [ ] Description accurately describes the CURRENT implementation (not an old approach that was removed)
- [ ] No references to removed code (e.g., "MutationLayoutSHOULDREMOVE")
- [ ] Stack overview section is up-to-date with current PR numbers
- [ ] Test Plan section references the actual test commands

### 4. Code Quality
- [ ] No Claude-generated residual comments (incomplete context, PR-number references like "# TODO: PR#5")
- [ ] No dead code from previous iterations
- [ ] Comments explain WHY, not WHAT
- [ ] No string-based type discrimination where enums/dataclasses would be safer
- [ ] Error handling is complete (no bare `except:`, no silently swallowed exceptions)
- [ ] Assertions have clear error messages

### 5. Design Review
- [ ] Is there a simpler/more elegant approach?
- [ ] Are there hardcoded values that should be configurable?
- [ ] Manual field copying (e.g., dataclass constructor) vs `dataclasses.replace()`?
- [ ] Module-level mutable state is thread-safe?
- [ ] New public API surface is intentional and documented?

### 6. Test Quality
- [ ] Each PR has tests for its specific functionality
- [ ] Tests verify codegen output (FileCheck/assertIn for expected patterns)
- [ ] Tests verify runtime correctness (compiled vs eager comparison)
- [ ] Tests are in the correct PR (not testing PR5 functionality in PR2's commit)
- [ ] Test names clearly indicate what they test
- [ ] No over-reliance on fragile codegen string matching

### 7. Cross-PR Consistency
- [ ] TODO comments reference issue numbers, not PR numbers (PR numbers change)
- [ ] Later PRs don't duplicate functionality from earlier PRs
- [ ] Each PR's commit message links to the correct GitHub PR number
- [ ] The stack builds incrementally (each PR is independently reviewable)

## Workflow (bottom to top)

For each PR from the base (PR1) to the tip (PR5):

```bash
# 1. Go to the PR's commit
sl goto <hash>

# 2. Check the diff
sl diff -c . --stat
sl diff -c . -- torch/ test/  # source files only

# 3. Run tests
conda activate pytorch-3.12
torchrun --nproc_per_node=2 -m pytest test/distributed/test_symmetric_memory.py -k "LoweringTest" -xvs

# 4. Check codegen output matches PR description
# (manually verify generated code patterns)

# 5. Run linter
run_lintrunner -m origin/main

# 6. Auto-fix lint issues
run_lintrunner -a

# 7. Amend the commit if changes were made
sl amend

# 8. Move to next PR
sl next
```

## Common Issues

### Submodule Contamination
```bash
# Check for submodule changes in a commit
sl diff -c <hash> --stat | grep 'third_party/'
# Fix: interactive amend to exclude submodule changes
sl uncommit --keep <submodule_file>
```

### Wrong PR for Changes
```bash
# If a change belongs in a different PR:
# 1. Note the file and change
# 2. Revert it from current commit
# 3. Move to correct commit and add it there
sl goto <correct_hash>
# make the change
sl amend
sl goto <tip>  # return to tip
sl rebase -d <base>  # rebase stack
```

### Stale Commit Messages
```bash
# Update commit message
sl metaedit -m "new message"
```
