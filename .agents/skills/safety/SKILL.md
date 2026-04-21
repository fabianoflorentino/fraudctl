---
name: safety
description: Prevents destructive operations by requiring confirmation and verification before executing risky commands.
origin: fraudctl-project
---

# Safety Guidelines

This skill prevents data loss by requiring verification before destructive operations.

## When to Activate

- Before executing any `rm -rf` command
- Before executing any command that deletes files or directories
- Before executing any command that modifies git history (e.g., `git reset --hard`)
- Before executing any command that could lose unsaved work

## Critical Rules

### NEVER execute without verification:

```bash
# BAD - This deletes without warning
rm -rf some_directory

# GOOD - This asks for confirmation first
# Analyze what will be deleted first, then ask user
ls -la some_directory  # Verify contents
# Then ask: "Are you sure you want to delete some_directory?"
```

### Before any destructive command:

1. **LIST** - List what's being affected
2. **VERIFY** - Check the path is correct
3. **ASK** - Ask for user confirmation if uncertain

```bash
# Step 1: List what will be deleted
ls -la path/to/directory

# Step 2: Verify contents
# - Check for important files (git history, node_modules, etc.)
# - Check for backups
# - Check for configuration files

# Step 3: Ask for confirmation
# If uncertain, ask the user: "This will permanently delete [files]. Continue?"
```

### Safe Patterns

```bash
# SAFE - Use -i flag for interactive deletion
rm -ri directory/

# SAFE - Use find with -ls first to preview
find directory/ -type f -ls

# SAFE - Move to trash instead of delete
mv directory/ ~/.trash/

# SAFE - Use git to check what's tracked
git ls-files directory/
```

### Commands that REQUIRE confirmation:

| Command | Risk | Action |
|---------|------|--------|
| `rm -rf` | Permanent deletion | List + Ask |
| `git reset --hard` | Lose uncommitted changes | Show status first |
| `git clean -fd` | Delete untracked files | List first |
| `dd if= of=` | Overwrite files | Verify twice |
| `> file` | Truncate file | Ask before |

## Before Deleting Directories

Always check:

1. Is it in `.gitignore`?
2. Does it contain uncommitted changes?
3. Is it a dependency directory (node_modules, vendor, etc.)?
4. Is it the only copy of important data?

```bash
# Check git status first
git status directory/

# Check if tracked
git ls-files directory/

# Check size
du -sh directory/
```

## Quick Reference

**Rule:** When in doubt, ASK. Never delete something you can't recover.

**Before rm -rf:**
1. `ls -la` to see what's there
2. Check if tracked by git
3. Ask user for confirmation

**After accidental deletion:**
1. Don't panic
2. `git status` may show recoverable files
3. File system tools may recover recent deletions
