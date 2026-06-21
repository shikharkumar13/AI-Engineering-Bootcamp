# Phase 00, Part 1 — Cheat Sheet

> Keep this open in a second window while you practice. You don't need to
> memorize any of this — you'll absorb it naturally within a few days of use.

---

## Terminal Navigation

| Task | Windows | Mac |
|---|---|---|
| Where am I right now? | `cd` | `pwd` |
| List files in this folder | `dir` | `ls` |
| Move into a folder | `cd foldername` | `cd foldername` |
| Move up one level | `cd ..` | `cd ..` |
| Go to home folder | `cd %USERPROFILE%` | `cd ~` |
| Create a new folder | `mkdir foldername` | `mkdir foldername` |
| Clear the screen | `cls` | `clear` |
| Run a Python file | `python file.py` | `python3 file.py` |

**Productivity tips:**
- Press `Tab` to auto-complete file/folder names as you type
- Press `↑` (up arrow) to bring back your last command
- `Ctrl+C` cancels whatever is currently running in the terminal

---

## Python & pip

| Task | Windows | Mac |
|---|---|---|
| Check Python is installed | `python --version` | `python3 --version` |
| Check pip is installed | `pip --version` | `pip3 --version` |
| Install a package | `pip install packagename` | `pip3 install packagename` |
| Install from a requirements file | `pip install -r requirements.txt` | `pip3 install -r requirements.txt` |
| List installed packages | `pip list` | `pip3 list` |

---

## Virtual Environments

| Task | Windows | Mac |
|---|---|---|
| Create a venv | `python -m venv venv` | `python3 -m venv venv` |
| Activate (Command Prompt) | `venv\Scripts\activate` | — |
| Activate (PowerShell) | `venv\Scripts\Activate.ps1` | — |
| Activate | — | `source venv/bin/activate` |
| Deactivate | `deactivate` | `deactivate` |
| **How to tell it's active** | `(venv)` appears at the start of your prompt | same |

---

## Git — One-Time Setup (do this once per computer)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Git — Per-Project Setup (do this once per project)

```bash
git init
git remote add origin https://github.com/yourusername/reponame.git
git branch -M main
```

## Git — The Daily Loop (do this every time you save progress)

```bash
git status                       # see what's changed
git add .                        # stage everything changed (or: git add specific_file.py)
git commit -m "what you did"     # save a snapshot with a description
git push                         # upload to GitHub
```

## Git — Other Useful Commands

| Command | What it does |
|---|---|
| `git status` | Shows what's changed since your last commit |
| `git log` | Shows your commit history |
| `git clone <url>` | Downloads an existing repo to your computer |
| `git diff` | Shows the exact lines that changed (before staging) |

---

## The Daily Workflow Checklist

Every time you sit down to work on this course:

```
[ ] Open VS Code, open the project folder
[ ] Open the integrated terminal (Ctrl+`)
[ ] Activate the virtual environment — check for (venv) in the prompt
[ ] Write / edit code
[ ] Run it, check the output
[ ] When a piece of work is done: git add . → git commit -m "..." → git push
```

---

## Common Gotchas (see the full Troubleshooting FAQ in the main guide for fixes)

- Windows: forgot to check "Add python.exe to PATH" during install
- Mac: use `python3`, not `python`
- Closed and reopened the terminal: you must reactivate the venv every time
- `git push` rejecting your password: GitHub needs a Personal Access Token
  or the VS Code "Publish to GitHub" flow now, not your account password
