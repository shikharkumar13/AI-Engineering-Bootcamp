# Phase 00, Part 1 — Dev Environment & Tools

> **Who this is for:** You have never opened a terminal, never installed a
> programming language, and never used an IDE. That is completely fine — this
> guide assumes nothing.  
> **What you'll have by the end:** A working terminal you're comfortable in,
> Python installed correctly, your first virtual environment, VS Code set up
> and running your first program, and your first project pushed to GitHub.  
> **Time:** 2-3 hours, split across 1-2 days. Don't rush it — get comfortable
> with each piece before moving to the next.

---

## Table of Contents

1. [Why This Part Matters](#1-why-this-part-matters)
2. [Understanding Your Computer's File System](#2-understanding-your-computers-file-system)
3. [The Terminal — Your New Best Friend](#3-the-terminal--your-new-best-friend)
4. [Installing Python](#4-installing-python)
5. [Virtual Environments — Why Every Project Needs Its Own Box](#5-virtual-environments--why-every-project-needs-its-own-box)
6. [Installing VS Code](#6-installing-vs-code)
7. [Git & GitHub — Save Points for Your Code](#7-git--github--save-points-for-your-code)
8. [The Full Workflow — Putting It All Together](#8-the-full-workflow--putting-it-all-together)
9. [Checkpoint Project](#9-checkpoint-project)
10. [Troubleshooting FAQ](#10-troubleshooting-faq)
11. [Key Takeaways](#11-key-takeaways)
12. [What's Next](#12-whats-next)

---

## 1. Why This Part Matters

Every phase in this course — from Phase 01's first LLM API call to Phase 08's
Docker deployment — assumes you can do five things without thinking about them:

1. Open a terminal and type a command
2. Have Python installed and working
3. Keep each project's dependencies separate (a "virtual environment")
4. Write and run code in an editor (VS Code)
5. Save your work and put it on GitHub

None of these are hard individually. The reason they feel overwhelming
together is that nobody usually explains them slowly, one at a time, with the
"why" before the "how." That's exactly what this guide does.

**A note on operating systems:** Instructions below are given for both
**Windows** and **Mac**, clearly labeled. Follow only the section for your
computer. If you're on Linux, the Mac instructions almost always work for you
too (Linux and Mac are both built on similar foundations).

---

## 2. Understanding Your Computer's File System

Before touching a terminal, you need one mental model: **everything on your
computer lives inside folders, and folders live inside other folders.**

You already know this from File Explorer (Windows) or Finder (Mac) — you've
double-clicked into folders to find files. The terminal does the exact same
thing, just with typed commands instead of double-clicks.

### 2.1 Folders, paths, and addresses

A **path** is just the "address" of a file or folder — like a street address,
but for your computer. For example:

```
Windows: C:\Users\Sam\Desktop\my_project\hello.py
Mac:     /Users/sam/Desktop/my_project/hello.py
```

Reading this left to right: "Start at the very top of the computer (`C:\` or
`/`), go into `Users`, then `Sam`, then `Desktop`, then `my_project`, and
there's the file `hello.py`."

Two important terms:

| Term | Meaning |
|---|---|
| **Absolute path** | The full address starting from the very top (`C:\Users\Sam\...` or `/Users/sam/...`) |
| **Relative path** | An address starting from *where you currently are* (e.g., just `my_project/hello.py` if you're already in `Desktop`) |
| **Home folder** | Your personal starting folder — `C:\Users\YourName` on Windows, `/Users/yourname` on Mac. Often written as a shortcut: `~` |

### 2.2 Quick exercise (no terminal yet)

Open File Explorer (Windows) or Finder (Mac) right now and find your
**Desktop** folder. Look at the address bar at the top — that's a path,
exactly like the ones above. This is the same thing the terminal will show
you, just in a different form.

---

## 3. The Terminal — Your New Best Friend

### 3.1 What it actually is

The terminal (also called "command line" or "console") is a text-based way
to tell your computer what to do, instead of clicking icons. You type a
command, press Enter, and your computer does it immediately.

**Why developers use it instead of clicking around:** it's faster once you
know it, it's far more precise, and — most importantly for this course —
almost every tool you'll use (Python, pip, git, Docker, uvicorn) is *run*
through the terminal. There's no way around learning this; the good news is
it only takes about 10 commands to be productive.

### 3.2 Opening your terminal

**Windows:**
1. Press the `Windows` key
2. Type `Windows Terminal` (if you don't have it, type `Command Prompt` instead — it works fine for everything in this guide)
3. Press Enter

**Mac:**
1. Press `Cmd + Space` to open Spotlight Search
2. Type `Terminal`
3. Press Enter

You should now see a mostly-empty window with some text and a blinking
cursor. That text is called the **prompt** — it shows where you currently are.

```
Windows example:  C:\Users\Sam>
Mac example:       sam@MacBook-Pro ~ %
```

Whatever it looks like on your machine, that's where you type.

### 3.3 Your first commands

Type each of these one at a time, pressing Enter after each. Don't worry
about memorizing — you'll use these so often they'll become automatic within
a day.

| What you want to do | Windows | Mac |
|---|---|---|
| See where you currently are | `cd` (with nothing after it) | `pwd` |
| List what's in the current folder | `dir` | `ls` |
| Move into a folder | `cd foldername` | `cd foldername` |
| Move up one level (out of a folder) | `cd ..` | `cd ..` |
| Go straight to your home folder | `cd %USERPROFILE%` | `cd ~` |
| Create a new folder | `mkdir foldername` | `mkdir foldername` |
| Clear the screen (just visual, doesn't delete anything) | `cls` | `clear` |

**Try this right now, step by step:**

```bash
# 1. Go to your home folder
cd ~              # Mac
cd %USERPROFILE%  # Windows

# 2. See what's there
ls    # Mac
dir   # Windows

# 3. Create a folder for this entire course
mkdir ai-engineer-course

# 4. Move into it
cd ai-engineer-course

# 5. Confirm you're in the right place
pwd   # Mac — should print .../ai-engineer-course
cd    # Windows — should print ...\ai-engineer-course
```

If you see your new folder's path printed back at you, **it worked.**
Congratulations — you just gave your computer instructions without clicking
a single icon.

### 3.4 Two productivity tricks (use these immediately)

- **Tab completion:** Start typing a folder or file name and press `Tab`.
  The terminal auto-completes it for you. This alone will save you constant
  typos.
- **Up arrow:** Press the `↑` key to bring back your previous command, so you
  don't have to retype things you just ran.

---

## 4. Installing Python

Python is the programming language every single file in this course is
written in. Let's get it installed correctly — the most common beginner
mistake happens during this exact step, so follow closely.

### 4.1 Windows installation

1. Go to **[python.org/downloads](https://www.python.org/downloads/)**
2. Click the big yellow "Download Python 3.x.x" button
3. Run the downloaded installer
4. **Critical step:** On the very first install screen, there is a checkbox
   at the bottom that says **"Add python.exe to PATH"** — check this box
   before clicking anything else. This is the single most common thing
   beginners miss, and it causes "python is not recognized" errors later.
5. Click "Install Now"
6. Wait for it to finish, then close the installer

### 4.2 Mac installation

1. Go to **[python.org/downloads](https://www.python.org/downloads/)**
2. Click the "Download Python 3.x.x" button (it detects macOS automatically)
3. Open the downloaded `.pkg` file
4. Click through the installer (defaults are fine) until it finishes

> **Why install from python.org even though Mac sometimes has Python
> pre-installed?** Older Macs ship with an outdated, system-reserved version
> of Python that the operating system itself depends on. Modifying or
> relying on it can cause problems. Installing your own copy from python.org
> keeps things clean and gives you a current version.

### 4.3 Verify it worked

**Close your terminal completely and open a brand new one** — this matters,
because the terminal only picks up the Python installation when it starts
fresh.

```bash
# Windows
python --version

# Mac
python3 --version
```

You should see something like `Python 3.12.1` (any 3.9 or higher is fine for
this course). If you see a version number, **you're done with this step.**

> **Why `python` on Windows but `python3` on Mac?** Macs sometimes have an
> old system Python registered as just `python`, so the installer registers
> the new one as `python3` to avoid overwriting it. Windows doesn't have this
> conflict, so `python` works directly. Just remember which one is yours —
> every command in this course that says `python` should be typed as
> `python3` if you're on Mac.

### 4.4 What is pip?

`pip` is Python's **package manager** — it downloads and installs other
people's code (called "packages" or "libraries") so you don't have to write
everything from scratch. Every single `requirements.txt` file in this course
gets installed using pip.

It comes bundled with Python automatically. Verify it:

```bash
# Windows
pip --version

# Mac
pip3 --version
```

You'll use it like this (don't run this yet, just look at the shape of it):

```bash
pip install requests
```

This downloads a package called `requests` from the internet and makes it
available to your Python programs.

---

## 5. Virtual Environments — Why Every Project Needs Its Own Box

### 5.1 The problem this solves

Imagine two projects on your computer:

- **Project A** needs version 1 of a package called `openai`
- **Project B** needs version 2 of that same package

If you install packages "globally" (system-wide), only one version can exist
at a time — installing one breaks the other. This is a real, common problem.

### 5.2 The solution: a sealed toolbox per project

A **virtual environment** is an isolated, self-contained copy of Python and
pip that lives inside one project's folder. Packages installed inside it
only affect that project — completely sealed off from every other project on
your computer.

```
Your computer
├── Project A
│   └── venv/  ← has openai v1.0 installed, only visible to Project A
├── Project B
│   └── venv/  ← has openai v2.0 installed, only visible to Project B
```

This is why every phase folder in this course (`phase01_project/`,
`phase02_project/`, etc.) is meant to get its own virtual environment.

### 5.3 Creating one

Inside your `ai-engineer-course` folder from earlier:

```bash
# Windows
python -m venv venv

# Mac
python3 -m venv venv
```

This creates a new folder called `venv` containing an isolated Python. Run
`ls` (Mac) or `dir` (Windows) and you'll see it sitting there.

### 5.4 Activating it

"Activating" tells your terminal "use the Python inside this `venv` folder,
not the system one, until I say otherwise."

```bash
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Mac
source venv/bin/activate
```

**How to know it worked:** your prompt changes to show `(venv)` at the
start:

```
(venv) C:\Users\Sam\ai-engineer-course>          (Windows)
(venv) sam@MacBook-Pro ai-engineer-course %      (Mac)
```

That `(venv)` prefix means "everything I install right now stays inside this
project." This is the signal you'll look for at the start of every coding
session from Phase 01 onward.

### 5.5 Deactivating

When you're done working on a project:

```bash
deactivate
```

The `(venv)` prefix disappears, confirming you're back to your normal
system-wide terminal.

> **PowerShell users — a common snag:** if `Activate.ps1` gives you an error
> about "running scripts is disabled," see the Troubleshooting section at
> the end of this guide.

---

## 6. Installing VS Code

### 6.1 What is an IDE, and why not just use Notepad?

A plain text editor (Notepad, TextEdit) just stores characters. An **IDE**
(Integrated Development Environment) understands that you're writing code,
and helps you:

- **Syntax highlighting** — colors your code so it's easier to read
- **Autocomplete** — suggests what you're probably typing
- **Integrated terminal** — a terminal built right into the same window, so
  you never have to alt-tab between an editor and a separate terminal app
- **Error detection** — underlines mistakes before you even run the code
- **Extensions** — add-ons for specific languages, including Python

VS Code (Visual Studio Code) is free, lightweight, and the most widely used
editor for exactly this kind of work.

### 6.2 Installing it

1. Go to **[code.visualstudio.com](https://code.visualstudio.com)**
2. Click the big "Download" button — it detects your OS automatically
3. Run the installer
   - **Windows:** accept the defaults; on the "Additional Tasks" screen, it's
     worth checking "Add to PATH" if offered
   - **Mac:** drag the VS Code icon into your Applications folder
4. Launch VS Code

### 6.3 The first-launch tour

When VS Code opens, here's what you're looking at:

```
┌─────┬──────────────────────────────────────────────┐
│     │                                                │
│  E  │              Editor area                       │
│  x  │     (your code appears here when you           │
│  t  │      open a file)                               │
│  e  │                                                │
│  n  │                                                │
│  s  │                                                │
│  i  ├──────────────────────────────────────────────┤
│  o  │           Integrated Terminal                   │
│  n  │   (toggle with Terminal > New Terminal)          │
│  s  │                                                │
└─────┴──────────────────────────────────────────────┘
  ↑
Sidebar icons: Explorer (files), Search, Source Control
(Git), Extensions
```

The vertical strip of icons on the far left is your main navigation:

| Icon | What it does |
|---|---|
| Files/Explorer | Browse the folder you have open |
| Search | Find text across all files in your project |
| Source Control | Git — staging, committing, pushing (Section 7) |
| Extensions | Install add-ons, like the Python extension below |

### 6.4 Installing the Python extension

1. Click the **Extensions** icon in the left sidebar (it looks like four
   squares, one detached)
2. Type `Python` in the search box
3. Find the one published by **Microsoft** (it'll usually be the first
   result, with a blue checkmark)
4. Click **Install**

This gives VS Code the ability to understand Python specifically — error
checking, autocomplete tailored to Python, and a "Run" button for `.py` files.

### 6.5 Opening your project folder

1. Go to **File > Open Folder...**
2. Navigate to and select the `ai-engineer-course` folder you created in
   Section 3.3
3. Click "Select Folder" (Windows) or "Open" (Mac)

VS Code now shows that folder's contents in the Explorer sidebar on the left
— including the `venv` folder you created earlier.

### 6.6 The integrated terminal

Go to **Terminal > New Terminal** in the menu bar (or press `` Ctrl+` ``,
the backtick key, on both Windows and Mac).

A terminal panel opens at the bottom of VS Code. **This is the exact same
terminal you used in Section 3** — just conveniently embedded in your editor
now, automatically opened in your project's folder. From here on, you can do
everything in one window: write code above, run commands below.

Activate your virtual environment here too, exactly as in Section 5.4 — you
should see `(venv)` appear in this terminal as well.

### 6.7 Writing and running your first program

1. In the Explorer sidebar, right-click your folder name and choose
   **New File**
2. Name it `hello.py`
3. Type this single line:

```python
print("Hello, AI Engineer!")
```

4. Run it — there are three equivalent ways:
   - Click the **▷ Run** button in the top-right corner of the editor
   - Right-click anywhere in the file and choose **"Run Python File in Terminal"**
   - Or, simplest: in the integrated terminal, type:
     ```bash
     python hello.py    # Windows
     python3 hello.py   # Mac
     ```

**What you should see:** the terminal panel at the bottom prints:

```
Hello, AI Engineer!
```

If you see that line, **you just wrote and ran your first program.**
Everything from Phase 01 onward is this exact same loop — write code, run
it, see the output — just with more interesting code.

> **A note on autocomplete:** as you type in VS Code, a small popup menu
> will often appear suggesting how to finish what you're typing. This is
> normal and helpful, not an error. Press `Esc` to dismiss it, or `Tab`/`Enter`
> to accept a suggestion.

---

## 7. Git & GitHub — Save Points for Your Code

### 7.1 The core idea

Think of a video game's save system. You can save your progress at any
point, and if something goes wrong later, you can load an earlier save.
**Git does this for code** — except instead of one save slot, you get
unlimited saves, each with a note describing what changed, and you can
compare any two saves to see exactly what's different.

**Git** is the tool that does this tracking, running on your own computer.
**GitHub** is a website that stores a copy of your project online — so you
can access it from anywhere, share it, and (importantly for this course)
show it off as a portfolio to employers.

```
Your computer                    GitHub (the internet)
┌─────────────────┐              ┌─────────────────┐
│  Your project    │   git push   │  Same project,   │
│  + Git history   │ ───────────► │  backed up &     │
│  (local)         │              │  shareable       │
└─────────────────┘              └─────────────────┘
```

### 7.2 Installing Git

**Windows:**
1. Go to **[git-scm.com](https://git-scm.com)**
2. Download and run the installer
3. Accept all the default options (they're sensible for beginners) — just
   keep clicking "Next" through the setup screens

**Mac:**
Git is often already installed. Check first:
```bash
git --version
```
If you see a version number, you're done. If instead you get a prompt to
install "Command Line Developer Tools," click **Install** and wait for it to
finish — that installs Git along with some other useful tools.

### 7.3 One-time setup: tell Git who you are

Every save you make gets labeled with your name and email. Set this once,
ever, on this computer:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

(Use the same email you'll use for your GitHub account in the next step.)

### 7.4 Creating a GitHub account

1. Go to **[github.com](https://github.com)**
2. Click **Sign up**
3. Follow the prompts (username, email, password, email verification)

Pick your username thoughtfully — this becomes part of your portfolio URL
(`github.com/yourusername`), which you'll eventually share with employers.

### 7.5 The core Git workflow, explained before you type anything

Three commands, used in this order, over and over:

| Command | What it does | Analogy |
|---|---|---|
| `git add <file>` | Marks a file as "ready to be saved" | Selecting which items go in your save file |
| `git commit -m "message"` | Actually saves that snapshot, with a description | Pressing "Save Game," with a note |
| `git push` | Uploads your saved snapshots to GitHub | Backing up your save to the cloud |

There's also one command you run **once per project**, at the very start:

| Command | What it does |
|---|---|
| `git init` | Turns a normal folder into a Git-tracked project |

### 7.6 Two ways to do this: terminal vs VS Code's built-in panel

You can run every Git command by typing it in the terminal (good for
understanding what's actually happening), **or** click buttons in VS Code's
Source Control panel (faster day-to-day, and harder to make a typo in).

**This guide teaches the terminal commands first** so you understand the
underlying mechanics — then shows you the VS Code shortcut, which is what
you'll likely use most days going forward.

### 7.7 Guided walkthrough — terminal method

Inside your `ai-engineer-course` folder (with `hello.py` already created
from Section 6.7):

```bash
# 1. Turn this folder into a Git project (only ever done once per project)
git init

# 2. See what Git notices has changed
git status
# → shows hello.py as "untracked" (Git sees it exists but isn't saving it yet)

# 3. Stage the file (mark it ready to save)
git add hello.py
# (or: git add .   ← the dot means "everything in this folder")

# 4. Commit — actually save this snapshot
git commit -m "My first commit - hello world"
```

You should see a confirmation message showing 1 file changed.

**Now create the GitHub side:**

1. Go to **github.com** and click the **+** icon (top right) → **New repository**
2. Name it `ai-engineer-course`
3. Leave it **Public** (so it's visible on your portfolio) or **Private**
   (your choice)
4. **Important:** do NOT check "Add a README" — leave the repo completely
   empty, since you already have content locally
5. Click **Create repository**
6. GitHub shows you a page with some commands — copy the URL that looks like
   `https://github.com/yourusername/ai-engineer-course.git`

**Back in your terminal:**

```bash
# 5. Connect your local project to the GitHub repo you just made
git remote add origin https://github.com/yourusername/ai-engineer-course.git

# 6. Name your main branch (a one-time naming convention step)
git branch -M main

# 7. Push — upload your commit to GitHub
git push -u origin main
```

**The first time you push, GitHub will ask you to authenticate.** GitHub no
longer accepts your regular password from the terminal — see the
Troubleshooting section below for the two easiest ways to handle this.

Once it succeeds, refresh your GitHub repository page in the browser — you
should see `hello.py` sitting there. **That file is now backed up online and
shareable with a link.**

### 7.8 The easier day-to-day method: VS Code's Source Control panel

Once a repo is connected (as above), you rarely need to type `git` commands
again. In VS Code:

1. Click the **Source Control** icon in the left sidebar (the branching icon)
2. Any changed files appear in a list, with a `+` button next to each —
   clicking it **stages** the file (same as `git add`)
3. Type a short message in the text box at the top ("describe your change")
4. Click the **✓ Commit** button (same as `git commit`)
5. Click **Sync Changes** or **Push** (same as `git push`)

For a brand-new project that isn't connected to GitHub yet, VS Code also
offers a one-click **"Publish to GitHub"** button right in this panel — it
handles `init`, `remote add`, and the first `push` all at once, and walks
you through signing in to GitHub if you haven't already. This is the
fastest path for future projects once you understand what's happening
underneath (which is exactly what Section 7.7 just taught you).

### 7.9 One more essential command: clone

Later in this course (and in your career), you'll often want to download an
existing project from GitHub onto a new computer. That command is:

```bash
git clone https://github.com/someone/some-project.git
```

This downloads the entire project, with its full history, into a new folder
on your computer.

---

## 8. The Full Workflow — Putting It All Together

This is the loop you'll repeat for literally every phase in this course,
start to finish:

```
1. Open VS Code, open your project folder
        │
2. Open the integrated terminal (Ctrl+`)
        │
3. Activate your virtual environment  →  (venv) appears in the prompt
        │
4. Write or edit code in the editor
        │
5. Run it in the terminal  →  python somefile.py
        │
6. See output, fix anything broken, repeat steps 4-5 as needed
        │
7. Stage + commit + push your changes (terminal or Source Control panel)
        │
   (back to step 4 for your next change)
```

Every single project in Phases 01-08 follows exactly this loop. The tools
change (sometimes Docker, sometimes a Streamlit app), but this core cycle —
edit, run, see output, save, share — never does.

---

## 9. Checkpoint Project

Time to prove all of this works together, end to end. Don't skip this — it's
the single best confidence builder before Phase 02.

**Goal:** Inside your `ai-engineer-course` folder, create a new subfolder
called `phase00-checkpoint`, set it up with its own virtual environment,
write a short personalized Python script, run it successfully, and push the
whole thing to a **new** GitHub repository.

Step by step:

```bash
# 1. Make sure you're in your course folder
cd ai-engineer-course    # adjust path if needed

# 2. Create and enter a new subfolder for this checkpoint
mkdir phase00-checkpoint
cd phase00-checkpoint

# 3. Create and activate a virtual environment for this specific project
python -m venv venv          # Windows: python, Mac: python3
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac
# Confirm you see (venv) in your prompt before continuing
```

Now open this `phase00-checkpoint` folder in VS Code (**File > Open Folder**),
and create a new file called `setup_check.py` with this content (type it
yourself — typing builds muscle memory faster than copy-pasting):

```python
import sys

name = "Your Name"  # change this to your actual name

print(f"Hello, {name}!")
print(f"You're running Python {sys.version_info.major}.{sys.version_info.minor}")
print("If you can see this message, your environment is working.")
```

Run it (`python setup_check.py` or the ▷ button). You should see your
personalized message with your real Python version number.

Finally, save and publish this to GitHub:

```bash
git init
git add .
git commit -m "Phase 00 checkpoint - environment working"
```

Create a new, empty repository on GitHub called `phase00-checkpoint`, then:

```bash
git remote add origin https://github.com/yourusername/phase00-checkpoint.git
git branch -M main
git push -u origin main
```

**You're done when:** you can open `github.com/yourusername/phase00-checkpoint`
in a browser and see `setup_check.py` sitting there.

---

## 10. Troubleshooting FAQ

**"python is not recognized as an internal or external command" (Windows)**
Python wasn't added to PATH during install. Reinstall Python from
python.org and make sure to check **"Add python.exe to PATH"** on the first
install screen. Alternatively, search "Edit environment variables for your
account" in the Start menu and manually add your Python install folder.

**"command not found: python" (Mac)**
Use `python3` instead of `python` — this is expected, see Section 4.3.

**PowerShell says "running scripts is disabled on this system" when activating a venv**
Open PowerShell **as Administrator** and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then type `Y` to confirm, close that window, and try activating again in a
normal PowerShell window.

**`git push` asks for a password and rejects it**
GitHub stopped accepting account passwords for terminal authentication.
Two fixes, easiest first:
1. **Use VS Code's Source Control panel** (Section 7.8) and click "Publish
   to GitHub" — it handles authentication via your browser automatically.
2. **Create a Personal Access Token:** on GitHub, go to **Settings >
   Developer settings > Personal access tokens > Tokens (classic) >
   Generate new token**, check the `repo` scope, generate it, copy the long
   token string, and paste it in place of your password when the terminal
   asks (you won't see it appear as you paste — that's normal, paste anyway
   and press Enter).

**VS Code's integrated terminal doesn't show `(venv)` even after activating**
Sometimes VS Code's default terminal profile differs from your system
terminal. Click the dropdown arrow next to the `+` in the terminal panel and
try switching between "Command Prompt" and "PowerShell" (Windows), or just
close and reopen the terminal panel after activating.

**VS Code shows red squiggly lines under code that looks correct, or autocomplete seems "off"**
VS Code might be pointing at the wrong Python interpreter. Press
`Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac) to open the Command Palette,
type "Python: Select Interpreter," and choose the one inside your project's
`venv` folder.

**I closed my terminal and now `(venv)` is gone**
That's expected — virtual environments must be re-activated every time you
open a new terminal session. Just re-run the activate command from Section
5.4. This will happen constantly and is completely normal; soon it'll be
automatic muscle memory.

---

## 11. Key Takeaways

1. **The terminal is just a typed alternative to clicking.** A handful of
   commands (`cd`, `ls`/`dir`, `mkdir`, `pwd`) cover almost everything you'll
   need.

2. **Always check "Add to PATH" when installing Python on Windows** — this
   single checkbox prevents the most common beginner error.

3. **A virtual environment is a sealed toolbox per project**, preventing
   package version conflicts between different projects. Activate it every
   time you start working; you'll see `(venv)` in your prompt as confirmation.

4. **VS Code is a text editor that understands code** — syntax highlighting,
   autocomplete, an integrated terminal, and (with the Python extension)
   one-click running of `.py` files.

5. **Git tracks save points; GitHub hosts them online.** The core loop is
   `add` (stage) → `commit` (save) → `push` (upload). VS Code's Source
   Control panel does all three with clicks instead of typed commands once
   you understand what they do.

6. **The full development loop — edit, run, see output, save, share — is
   the same loop you'll repeat in every phase of this course**, from Phase
   01's first API call through Phase 08's Docker deployment.

---

## 12. What's Next

**Part 2 — Python Fundamentals I** starts actually teaching you to write
Python: variables, data types, strings, lists, dictionaries, control flow,
and functions. By the end of Part 2, you'll be able to build a working
command-line program from scratch — the exact building blocks every file in
Phases 01-08 is made of.

Say **"Start Part 2"** when you're ready, or revisit anything in this guide
first — there's no rush, and getting comfortable with your terminal now will
make everything else in this course noticeably smoother.
