# Phase 00, Part 4 — The Command Line, APIs & the Web

> **Who this is for:** You've never thought about what actually happens when
> an app "talks to the internet." This guide assumes nothing — by the end,
> you'll understand exactly what Phase 01's `client.chat.completions.create()`
> is doing under the hood.  
> **What you'll have by the end:** A working understanding of HTTP, JSON,
> what an API actually is, why secrets live in `.env` files, and what
> `localhost:8000` means — plus hands-on practice making real requests from
> both the terminal and Python.  
> **Time:** 2-3 hours.

---

## Table of Contents

1. [Why This Part Matters](#1-why-this-part-matters)
2. [The Client-Server Model](#2-the-client-server-model)
3. [HTTP — The Language of the Web](#3-http--the-language-of-the-web)
4. [JSON — The Format Almost Every API Speaks](#4-json--the-format-almost-every-api-speaks)
5. [What Is an API, Really?](#5-what-is-an-api-really)
6. [API Keys & .env Files](#6-api-keys--env-files)
7. [localhost & Ports](#7-localhost--ports)
8. [Hands-On Walkthrough](#8-hands-on-walkthrough)
9. [Checkpoint Project](#9-checkpoint-project)
10. [Troubleshooting FAQ](#10-troubleshooting-faq)
11. [Key Takeaways](#11-key-takeaways)
12. [What's Next](#12-whats-next)

---

## 1. Why This Part Matters

Phase 01 opens with code like this:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

If you don't know what an API is, this line is just magic — type some words,
get a response back, no idea what happened in between. After this guide,
you'll know **exactly** what's happening: your computer is sending a
structured text message over the internet to OpenAI's servers, formatted in
a specific way, with a secret key proving it's really you, and getting a
structured text message back.

Five things to understand, in order:

1. **HTTP** — the rules computers follow to talk to each other over the web
2. **JSON** — the format the messages are written in
3. **APIs** — what it means for a company to let you "call" their service
4. **API keys & `.env` files** — how you prove who you are, safely
5. **localhost & ports** — what's happening when YOUR computer is the server (Phase 08)

---

## 2. The Client-Server Model

Almost everything you do online follows the same basic pattern: two
computers talking to each other.

```
┌──────────────┐                              ┌──────────────┐
│    CLIENT    │  ──────  sends a request ───► │    SERVER     │
│ (your laptop,│                               │ (a company's  │
│  your phone) │  ◄──── sends back a response ─│  computer)    │
└──────────────┘                              └──────────────┘
```

- The **client** is whoever is *asking* for something — your browser, your
  phone's app, or in this course, your Python script.
- The **server** is whoever is *listening* for requests and responding —
  OpenAI's computers, a weather company's computers, or (in Phase 08) your
  *own* computer, pretending to be a server.

When you type a web address into your browser, your browser is the client,
and somewhere in the world a server receives your request and sends back
the webpage. When your Python script calls an LLM API, your script is the
client, and OpenAI's servers are doing the responding.

**This is the entire shape of every example in this course.** Phases 01-07
are all about being the *client*, asking someone else's server for things.
Phase 08 flips it — you build the *server* yourself.

---

## 3. HTTP — The Language of the Web

### 3.1 What HTTP is

**HTTP** (HyperText Transfer Protocol) is simply an agreed-upon set of rules
for how a client's request and a server's response should be formatted, so
that any client can talk to any server, regardless of who built either one.

Think of it like a phone call's social convention: you say "hello," they
say "hello" back, you state your business, they respond, you both say
"goodbye." HTTP is that same kind of agreed structure, but for computers
sending text to each other.

Every single time you've ever loaded a webpage, your browser made an HTTP
request and got an HTTP response back. Same thing happens when your Python
script calls an API.

### 3.2 HTTP methods — what kind of request is this?

Every HTTP request declares a **method** — essentially, what *kind* of
action you're asking for. You'll mostly see two in this course:

| Method | Meaning | Example use |
|---|---|---|
| `GET` | "Give me some information" | Loading a webpage, fetching weather data |
| `POST` | "Here's some data, do something with it" | Sending a chat message to an LLM, submitting a form |

Two others exist (`PUT` for updating something, `DELETE` for removing
something) but you'll mostly encounter `GET` and `POST` in this course.

**Why this matters for Phase 01:** every single LLM API call you'll make is
a `POST` request — you're sending data (your prompt) for the server to
*do something with* (generate a response), not just asking it to hand you
something that already exists.

### 3.3 Status codes — what happened?

Every HTTP response comes back with a 3-digit **status code** telling you,
at a glance, what happened. You don't need to memorize all of them — just
recognize the ranges:

| Range | Meaning | You'll see this when... |
|---|---|---|
| `200-299` | Success | The request worked |
| `300-399` | Redirect | "Go look somewhere else instead" |
| `400-499` | **Client error** — you did something wrong | Bad request, missing auth, not found |
| `500-599` | **Server error** — they did something wrong | Their server is overloaded or broken |

The specific codes you'll run into constantly in this course:

| Code | Name | Meaning |
|---|---|---|
| `200` | OK | Everything worked |
| `400` | Bad Request | Your request was malformed |
| `401` | Unauthorized | Missing or invalid API key |
| `404` | Not Found | The thing you asked for doesn't exist |
| `429` | Too Many Requests | You're being rate-limited |
| `500` | Internal Server Error | Something broke on their end |
| `503` | Service Unavailable | The server is temporarily overloaded |

> **Looking ahead:** Phase 01's error-handling section and Phase 08's
> FastAPI backend are *entirely* built around catching and responding to
> these exact codes. `429` is why Phase 01 teaches retry logic. `401` is
> why Phase 08 teaches API key authentication.

### 3.4 Headers — information riding alongside the request

A **header** is a small piece of metadata attached to a request or
response — information *about* the request, separate from its actual
content (the "body"). Think of headers like the information on a shipping
label: sender, recipient, what's inside, special handling instructions —
all separate from the actual package contents.

Two headers you'll see constantly:

```
Content-Type: application/json
Authorization: Bearer sk-abc123...
```

- `Content-Type` tells the server "the data I'm sending you is formatted as
  JSON" (more on JSON in a moment)
- `Authorization` is how you prove who you are — this is exactly where your
  API key goes (Section 6)

---

## 4. JSON — The Format Almost Every API Speaks

### 4.1 What JSON looks like

**JSON** (JavaScript Object Notation) is a text format for representing
structured data — a universally agreed-upon way to write down "here's a
collection of named values" so that any programming language can read it.

```json
{
  "name": "Claude",
  "is_helpful": true,
  "version": 4.6,
  "skills": ["writing", "coding", "reasoning"],
  "creator": {
    "company": "Anthropic",
    "founded": 2021
  }
}
```

### 4.2 JSON's building blocks

| Syntax | Meaning |
|---|---|
| `{ }` | An **object** — a set of key-value pairs (like a labeled box of named items) |
| `[ ]` | An **array** — an ordered list of values |
| `"key": value` | A named value inside an object |
| `"a string"` | Text, always in double quotes |
| `42` or `4.6` | A number |
| `true` / `false` | A boolean |
| `null` | "No value" |

Objects can contain arrays, arrays can contain objects, and objects can
contain other objects — nested as deeply as needed. That's exactly what
you see above: `"creator"` is an object nested inside the outer object.

### 4.3 JSON vs a Python dictionary

If this looks familiar, it should — JSON's `{ }` objects map almost exactly
onto Python's dictionaries, and JSON's `[ ]` arrays map onto Python's lists.

```python
# This Python dictionary...
data = {
    "name": "Claude",
    "is_helpful": True,
    "skills": ["writing", "coding"]
}

# ...is structurally identical to this JSON:
# {
#   "name": "Claude",
#   "is_helpful": true,
#   "skills": ["writing", "coding"]
# }
```

The only differences: Python's `True`/`False`/`None` become JSON's
lowercase `true`/`false`/`null`. Everything else looks almost the same.

**This is why every API response in this course gets handled the exact same
way:** the server sends back JSON text, and Python converts it straight
into a dictionary you can work with — you'll see this as `response.json()`
constantly throughout the course.

---

## 5. What Is an API, Really?

### 5.1 The plain-English definition

An **API** (Application Programming Interface) is a documented, agreed-upon
set of requests that a service promises to respond to in a predictable way.

**Restaurant analogy:** You don't walk into a restaurant's kitchen and cook
your own food. You look at a menu (the documentation), tell the waiter what
you want using the menu's exact item names (making a request), and the
kitchen (the server) prepares it and the waiter brings it back (the
response). You never need to know *how* the kitchen works internally — just
what's on the menu and how to ask for it.

An API is that menu, formalized: "Send a `POST` request to this exact
address, with your data formatted exactly like this, and I'll send back a
response formatted exactly like that."

### 5.2 What "calling an API" really means

When Phase 01 says "call the OpenAI API," here's the literal sequence:

```
1. Your Python script builds a JSON object:
   {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}

2. It sends this as the BODY of a POST request to:
   https://api.openai.com/v1/chat/completions

3. It attaches a header proving who you are:
   Authorization: Bearer sk-your-secret-key

4. OpenAI's server receives this, runs the actual AI model, and sends back
   a JSON response:
   {"choices": [{"message": {"content": "Hello! How can I help?"}}], ...}

5. Your Python script reads that JSON and pulls out the text you want
```

The `openai` Python library you used starting in Phase 01 is just a
convenience wrapper around exactly this process — it builds the JSON, sends
the HTTP request, attaches your key, and parses the JSON response, so you
don't have to write all of that by hand every time. **But it is not doing
anything magical** — it's just HTTP + JSON, the same two things you just
learned.

### 5.3 REST APIs, briefly

Most modern APIs (including OpenAI's, and the FastAPI backend you'll build
in Phase 08) follow a style called **REST**. You don't need to memorize a
formal definition — just recognize the pattern: different URLs represent
different *resources* or *actions*, and you use HTTP methods (`GET`,
`POST`, etc.) to interact with them.

```
GET    /users          → "give me a list of users"
GET    /users/42        → "give me user number 42"
POST   /users            → "create a new user with this data"
POST   /chat/completions  → "generate a chat response from this data"
```

That's it — that's most of what "REST API" means in practice.

---

## 6. API Keys & .env Files

### 6.1 What an API key actually is

An **API key** is a long, secret string of characters that identifies *you*
to a service, the same way a password does. When you send it in the
`Authorization` header, the server checks: "does this key belong to a real
account, and does that account have permission/credit to do this?"

```
Authorization: Bearer sk-proj-aBcD1234...
```

If the key is missing or wrong, you get back a `401 Unauthorized` — exactly
the status code from Section 3.3.

### 6.2 Why you must never hardcode a key in your code

```python
# NEVER DO THIS:
client = OpenAI(api_key="sk-proj-aBcD1234realSecretKeyHere")
```

This looks harmless, but here's exactly what goes wrong: the moment you
`git push` this file to a public GitHub repository (Part 1, Section 7),
your secret key is now sitting in plain text on the internet, in your
commit history, forever — even if you delete it in a later commit, it's
still recoverable from history. Automated bots constantly scan public
GitHub repositories specifically looking for leaked API keys, and will
start using yours (and racking up charges on your account) within minutes
of it appearing.

### 6.3 Environment variables — the fix

An **environment variable** is a named value that lives *outside* your
code, in the operating system or terminal session, that your program can
read at runtime. Your code asks "what is the value of OPENAI_API_KEY?"
without that value ever being written down inside the code file itself.

### 6.4 `.env` files + python-dotenv

In practice, you don't set environment variables by hand every time — you
write them once into a file named `.env`:

```bash
# .env
OPENAI_API_KEY=sk-proj-aBcD1234realSecretKeyHere
```

Then, in your Python code:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads the .env file and loads its contents

api_key = os.getenv("OPENAI_API_KEY")  # now your code has the value,
                                         # but the value itself is never
                                         # typed anywhere in this file
```

This is the **exact pattern** Phase 01 introduces in its very first setup
step — now you know precisely why it works this way.

### 6.5 Keeping `.env` out of GitHub — `.gitignore`

A `.env` file must **never** be committed to Git. The mechanism that
prevents this is a file called `.gitignore` — a plain text list of
filenames Git should always skip, even if you run `git add .`:

```bash
# .gitignore
.env
__pycache__/
```

Create this `.gitignore` file once, at the start of every project, *before*
you ever create your `.env` file. This is exactly what every phase project
in this course ships with — go back and look at any `requirements.txt`'s
neighboring files and you'll find a `.gitignore` doing exactly this job.

```
Your project folder
├── .env            ← your real secrets — NEVER committed (blocked by .gitignore)
├── .env.example    ← a TEMPLATE showing what keys are needed, safe to commit
├── .gitignore      ← tells Git to ignore .env
└── main.py         ← your code, reads secrets via os.getenv(), never hardcodes them
```

> **Why `.env.example` exists:** it's a safe, shareable file showing
> *which* environment variables a project needs, without containing any
> real secret values — e.g. `OPENAI_API_KEY=` with nothing after the equals
> sign. Anyone cloning your project from GitHub knows exactly what to fill
> in.

---

## 7. localhost & Ports

This section sets you up for Phase 08, where for the first time, **your own
computer becomes the server**, not just the client.

### 7.1 IP addresses, briefly

Every computer connected to a network has an **IP address** — a numeric
address that identifies it, the same way a street address identifies a
building. When your browser requests a webpage, it's really sending a
request to a specific IP address somewhere in the world.

### 7.2 localhost — talking to yourself

`localhost` is a special, reserved address that always means **"this same
computer, right here"** — no matter which physical computer you're on, on
that computer, `localhost` always refers to itself. Its actual numeric form
is `127.0.0.1`, but `localhost` is the friendly name for it.

So when you run a server on your own laptop (which is exactly what Phase 08
does), and then open a browser and go to `http://localhost:8000`, you are
making a request from your laptop, to your laptop, without ever touching
the actual internet. This is invaluable for development — you can build and
test a server entirely on your own machine before anyone else ever sees it.

### 7.3 Ports — which "door" on that computer?

A single computer can run many different programs simultaneously, each
capable of receiving network requests. A **port** is a number that
identifies *which specific program* on that computer a request is meant
for.

**Apartment building analogy:** the IP address (or `localhost`) is the
building's street address. The port number is the specific apartment
number. Mail addressed to "123 Main St, Apt 8000" only reaches whoever
lives in apartment 8000 — other apartments in the same building don't see
it.

```
http://localhost:8000
        └─────┬─────┘ └┬─┘
          this computer  apartment/port 8000
```

Common ports you'll encounter in this course:

| Port | Commonly used by |
|---|---|
| `80` | Standard, unencrypted web traffic (HTTP's default — usually invisible in URLs) |
| `443` | Standard, encrypted web traffic (HTTPS's default — also usually invisible) |
| `8000` | Phase 08's FastAPI backend (`uvicorn main:app --port 8000`) |
| `8501` | Phase 04/08's Streamlit frontend |
| `5000` / `3000` | Other common local development servers you'll see in the wild |

When Phase 08 has you run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

it's telling your computer: "start listening for requests on port 8000."
When you then visit `http://localhost:8000/docs` in a browser, you're
sending a request to your own computer, specifically to whatever program is
listening on port 8000 — which is the FastAPI server you just started.

---

## 8. Hands-On Walkthrough

Time to actually do this. We'll use a service called **httpbin.org** — a
free public website built *specifically* for learning and testing HTTP. It
doesn't do anything useful on its own; its entire purpose is to honestly
echo back exactly what you sent it, which makes it perfect for seeing HTTP
concepts in action.

### 8.1 Setup

Open your terminal (Part 1, Section 3) and make sure you're in your course
folder with your virtual environment activated — you should see `(venv)` in
your prompt (Part 1, Section 5.4).

```bash
pip install requests python-dotenv
```

This installs two packages: `requests` (for making HTTP calls from Python)
and `python-dotenv` (for reading `.env` files, Section 6.4).

### 8.2 Your first GET request — with curl

`curl` is a command-line tool for making HTTP requests directly from your
terminal, without writing any code. It comes pre-installed on Mac and
modern Windows.

```bash
curl https://httpbin.org/get
```

You should see a JSON response printed directly in your terminal, something
like:

```json
{
  "args": {},
  "headers": {
    "Accept": "*/*",
    "Host": "httpbin.org",
    "User-Agent": "curl/8.1.2"
  },
  "origin": "203.0.113.42",
  "url": "https://httpbin.org/get"
}
```

This is httpbin honestly telling you exactly what it received: your
headers, your origin (IP address), and the URL you hit. **You just made a
real HTTP `GET` request and got back a real JSON response** — the exact
same fundamental action as Phase 01's LLM API calls, just to a simpler
service.

### 8.3 Seeing the status code

By default, `curl` hides the status code (Section 3.3). Add `-i` to see the
full response, including headers and the status line:

```bash
curl -i https://httpbin.org/get
```

The very first line of the output will be something like:

```
HTTP/2 200
```

That `200` is the status code — confirmation the request succeeded.

### 8.4 Triggering an error on purpose

httpbin has a special endpoint that lets you request *any* status code you
want, specifically for practicing how to handle errors:

```bash
curl -i https://httpbin.org/status/404
```

You'll see `HTTP/2 404` at the top — you just deliberately triggered the
exact kind of error Phase 01 and Phase 08 teach you to catch and handle
gracefully. Try `https://httpbin.org/status/500` too, and notice the
difference between a client error (`404`, your fault) and a server error
(`500`, their fault) from Section 3.3.

### 8.5 Sending an API key as a header

httpbin also has an endpoint that requires authentication, specifically for
practicing the `Authorization` header pattern from Section 6:

```bash
curl -H "Authorization: Bearer my-test-token-123" https://httpbin.org/bearer
```

The `-H` flag adds a custom header to your request — here, the exact same
`Authorization: Bearer ...` pattern your OpenAI client uses internally
every single time you call an LLM. httpbin will respond confirming it
received your token:

```json
{
  "authenticated": true,
  "token": "my-test-token-123"
}
```

Now try it **without** the header:

```bash
curl -i https://httpbin.org/bearer
```

You'll get back `HTTP/2 401` — Unauthorized. **This is precisely what
happens if your `.env` file is missing or your API key is wrong** when you
start Phase 01's projects.

### 8.6 The exact same thing, in Python

Create a new file called `api_demo.py` and type this out:

```python
import requests  # the library that makes HTTP requests from Python

# A simple GET request — same as: curl https://httpbin.org/get
response = requests.get("https://httpbin.org/get")

print("Status code:", response.status_code)   # the 3-digit number, Section 3.3
print("As JSON:", response.json())             # .json() converts the JSON
                                                 # text response straight into
                                                 # a Python dictionary

# Sending an Authorization header — same as the curl -H example above
headers = {"Authorization": "Bearer my-test-token-123"}
auth_response = requests.get("https://httpbin.org/bearer", headers=headers)

print("\nAuthenticated request status:", auth_response.status_code)
print("Authenticated response:", auth_response.json())
```

Run it:

```bash
python api_demo.py    # Windows
python3 api_demo.py   # Mac
```

You should see the status codes and JSON dictionaries printed, matching
what `curl` showed you — confirming that `curl` and Python's `requests`
library are doing the exact same underlying thing, just from different
tools.

### 8.7 Doing it properly — the token in a `.env` file

Right now, `"my-test-token-123"` is hardcoded directly in the script — fine
for a harmless test token, but exactly the mistake to never make with a
real key (Section 6.2). Let's fix it properly.

Create a file named `.env` in the same folder:

```bash
# .env
DEMO_API_TOKEN=my-test-token-123
```

Create a `.gitignore` file in the same folder (Section 6.5):

```bash
# .gitignore
.env
```

Now update `api_demo.py`:

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env and makes its values available

token = os.getenv("DEMO_API_TOKEN")  # the secret never appears in this file

headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://httpbin.org/bearer", headers=headers)

print("Status code:", response.status_code)
print("Response:", response.json())
```

Run it again — same result, but now the secret value lives only in `.env`,
which `.gitignore` ensures never reaches GitHub. **This exact pattern is
what every single phase of this course uses from Phase 01 onward.**

---

## 9. Checkpoint Project

**Goal:** prove you can read documentation for a real public API you've
never seen before, and call it correctly using everything from this guide.

Use the free **Advice Slip API** — no key required:

1. In your browser or with `curl`, visit `https://api.adviceslip.com/advice`
   and look at the JSON structure you get back
2. Write a Python script called `advice.py` that:
   - Makes a `GET` request to that URL using `requests`
   - Checks the status code is `200` before doing anything else
   - Pulls the actual advice text out of the JSON response (look at the
     structure you saw in step 1 to figure out the right keys)
   - Prints it in a friendly format, like: `Today's advice: <the advice text>`
3. Run it a few times — notice you get different advice each time, since
   it's a live server responding fresh each call

**You're done when:** running your script prints a random piece of advice
each time, pulled live from the internet, using nothing but what you
learned in this guide.

(Optional stretch goal: commit and push this to your `ai-engineer-course`
GitHub repo from Part 1, with a proper `.gitignore` even though this
particular script doesn't have any real secrets to hide — building the
habit matters more than the specific file.)

---

## 10. Troubleshooting FAQ

**`curl: command not found`**
Rare on Mac (curl is pre-installed). On older Windows versions, install
[Git for Windows](https://git-scm.com) (Part 1, Section 7.2) — it includes
curl, or use Python's `requests` library instead, which works identically
for everything in this guide.

**`ModuleNotFoundError: No module named 'requests'`**
You forgot to `pip install requests`, or your virtual environment isn't
activated — check for `(venv)` in your prompt (Part 1, Section 5.4).

**My Python script's `response.json()` crashes with a JSON decode error**
This means the server didn't actually send back JSON — it might have sent
back an HTML error page instead. Print `response.text` (the raw response
as plain text) first to see exactly what came back before assuming it's
valid JSON.

**`os.getenv("DEMO_API_TOKEN")` returns `None`**
Usually means `load_dotenv()` couldn't find your `.env` file — double
check it's in the same folder you're running the script from, and that
it's named exactly `.env` (not `.env.txt` — some editors hide file
extensions and accidentally append one).

**I accidentally committed a real secret to GitHub**
Treat the key as permanently compromised — go to the service that issued
it (e.g., OpenAI's dashboard) and revoke/regenerate it immediately, even
if you delete the commit afterward. The old key may already be cached or
scraped. Then add `.env` to `.gitignore` before creating a new key.

---

## 11. Key Takeaways

1. **HTTP is just an agreed-upon format for "ask and respond."** A client
   sends a request (with a method, like `GET` or `POST`), a server sends
   back a response (with a status code telling you what happened).

2. **JSON is the shared text format almost every API speaks**, and it maps
   almost directly onto Python dictionaries and lists — `response.json()`
   converts one into the other automatically.

3. **An API is a documented menu of requests a service promises to handle**
   — "calling an API" is nothing more than sending a properly formatted
   HTTP request and reading the JSON response.

4. **API keys go in `.env` files, never in code**, read via
   `os.getenv()` after `load_dotenv()`, with `.gitignore` ensuring the
   `.env` file itself never reaches GitHub.

5. **localhost means "this same computer," and a port is which specific
   program on that computer should receive the request** — `localhost:8000`
   is how you'll talk to your own FastAPI server in Phase 08.

6. **`curl` and Python's `requests` library do the exact same thing** —
   curl is faster for a quick terminal check; `requests` is what you'll use
   inside actual programs.

---

## 12. What's Next

You now understand every piece of what's happening when Phase 01's
`client.chat.completions.create(...)` runs — it's a `POST` request, with a
JSON body, an `Authorization` header carrying your API key (safely loaded
from `.env`), sent to one of OpenAI's servers, getting a JSON response back.

If you haven't yet, **Parts 2 and 3** cover core Python programming
(variables, functions, classes, error handling) — worth doing before Phase
01 if any of the Python snippets in this guide felt unfamiliar.

**Part 5 — A Gentle, No-Math AI Primer** comes next in the Phase 00 series:
what machine learning and neural networks actually are, how an LLM "writes"
one word at a time, and an intuitive feel for tokens and embeddings —
exactly the vocabulary Phase 01 starts using on its very first page.

Say **"Start Part 5"** when you're ready, or **"Start Part 2"** if you'd
like to go back and cover Python fundamentals first.
