# Demo Video — Script & Recording Plan

Target length: ~5-7 minutes. Goal: show it WORKING (the "serious demo" Kevin asked for)
AND show you understand the engineering. Both matter.

Your Tavus avatar minutes are limited, so this plan records the avatar ONCE (the key
segment) and does everything else voice-only / on-screen (free).

---

## Recording order (record the avatar part FIRST, while you have minutes)

Record in **segments** — you don't need one continuous take. Stitch them or submit as-is.

### SEGMENT A — Live avatar interview  (avatar ON — uses your Tavus minutes)
Record this first, in ONE clean take. Keep it ~2-3 minutes. Speak in clear, complete
sentences (helps the speech-to-text).

**Before recording:** have the LiveKit Console ready, agent running, mic tested.

**On camera, do a short real interview:**
1. Start the session. Let Andrew (the avatar) greet you.
2. Give a clear background answer (e.g. your PhD + data-analyst background).
3. Answer the one follow-up.
4. Let it hand off to the experience stage; pick a project; answer 1-2 questions.
5. Let it close with the spoken ending.

**You don't narrate much during this** — let the product speak. Maybe one line at the start:
> "Here's the interviewer running live — a lip-synced avatar conducting a two-stage interview."

End the session promptly to conserve minutes.

---

### SEGMENT B — Intro  (voice-only / just you talking — free)
Record separately (can be before A in the final edit). ~30-45 sec.

> "Hi, I'm Abhinag. This is my submission for Part 1 — an AI mock interviewer. It's a
> real-time voice agent built on LiveKit: Deepgram for speech-to-text, Groq running Meta's
> Llama for the interview logic, ElevenLabs Flash for low-latency voice, and Tavus for the
> video avatar. It runs a two-stage interview — a rapport warm-up and an experience
> deep-dive — that hand off seamlessly."

---

### SEGMENT C — Engineering walkthrough  (voice-only + screen — free)
Show your terminal + VS Code + the LiveKit Console. ~2-3 min.

**1. The two-stage architecture (show the terminal logs):**
> "Each interview stage is its own agent. Watch the logs: Stage 1 rapport starts here, then
> this line — 'stage transition: rapport to experience, trigger function_call' — is the
> handoff firing, then Stage 2 experience begins. They share state so nothing repeats."

**2. The fallback timer (the standout — demo BOTH paths):**
> "The handoff normally fires when the model signals the intro is done. But if that never
> fires — say the candidate rambles — the interview would get stuck. So I added a time-based
> fallback."
- Show a normal run → point to `trigger=function_call`.
- Then run with a short timeout (`STAGE1_TIMEOUT_SECS=10`) and stay silent → point to
  `trigger=timeout`.
> "Same handoff, two triggers — the fallback guarantees the interview always progresses."

**3. Latency (show the Console Metrics tab):**
> "For low-latency TTS I used ElevenLabs Flash — time-to-first-token around 250
> milliseconds. End-to-end, including the avatar rendering, is roughly two seconds, which
> keeps it feeling conversational."

**4. The avatar wiring (one sentence):**
> "The avatar runs as a separate participant that renders the face; I disable the agent's
> own audio so Tavus handles it and the lip-sync stays tight. It also falls back to
> voice-only if the avatar's unavailable."

---

### SEGMENT D — Design reasoning + honesty  (just you — free)  ~1 min
> "A few decisions I want to call out. I made the pipeline provider-agnostic — during the
> build I hit free-tier and quota limits on several providers, so I swapped components by
> reading the actual errors, without rewriting the app. I used a small, fast Llama model for
> low latency and added a guard for its tool-calling quirks. And I built the fallback timer
> because real interviews don't always hit a clean end-of-stage signal.
>
> This was AI-assisted, as the brief encouraged — but every one of these decisions was mine,
> and I can walk through any part of the code."

---

### SEGMENT E — Close  (~15 sec)
> "All four required features work — the two stages, the smooth handoff, the time-based
> fallback, and the avatar with low-latency TTS. The code's on GitHub. Thanks for the
> opportunity — I'd love to talk more about it."

---

## Checklist before you hit record
- [ ] Agent runs cleanly (`python agent.py dev`), no errors.
- [ ] Fresh Tavus minutes confirmed (platform.tavus.io/billing).
- [ ] Mic tested; quiet room; speak in full sentences.
- [ ] Terminal + Console + VS Code windows arranged so logs are readable on screen.
- [ ] Rehearsed Segment A once in VOICE-ONLY mode (avatar off) so the avatar take is clean.

## Tools
- Screen record: Windows **Win+G**, or **OBS**, or **Loom** (Loom gives a shareable link).
- Export/host: Loom link, or upload to Google Drive / unlisted YouTube and share the link.

## Submitting to Kevin (Monday AM/noon CT)
Short professional email + GitHub repo link + video link. Keep it brief and confident.