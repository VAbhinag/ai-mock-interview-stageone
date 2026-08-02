# AI Mock Interview — Real-Time Voice Interviewer with Video Avatar

A real-time, voice-driven AI mock interviewer built on **LiveKit Agents**. It runs a
two-stage interview — a rapport warm-up and an experience deep-dive — as separate agents
that hand off seamlessly, guarantees forward progress with a time-based fallback, and
presents as a lip-synced video avatar.

Built for the Jobnova / Liba Space **AI Algorithm Engineer** take-home (Part 1: AI Mock
Interview).

---

## What it does

- **Two interview stages that feel like one conversation**
  - *Stage 1 — Rapport:* greets the candidate, asks about their background, and asks one
    follow-up based on their answer.
  - *Stage 2 — Experience:* the candidate picks a project/role, and the interviewer probes
    it, then closes the interview with a spoken wrap-up.
- **Seamless handoff** between stages via a function-tool call, with shared state so
  questions never repeat.
- **Time-based fallback** — if the normal handoff trigger doesn't fire within a
  configurable window, the interview forces the transition so it never gets stuck.
- **Video avatar** — a lip-synced digital human (Tavus) speaks the interviewer's lines in
  real time, with the avatar degrading gracefully to voice-only if unavailable.
- **Low latency** — streaming STT → LLM → TTS pipeline; ~250 ms LLM time-to-first-token.

---

## Architecture

```
You (speech)
   │
   ▼
Deepgram  (STT — speech to text)
   │
   ▼
Groq / Llama 3.1 8B  (LLM — the interview logic / "brain")
   │
   ▼
ElevenLabs Flash  (TTS — text to speech, low latency)
   │
   ▼
Tavus  (renders the video avatar + lip-sync, plays the audio)
   │
   ▼
You (see + hear the interviewer)

LiveKit  — the real-time room carrying audio/video between you and the agent.
Silero   — voice-activity detection (knows when you stop talking).
```

Each interview stage is its own `Agent` subclass inside a single `AgentSession`. A shared
`@dataclass` carries state (candidate name, intro notes) across the handoff so Stage 2
builds on Stage 1 instead of restarting.

**Design decisions worth calling out:**

- **Two agents, one handoff mechanism.** The normal handoff (an `intro_complete` function
  call) and the fallback timer both trigger the *same* transition primitive — one real
  handoff path, not two.
- **Fallback timer handles the race condition.** If the timer and the normal trigger fire
  near-simultaneously, a guard flag ensures the interview switches exactly once.
- **Avatar audio is handled by Tavus.** The agent's own audio output is disabled when the
  avatar is active, so Tavus plays the audio and keeps lip-sync tight (no double-talk). If
  the avatar can't start, the app falls back to voice-only automatically.
- **Provider-agnostic pipeline.** STT/LLM/TTS are independent plug-ins, so providers can be
  swapped in one line — which is exactly what happened during the build when free-tier
  limits were hit on several services.

---

## Tech stack

| Layer | Provider / Library |
|-------|-------------------|
| Orchestration | LiveKit Agents (Python) |
| Speech-to-text (STT) | Deepgram `nova-3` |
| LLM (interview logic) | Groq — Llama 3.1 8B Instant |
| Text-to-speech (TTS) | ElevenLabs `eleven_flash_v2_5` |
| Video avatar | Tavus (Phoenix face, LiveKit integration) |
| Voice-activity detection | Silero |

---

## Setup

### 1. Prerequisites
- Python 3.10+ (developed on 3.14)
- A [LiveKit Cloud](https://cloud.livekit.io) project
- API keys: LiveKit, Deepgram, Groq, ElevenLabs, Tavus (all have free tiers)

### 2. Install
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure
Copy `.env.example` to `.env` and fill in your keys:
```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
ELEVEN_API_KEY=...
TAVUS_API_KEY=...
TAVUS_REPLICA_ID=...
# optional:
STAGE1_TIMEOUT_SECS=90     # fallback timer for the stage-1 -> stage-2 handoff
```

### 4. Run
```bash
python agent.py dev
```
Then open the **LiveKit Agents Console** (cloud.livekit.io → Agents → Console), start a
session, allow your microphone, and talk to the interviewer.

---

## How the requirements are met (Part 1)

1. **Self-intro + experience stages, low latency, follow-ups** — two-agent design, streaming
   pipeline, ~250 ms time-to-first-token.
2. **Smooth transition, no repeated prompts** — function-call handoff + shared state; the
   second stage continues the conversation instead of restarting.
3. **Time-based fallback mechanism** — `STAGE1_TIMEOUT_SECS` timer forces the handoff if the
   normal trigger doesn't fire; logs `trigger=function_call` vs `trigger=timeout`.
4. **Tavus avatar with lip-sync + low-latency TTS** — real-time avatar via the LiveKit Tavus
   plugin; ElevenLabs Flash for fast TTS; graceful voice-only fallback.

Observable in the logs during a run:
```
========== STAGE 1: RAPPORT (background + one follow-up) ==========
stage transition: rapport -> experience (trigger=function_call) ...
========== STAGE 2: EXPERIENCE (project deep-dive) ==========
interview complete
```

---

## Notes

- The interview is designed to feel like one continuous conversation; the two-agent split is
  an internal architecture detail.
- Free-tier note: the Tavus avatar is metered in conversational minutes, so avatar time is
  used sparingly; the pipeline runs fully in voice-only mode without it.
- AI-assisted development was used, as encouraged in the brief. Every design decision above
  was made and can be explained by the author.

## Possible next steps
- Resume upload to drive resume-specific experience questions.
- A scenario/hypothetical question to test how a candidate reasons through a new problem.
- A dedicated web frontend and a hosted, shareable demo link.