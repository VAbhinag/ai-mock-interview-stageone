# Build Plan - feed these to Claude Code one stage at a time

Golden rule: get each stage fully working before starting the next. Commit after
every green stage so you can roll back.

Status: Stages 1-3 done. Stage 4 (Tavus avatar + frontend) is next.

## Stage 0 - Accounts & keys (do yourself)
Create accounts + grab keys: LiveKit Cloud, Deepgram, Groq, ElevenLabs, Tavus.
Put them in .env (never commit). Watch Tavus trial minutes.

## Stage 1 - Bare voice agent (PROVE AUDIO FIRST) - DONE
> Read CLAUDE.md. We build in stages - do NOT build everything at once. Stage 1:
> install the LiveKit Agents plugin deps we need, and create a minimal voice agent
> (Deepgram STT -> LLM -> TTS, Silero VAD) that just chats with me in a LiveKit room.
> Give me .env.example, requirements.txt, and the exact commands to run and test it.
> Stop after Stage 1 so I can verify audio works.

Done when: you speak, it answers, latency feels ok. Commit.
Actually built: Deepgram nova-3 -> Groq llama-3.1-8b-instant -> Cartesia TTS, Silero
VAD. Committed as `bdfb146`.

## Stage 2 - Two stages + normal handoff - DONE
> Now split into two Agent classes inside one AgentSession: a self-intro (rapport)
> interviewer and a past-experience (deep-dive) interviewer, sharing a @dataclass
> for state. Stage 1 asks the background opener + ONE response-based follow-up, then
> hands off via a function call. No duplicated prompts. Model it on the LiveKit
> multi_agent.py storyteller handoff.

Done when: intro completes and cleanly transitions to the experience stage. Commit.
Actually built: RapportAgent + ExperienceAgent share `InterviewData` (candidate_name,
intro_notes). RapportAgent calls the `intro_complete` function tool, which returns
`(ExperienceAgent(...), transition_line)` - the pattern AgentSession uses to swap the
active agent. Hit two snags along the way: Cartesia free-tier limits mid-build (swapped
TTS to ElevenLabs, pipeline is provider-agnostic so this was config-only), and the 8B
model occasionally leaking the `intro_complete` call as spoken text instead of invoking
it (fixed with action-based instruction phrasing + an `llm_node()` guard that strips any
leak before it reaches TTS/transcript). Committed as `d7e93cb`.

## Stage 3 - Time-based fallback timer (THE graded feature) - DONE
> Add a configurable time-based fallback: start an asyncio timer when stage 1 begins;
> if the normal "intro complete" trigger hasn't fired within N seconds (env var),
> force the SAME transition to stage 2. Log which path fired (trigger vs fallback).

Done when: you can demo BOTH paths - normal completion AND a forced timeout. Commit.
Actually built: `RapportAgent.on_enter()` starts an asyncio task on entry; after
`STAGE1_TIMEOUT_SECS` (env var, default 90) it calls `session.update_agent()` directly -
the same primitive the normal tool-call handoff uses internally, so there's one handoff
mechanism, not two. A `_handed_off` flag plus explicit `.cancel()` on the timer task
prevent a double transition either way. Logs `trigger=function_call` vs `trigger=timeout`.
Not yet committed - code complete in the working tree.

## Stage 4 - Tavus avatar (current next step)
> Create/confirm a Tavus persona with transport_type: livekit. Attach
> tavus.AvatarSession(replica_id, persona_id) and DISABLE the agent's own audio
> output. Wire a LiveKit starter frontend so I can see and talk to the avatar.

Done when: lip-synced avatar interviews me end-to-end. Commit.

## Stage 5 - Resume + Stage 2 depth logic
> Add optional resume upload: parse the PDF with pdfplumber and summarize to ~5
> bullets ONCE on upload, store in state. In Stage 2: ask the candidate which project
> to dive into, run the problem->identify->approach->outcome arc with follow-ups that
> fuse their answer + resume, then ask ONE scenario question generated from that
> project. No resume = general questions.

Done when: resume-driven deep dive + scenario works, with graceful no-resume fallback. Commit.

## Stage 6 - Polish for the demo
> Tune interruption handling and turn detection. Add logging for reply latency and
> stage transitions. Write a README covering setup + the 3 design decisions: the
> handoff, the fallback timer, and disabling agent audio for Tavus.

Done when: smooth on camera, README explains the "why". Commit.

## Video narration cheat-sheet
1. Overview - what it does, two-stage flow, the stack.
2. Handoff - each stage is its own agent; transition is a function call. Show it.
3. Fallback timer - WHY it exists (LLM triggers can miss), then demo the timeout path.
4. Avatar - Tavus is a separate participant; agent audio disabled. Name this gotcha.
5. Latency/interruptions - show a number, show yourself cutting in naturally.
6. Honesty beat - "AI-assisted, as encouraged; here's why each decision was made."