# Build Plan - feed these to Claude Code one stage at a time

Golden rule: get each stage fully working before starting the next. Commit after
every green stage so you can roll back.

## Stage 0 - Accounts & keys (do yourself)
Create accounts + grab keys: LiveKit Cloud, OpenAI, Deepgram, Cartesia, Tavus.
Put them in .env (never commit). Watch Tavus trial minutes.

## Stage 1 - Bare voice agent (PROVE AUDIO FIRST)
> Read CLAUDE.md. We build in stages - do NOT build everything at once. Stage 1:
> install the LiveKit Agents plugin deps we need, and create a minimal voice agent
> (Deepgram STT -> Gemini LLM -> Cartesia TTS, Silero VAD) that just chats with me
> in a LiveKit room. Give me .env.example, requirements.txt, and the exact commands
> to run and test it. Stop after Stage 1 so I can verify audio works.

Done when: you speak, it answers, latency feels ok. Commit.

## Stage 2 - Two stages + normal handoff
> Now split into two Agent classes inside one AgentSession: a self-intro (rapport)
> interviewer and a past-experience (deep-dive) interviewer, sharing a @dataclass
> for state. Stage 1 asks the background opener + ONE response-based follow-up, then
> hands off via a function call. No duplicated prompts. Model it on the LiveKit
> multi_agent.py storyteller handoff.

Done when: intro completes and cleanly transitions to the experience stage. Commit.

## Stage 3 - Time-based fallback timer (THE graded feature)
> Add a configurable time-based fallback: start an asyncio timer when stage 1 begins;
> if the normal "intro complete" trigger hasn't fired within N seconds (env var),
> force the SAME transition to stage 2. Log which path fired (trigger vs fallback).

Done when: you can demo BOTH paths - normal completion AND a forced timeout. Commit.

## Stage 4 - Resume + Stage 2 depth logic
> Add optional resume upload: parse the PDF with pdfplumber and summarize to ~5
> bullets ONCE on upload, store in state. In Stage 2: ask the candidate which project
> to dive into, run the problem->identify->approach->outcome arc with follow-ups that
> fuse their answer + resume, then ask ONE scenario question generated from that
> project. No resume = general questions.

Done when: resume-driven deep dive + scenario works, with graceful no-resume fallback. Commit.

## Stage 5 - Tavus avatar
> Create/confirm a Tavus persona with transport_type: livekit. Attach
> tavus.AvatarSession(replica_id, persona_id) and DISABLE the agent's own audio
> output. Wire a LiveKit starter frontend so I can see and talk to the avatar.

Done when: lip-synced avatar interviews me end-to-end. Commit.

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