# Project: AI Mock Interview (take-home Part 1)

Real-time VOICE mock interview with a Tavus video avatar, built on the LiveKit
Agents Python framework. Two interview stages that hand off cleanly:
self-introduction (rapport) -> past-experience (deep dive).

This is a job take-home. AI-assisted coding is explicitly encouraged. The bar is a
working demo + being able to explain every design decision on camera.

## Hard requirements (do not drop any)
- Voice, not text. Low latency (user stops speaking -> avatar replies under ~1.5s).
- Two stages with a clean handoff and NO duplicated/repeated prompts.
- A time-based fallback timer that FORCES the transition to stage 2 if the normal
  "intro complete" trigger never fires. This is the key graded feature.
- Gentle interruption handling (user can cut in without the agent talking over them).
- Tavus avatar: real-time render + accurate lip-sync.

## Stack
- livekit-agents (1.x), livekit-plugins-tavus
- - STT: deepgram (nova-3) | LLM: google gemini (gemini-2.0-flash) | TTS: cartesia
- VAD/turn detection: silero | resume parsing: pdfplumber
- python-dotenv for secrets

## Interview design (the logic)
- Stage 1 (rapport): opener "tell me about your background" -> ONE follow-up drawn
  ONLY from the candidate's response (NOT the resume) to build comfort -> warm
  handoff line that signals the shift to depth. Keep resume OUT of stage 1.
- Stage 2 (depth): candidate PICKS which project/role to dive into -> deep-dive
  arc: problem -> how they identified it -> how they approached it -> outcome.
  Follow-ups fuse the candidate's response WITH the resume for that project.
  Then a SCENARIO question generated from that project (a new twist) to test how
  they'd approach an unfamiliar problem. Then close.
- One avatar + one voice across both stages (the two-agent split is internal only).
- Resume: parsed + summarized ONCE on upload, held in state. Optional input with a
  graceful fallback (no resume = general experience questions).

## Architecture
- Each stage = its own Agent subclass with its own instructions.
- One AgentSession runs the whole interview; a shared @dataclass carries state
  (candidate_name, intro_notes, resume_summary, chosen_project, covered_topics).
- Handoff (normal): stage-1 agent calls a function when the intro is complete.
- Handoff (fallback): an asyncio timer fires the SAME transition after N seconds
  if the normal trigger hasn't fired. N is configurable via env.
- Tavus runs as a separate participant that renders the avatar; the agent's own
  audio output is DISABLED (Tavus handles audio) to avoid double-playback.

## Rules for the assistant (Claude Code)
1. Build in small, testable stages. After each stage, STOP and give me the exact
   commands to run + what I should see/hear. Do not jump ahead.
2. Never commit .env. All secrets live in .env only; provide .env.example.
3. Disable the AgentSession's own audio output once Tavus is attached.
4. Tavus persona must use transport_type: livekit.
5. Prefer clear, commented code over cleverness - I need to explain it on video.
6. Add lightweight logging for reply latency and stage transitions.
7. Run the code yourself after each stage and fix your own errors before handing back.