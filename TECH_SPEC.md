# Technical Spec - decide before building

## Libraries (installed stage by stage, not all at once)
livekit-agents (done) | livekit-plugins-deepgram | livekit-plugins-google |
livekit-plugins-cartesia | livekit-plugins-silero | livekit-plugins-tavus |
pdfplumber | python-dotenv

## Avatar (Tavus)
- Tavus Persona API, low-latency real-time replica (Phoenix-class - pick the
  fastest real-time replica in the Tavus dashboard).
- Persona layers MUST set transport_type: livekit.
- Need TAVUS_REPLICA_ID + TAVUS_PERSONA_ID.
- Attach via tavus.AvatarSession(...); DISABLE the agent's own audio output.

## Low-latency techniques (the "algo")
Target: user stops speaking -> avatar replies under ~1.5s.
- Cascaded STREAMING pipeline: STT -> LLM -> TTS, every stage streaming.
- - Fast models: Deepgram nova-3, Gemini 2.0 Flash, Cartesia.
- VAD (Silero) + LiveKit semantic turn detection (waits for real end-of-turn).
- Preemptive generation: start the reply as the user finishes.
- Gentle interruption: allow interruptions but set a small min_interruption_duration.
- Short prompts + short resume summary = faster first token.

## Other decisions
- Resume: parse -> summarize to ~5 bullets ONCE on upload; store in state. Optional
  with graceful fallback.
- State: one @dataclass (candidate_name, intro_notes, resume_summary,
  chosen_project, covered_topics), passed across the handoff.
- Handoff: content-based stage_complete() function + time-based fallback timer
  (env STAGE1_TIMEOUT_SECS).
- Single persona/voice across both stages.
- Secrets in .env; ship .env.example; never commit .env.
- Log reply latency + which transition path fired.

## Deployment
- Agent: LiveKit Cloud (managed, via LiveKit CLI + Dockerfile). Or run locally with
  `python agent.py dev` and screen-record - fine for the submission.
- Frontend: LiveKit Next.js starter, deploy to Vercel (or run localhost for the demo).

## Repo hygiene (for the Kevin submission)
- .gitignore: .env, venv/, node_modules/, __pycache__/
- README: what it is, setup, how to run, the 3 design decisions.
- Clean commits (one per working stage). Tag v1.0 when done.