# Technical Spec - decide before building

## Libraries (installed stage by stage, not all at once)
livekit-agents (done) | livekit-plugins-deepgram (done) | livekit-plugins-groq (done) |
livekit-plugins-elevenlabs (done) | livekit-plugins-silero (done) | livekit-plugins-tavus |
pdfplumber | python-dotenv

Note: LLM started as Gemini/OpenAI in early planning, settled on Groq
llama-3.1-8b-instant. TTS started as Cartesia, swapped to ElevenLabs
(eleven_flash_v2_5) after hitting Cartesia free-tier limits mid-Stage-2. The
pipeline is provider-agnostic by design, so this was a config change only.

## Avatar (Tavus)
- Tavus Persona API, low-latency real-time replica (Phoenix-class - pick the
  fastest real-time replica in the Tavus dashboard).
- Persona layers MUST set transport_type: livekit.
- Need TAVUS_REPLICA_ID + TAVUS_PERSONA_ID.
- Attach via tavus.AvatarSession(...); DISABLE the agent's own audio output.

## Low-latency techniques (the "algo")
Target: user stops speaking -> avatar replies under ~1.5s.
- Cascaded STREAMING pipeline: STT -> LLM -> TTS, every stage streaming.
- Fast models: Deepgram nova-3, Groq llama-3.1-8b-instant, ElevenLabs eleven_flash_v2_5.
- VAD (Silero) + LiveKit semantic turn detection (waits for real end-of-turn).
- Preemptive generation: start the reply as the user finishes.
- Gentle interruption: allow interruptions but set a small min_interruption_duration.
- Short prompts + short resume summary = faster first token.

## Other decisions
- Resume: parse -> summarize to ~5 bullets ONCE on upload; store in state. Optional
  with graceful fallback. NOT YET BUILT (Stage 5).
- State: one @dataclass (currently candidate_name, intro_notes - resume_summary,
  chosen_project, covered_topics land in Stage 5), passed across the handoff.
- Handoff (done): content-based intro_complete() function tool + time-based
  fallback timer (env STAGE1_TIMEOUT_SECS, default 90s). Logs trigger=function_call
  vs trigger=timeout so both paths are demoable.
- Small-model guard (done): llama-3.1-8b-instant occasionally leaked the
  intro_complete tool call as spoken text instead of invoking it. Fixed with
  action-based prompt phrasing (describe the action, not the literal tool name)
  plus an llm_node() override that strips any leak before it reaches TTS/transcript.
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