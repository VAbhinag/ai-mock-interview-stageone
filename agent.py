"""
Stage 1: bare-bones voice agent (PROVE AUDIO FIRST).

Pipeline: Deepgram (STT) -> OpenAI gpt-4o-mini (LLM) -> Cartesia (TTS),
with Silero VAD for turn detection. No interview logic yet, no two
stages, no avatar, no resume - just a friendly assistant that chats in
a LiveKit room. This proves the end-to-end audio pipeline before any
interview-specific logic is layered on top.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import cartesia, deepgram, groq, silero

load_dotenv()

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, warm voice assistant having a casual spoken "
                "conversation. Keep replies short and conversational (1-3 sentences) "
                "since this is voice, not text. Ask follow-up questions to keep the "
                "conversation going."
            )
        )


async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    # Lightweight latency logging now so we have a number to point at later
    # when tuning for the <1.5s reply target.
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(),
    )

    await ctx.connect()

    # Kick off the conversation so the user isn't greeted by silence.
    session.generate_reply(
        instructions="Greet the user warmly, introduce yourself briefly, and ask how they're doing."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
