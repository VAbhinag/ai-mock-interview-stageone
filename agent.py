"""
Stage 3: two agents, one session, a handoff with a time-based fallback
(no resume, no scenario question, no avatar yet - that's Stage 4+).

Pipeline: Deepgram (STT) -> Groq llama-3.1-8b-instant (LLM) -> ElevenLabs
eleven_flash_v2_5 (TTS), Silero VAD.

Interview flow:
  RapportAgent   - greets the candidate, asks about their background, asks
                   ONE follow-up drawn only from what they just said, then
                   hands off. Two ways that handoff can fire:
                     - trigger=function_call: the model calls intro_complete
                       once the follow-up is answered (the normal path).
                     - trigger=timeout: if that hasn't happened within
                       STAGE1_TIMEOUT_SECS seconds (env var, default 90), an
                       asyncio timer forces the SAME transition. This is the
                       safety net for when the LLM never calls the tool.
  ExperienceAgent - takes over with a warm transition line (no re-greeting,
                   no repeated questions), asks the candidate to pick a
                   project/role, and asks one question about it.

Handoff pattern is modeled on LiveKit's multi_agent.py storyteller example:
a function tool returns (next_agent_instance, transition_hint). Returning an
Agent from a function_tool is how AgentSession swaps the active agent - the
transition hint becomes the tool's output in the shared chat history, so the
new agent's on_enter() can generate_reply() and continue naturally instead of
starting the conversation over. The fallback timer reuses the exact same
mechanism directly (session.update_agent()) instead of duplicating it.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    ModelSettings,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    llm,
    metrics,
)
from livekit.plugins import elevenlabs, deepgram, groq, silero

load_dotenv()

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

# Groq's small model occasionally fumbles function calling and writes the
# tool call out as plain text (e.g. "intro_complete>{...}") instead of a
# structured tool call. Matches the tool name onward so that leaked text -
# and anything after it, since it's just JSON garbage - never reaches the
# transcript or TTS.
_LEAKED_TOOL_CALL_RE = re.compile(r"\bintro_complete\b.*", re.IGNORECASE | re.DOTALL)

# How long RapportAgent waits for the normal intro_complete handoff before
# forcing the transition itself. Configurable so it can be set short for a
# demo of the timeout path without editing code.
STAGE1_TIMEOUT_SECS = float(os.getenv("STAGE1_TIMEOUT_SECS", "90"))


@dataclass
class InterviewData:
    """Shared state carried across the rapport -> experience handoff."""

    candidate_name: str | None = None
    intro_notes: str | None = None


class RapportAgent(Agent):
    """Stage 1 of the interview: warm up the candidate, no resume involved."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a warm, friendly interviewer conducting the opening, "
                "rapport-building part of a mock interview. Speak briefly and "
                "conversationally, this is voice, not text. Never recap or summarize "
                "what the candidate just said back to them - acknowledge it in a few "
                "words (e.g. \"Got it,\" \"That makes sense,\" \"Interesting -\") and "
                "move straight to the next question.\n\n"
                "1. Greet the candidate and ask them to tell you about their background.\n"
                "2. After they respond, ask exactly ONE follow-up question based ONLY on "
                "what they just said. Do not invent details or reference a resume.\n"
                "3. As soon as they answer that follow-up, silently move the interview "
                "forward to the next part. This happens as a background action, not as "
                "something you say - do not ask a second follow-up, and do not announce "
                "or narrate the transition yourself in words.\n\n"
                "Your spoken replies should only ever be natural interview conversation. "
                "Never write out code, JSON, or any programming syntax in your reply."
            )
        )
        # Guards the rapport -> experience handoff so it only ever fires
        # once, whichever path (tool call or timeout) gets there first.
        self._handed_off = False
        self._fallback_timer: asyncio.Task[None] | None = None

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the candidate warmly, introduce yourself as their interviewer, "
                "and ask them to tell you about their background."
            )
        )
        self._fallback_timer = asyncio.create_task(self._fallback_after_timeout())

    async def _fallback_after_timeout(self) -> None:
        """Force the rapport -> experience handoff if intro_complete never fires.

        This is the graded safety net: LLM tool calls can silently fail to
        fire (especially on a small model), so the interview must not get
        stuck in stage 1 forever. Reuses the exact same transition primitive
        (session.update_agent) that the normal tool-call path ends up using
        internally, so there's only one real handoff mechanism, not two.
        """
        await asyncio.sleep(STAGE1_TIMEOUT_SECS)
        if self._handed_off:
            return  # normal handoff already happened, nothing to do
        self._handed_off = True

        intro_notes = (
            self.session.userdata.intro_notes
            or "The candidate didn't finish the background discussion before "
            "time ran out - ask general experience questions."
        )
        logger.info(
            "stage transition: rapport -> experience (trigger=timeout) "
            "timeout_secs=%s intro_notes=%s",
            STAGE1_TIMEOUT_SECS,
            intro_notes,
        )
        self.session.update_agent(ExperienceAgent(intro_notes=intro_notes))

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        """Strip a leaked `intro_complete` tool call out of the text stream.

        This is a belt-and-suspenders guard alongside the instructions above:
        if the model still writes the tool call as text instead of invoking
        it, cut it before it reaches the transcript or TTS, so the candidate
        never hears or sees it. Once the leak starts we suppress the rest of
        the turn too - the leaked JSON is streamed across many small chunks,
        and everything after the marker is tool-call plumbing, not something
        meant for the candidate.
        """
        suppressing = False
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            content = chunk if isinstance(chunk, str) else (chunk.delta.content if chunk.delta else None)
            if content is None:
                yield chunk
                continue

            if suppressing:
                continue

            if not _LEAKED_TOOL_CALL_RE.search(content):
                yield chunk
                continue

            suppressing = True
            cleaned = _LEAKED_TOOL_CALL_RE.sub("", content).rstrip()
            if isinstance(chunk, str):
                if cleaned:
                    yield cleaned
            else:
                chunk.delta.content = cleaned or None
                yield chunk

    @function_tool
    async def intro_complete(
        self,
        context: RunContext[InterviewData],
        candidate_name: str,
        intro_notes: str,
    ) -> tuple[Agent, str] | None:
        """Call this once the candidate has answered your one follow-up question.

        Args:
            candidate_name: The candidate's first name if they mentioned it,
                otherwise "the candidate".
            intro_notes: A 2-3 sentence summary of their background and the
                follow-up answer, for the next interviewer to build on.
        """
        if self._handed_off:
            # The fallback timer already forced the transition; this call
            # arrived too late (e.g. right at the timeout boundary). The
            # session is already on ExperienceAgent, so there's nothing to do.
            return None
        self._handed_off = True

        if self._fallback_timer is not None and not self._fallback_timer.done():
            self._fallback_timer.cancel()

        context.userdata.candidate_name = candidate_name
        context.userdata.intro_notes = intro_notes

        logger.info(
            "stage transition: rapport -> experience (trigger=function_call) "
            "candidate_name=%s intro_notes=%s",
            candidate_name,
            intro_notes,
        )

        next_agent = ExperienceAgent(intro_notes=intro_notes)
        return next_agent, "Great, thanks for sharing that! Let's shift gears and talk about your experience."


class ExperienceAgent(Agent):
    """Stage 2 of the interview: pick a project/role, ask about it."""

    def __init__(self, intro_notes: str) -> None:
        super().__init__(
            instructions=(
                "You are the SAME interviewer as before, now moving into the "
                "experience deep-dive. Here is what you learned about the candidate "
                f"in the intro, for your own context only: {intro_notes}\n\n"
                "Do not re-greet the candidate and do not repeat earlier questions. "
                "Pick up naturally from the transition. Ask the candidate to pick one "
                "project or role they'd like to talk about, then ask ONE thoughtful "
                "question about it. Keep replies short and conversational. Never recap "
                "or summarize what the candidate just said back to them - acknowledge "
                "it in a few words (e.g. \"Got it,\" \"That makes sense,\" \"Interesting -\") "
                "and move straight to the next question."
            )
        )

    async def on_enter(self) -> None:
        # No explicit instructions: the chat history already has the rapport
        # conversation plus the handoff's transition line, so the model
        # continues naturally instead of restarting the conversation.
        self.session.generate_reply()


async def entrypoint(ctx: JobContext) -> None:
    userdata = InterviewData()
    session = AgentSession[InterviewData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=elevenlabs.TTS(model="eleven_flash_v2_5"),
        vad=silero.VAD.load(),
    )

    # Lightweight latency logging now so we have a number to point at later
    # when tuning for the <1.5s reply target.
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    # Connect before starting the session: RapportAgent's on_enter() fires a
    # greeting as soon as the session starts, so the room needs to already be
    # connected for that first reply to have somewhere to go.
    await ctx.connect()

    await session.start(
        room=ctx.room,
        agent=RapportAgent(),
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
