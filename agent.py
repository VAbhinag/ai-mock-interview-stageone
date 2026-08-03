"""
Stage 3: two agents, one session, a handoff with a time-based fallback
(no resume, no scenario question, no avatar yet - that's Stage 4+).

Pipeline: Deepgram (STT) -> Groq llama-3.1-8b-instant (LLM) -> Deepgram Aura
aura-2-orion-en (TTS), Silero VAD.

Interview flow:
  RapportAgent   - greets the candidate, asks about their background, asks
                   ONE follow-up drawn only from what they just said, then
                   hands off. Two ways that handoff can fire:
                     - trigger=function_call: the model calls intro_complete
                       once the follow-up is answered (the normal path).
                     - trigger=timeout: if that hasn't happened within
                       STAGE1_TIMEOUT_SECS seconds (env var, default 180), an
                       asyncio timer forces the SAME transition - but only once
                       the candidate isn't mid-answer and Andrew isn't mid-reply,
                       so it never cuts anyone off. This is the safety net for
                       when the LLM never calls the tool, not a hard deadline.
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
    AgentStateChangedEvent,
    JobContext,
    MetricsCollectedEvent,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    llm,
    metrics,
)
from livekit.plugins import deepgram, groq, silero, tavus

load_dotenv()

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

# Groq's small model occasionally fumbles function calling and writes a tool
# call out as plain text instead of a structured tool call - sometimes bare
# ("intro_complete>{...}"), sometimes wrapped ("(function=intro_complete>{...})",
# "<function=interview_complete>{...}</function>"). This matches the ENTIRE
# leaked artifact, including any wrapper, so nothing but the tool name onward
# is left dangling.
_LEAK_RE = re.compile(
    r"(?:[\(<]?\s*function\s*=\s*)?\b(?:intro_complete|interview_complete)\b.*",
    re.IGNORECASE | re.DOTALL,
)

# A leak can also show up as a bare/partial function-call TAG fragment with
# no tool name attached at all - most often a stray "</function>" left over
# from the model's own tool-call template, trailing after an otherwise clean
# message (e.g. "...best of luck. </function>"). Unlike _LEAK_RE, a match
# here has no JSON payload riding along with it, so only the tag itself gets
# removed - the rest of the turn is not suppressed.
#
# The closing ">" is REQUIRED here, unlike at end-of-stream below: while more
# chunks are still coming, an optional ">" would fire on a bare "<function"
# the instant it arrives, before we can tell whether "=intro_complete" was
# about to follow it (which needs to stay intact for _LEAK_RE to catch the
# wrapped-tool-name leak). Only a fully-closed tag is safe to strip mid-stream.
_LEAK_TAG_RE = re.compile(r"</\s*function\s*>|<\s*function\s*>|function\s*>", re.IGNORECASE)

# A trailing "<function" / "</function" that's still unclosed - only safe to
# strip once the stream has actually ended, since nothing is left that could
# still complete it into a real tag (or a wrapped tool-name leak).
_LEAK_TRAILING_TAG_RE = re.compile(r"</?\s*function\s*$", re.IGNORECASE)

# The leak streams across many small chunks, so a wrapper prefix like
# "(function=" - or even the tool name itself - can arrive one chunk at a
# time. This is every literal string a leak could look like WHILE it's still
# forming (every wrapper variant x both tool names, including no wrapper at
# all, plus the bare closing/stray tag forms above). _suspicious_tail_len()
# holds back any buffer tail that's a prefix of one of these, however far
# into the string it's gotten, so a partial leak never gets released before
# we know whether it's actually going to become one.
_LEAK_LEAD_INS = tuple(
    f"{wrapper}{name}"
    for wrapper in ("", "function=", "(function=", "<function=")
    for name in ("intro_complete", "interview_complete")
) + ("</function>", "function>")


def _suspicious_tail_len(text: str) -> int:
    """Length of the longest tail of `text` that's a case-insensitive prefix
    of one of `_LEAK_LEAD_INS`. 0 means nothing is ambiguous - the whole
    text is safe to release.
    """
    lowered = text.lower()
    max_lead_in_len = max(len(lead_in) for lead_in in _LEAK_LEAD_INS)
    for tail_len in range(min(len(text), max_lead_in_len), 0, -1):
        tail = lowered[-tail_len:]
        if any(lead_in.startswith(tail) for lead_in in _LEAK_LEAD_INS):
            return tail_len
    return 0


async def _filter_leaked_tool_calls(chunks):
    """Strip a leaked intro_complete/interview_complete tool call (wrapper
    included) out of a streaming LLM text/ChatChunk stream, without ever
    emitting a partial one.

    Shared by RapportAgent.llm_node and ExperienceAgent.llm_node so both
    stay in sync - this only ever touches `.content` text, never the
    structured `tool_calls` field, so a real tool invocation (or a leaked-
    but-suppressed one) can never itself trigger a handoff; the handoff
    still only fires from the actual function_tool call.
    """
    held = ""
    suppressing = False
    async for chunk in chunks:
        content = chunk if isinstance(chunk, str) else (chunk.delta.content if chunk.delta else None)
        if content is None:
            yield chunk
            continue

        if suppressing:
            continue

        combined = held + content
        match = _LEAK_RE.search(combined)
        if match:
            # Confirmed leak: release whatever was clean before it, drop the
            # leak itself plus the rest of the turn (it's just JSON/tag
            # garbage from here on).
            held = ""
            suppressing = True
            safe = combined[: match.start()].rstrip()
            if safe:
                yield safe
            continue

        # No tool-name leak, but strip any bare/partial tag fragment (e.g. a
        # stray "</function>") on its own - no JSON payload attached, so no
        # need to suppress the rest of the turn, just remove the fragment.
        # Only trim the whitespace the tag left behind, not a legitimate
        # trailing space that just hasn't been followed by the next chunk yet.
        de_tagged = _LEAK_TAG_RE.sub("", combined)
        combined = de_tagged.rstrip() if de_tagged != combined else combined

        hold_len = _suspicious_tail_len(combined)
        safe, held = combined[: len(combined) - hold_len], combined[len(combined) - hold_len :]
        if safe:
            yield safe

    if held and not suppressing:
        # Stream ended with text still held back - confirm it one last time
        # before releasing it. This matters most right at the very end of
        # the final message, where a trailing "</function>" (closed or not)
        # can end up sitting in the buffer with no further chunks left to
        # trigger the checks above.
        match = _LEAK_RE.search(held)
        if match:
            held = held[: match.start()].rstrip()
        held = _LEAK_TAG_RE.sub("", held)
        held = _LEAK_TRAILING_TAG_RE.sub("", held).rstrip()
        if held:
            yield held

# How long RapportAgent waits for the normal intro_complete handoff before
# forcing the transition itself. Configurable so it can be set short for a
# demo of the timeout path without editing code. Raised 90 -> 180 -> 300:
# even 180 was still racing a normal-paced rapport exchange and occasionally
# winning, forcing the handoff after just the candidate's first answer. Five
# minutes is comfortably longer than a greeting + one follow-up should ever
# take, so in a normal run the function-call handoff always fires first and
# this stays a rare safety net instead of a second, competing path.
STAGE1_TIMEOUT_SECS = float(os.getenv("STAGE1_TIMEOUT_SECS", "300"))

# Once the timeout elapses, how often to re-check whether it's a safe moment
# to force the handoff (candidate not mid-answer, Andrew not mid-reply).
_FALLBACK_QUIET_POLL_SECS = 1.0


@dataclass
class InterviewData:
    """Shared state carried across the rapport -> experience handoff."""

    candidate_name: str | None = None
    intro_notes: str | None = None
    interview_ending: bool = False


class RapportAgent(Agent):
    """Stage 1 of the interview: warm up the candidate, no resume involved."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Andrew. You are a warm, friendly interviewer conducting "
                "the opening, rapport-building part of a mock interview. If asked your "
                "name, you are Andrew - never introduce yourself as anyone else. Speak "
                "briefly and conversationally, this is voice, not text. Never recap or summarize "
                "what the candidate just said back to them - acknowledge it in a few "
                "words (e.g. \"Got it,\" \"That makes sense,\" \"Interesting -\") and "
                "move straight to the next question.\n\n"
                "1. Greet the candidate and ask them to tell you about their background.\n"
                "2. After they respond, ask EXACTLY ONE follow-up question based ONLY on "
                "what they just said. Do not invent details or reference a resume. This "
                "is your ONLY follow-up - once you've asked it, you are DONE asking "
                "questions in this stage.\n"
                "3. As soon as the candidate answers that one follow-up, STOP asking "
                "questions - do NOT ask a second follow-up, a clarifying question, or "
                "any other question, even if their answer seems to invite one. Instead, "
                "silently move the interview forward to the next part. This happens as a "
                "background action, not as something you say - do not announce or "
                "narrate the transition yourself in words.\n\n"
                "Your spoken replies should only ever be natural interview conversation. "
                "Never write out code, JSON, or any programming syntax in your reply."
            )
        )
        # Guards the rapport -> experience handoff so it only ever fires
        # once, whichever path (tool call or timeout) gets there first.
        self._handed_off = False
        self._fallback_timer: asyncio.Task[None] | None = None

    async def on_enter(self) -> None:
        logger.info("========== STAGE 1: RAPPORT (background + one follow-up) ==========")
        self.session.generate_reply(
            instructions=(
                "Greet the candidate warmly, introduce yourself as Andrew, their "
                "interviewer, and ask them to tell you about their background."
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

        # Don't force the handoff mid-answer or mid-reply - once the timeout
        # has elapsed, wait for a quiet moment instead of cutting anyone off.
        # Re-checked every poll in case intro_complete fires while waiting.
        while not self._handed_off and (
            self.session.user_state == "speaking"
            or self.session.agent_state in ("speaking", "thinking")
        ):
            await asyncio.sleep(_FALLBACK_QUIET_POLL_SECS)
        if self._handed_off:
            return  # normal handoff fired while we were waiting for quiet

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
        """Belt-and-suspenders guard alongside the instructions above: strip a
        leaked `intro_complete` call (and any wrapper around it) out of the
        text stream before it reaches the transcript or TTS. See
        _filter_leaked_tool_calls() for the buffering logic - shared with
        ExperienceAgent.llm_node so both stay in sync.
        """
        async for chunk in _filter_leaked_tool_calls(
            Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        ):
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
                "You are Andrew, the SAME interviewer as before, now moving into the "
                "experience deep-dive. If asked your name, you are Andrew - never "
                "introduce yourself as anyone else. Here is what you learned about the candidate "
                f"in the intro, for your own context only: {intro_notes}\n\n"
                "Do not re-greet the candidate and do not repeat earlier questions. Pick "
                "up naturally from the transition, however it happened - never mention "
                "or narrate a stage change, a function/tool call, a timer, an error, or "
                "that anything unusual occurred (e.g. never say things like \"it looks "
                "like this part is complete\" or \"I made an error\"). If anything about "
                "the transition seems unclear, just move forward naturally as if it "
                "wasn't. Run the deep-dive as follows:\n\n"
                "1. Ask the candidate to pick one project or role they'd like to talk "
                "about.\n"
                "2. Ask about it across exactly THREE questions, one at a time, roughly "
                "covering: the problem they were solving, how they approached it, and "
                "the outcome. Keep each reply short and conversational. Never recap or "
                "summarize what the candidate just said back to them - acknowledge it in "
                "a few words (e.g. \"Got it,\" \"That makes sense,\" \"Interesting -\") and "
                "move straight to the next question.\n"
                "3. As soon as the candidate answers your THIRD question, wrap up the "
                "interview on your own - do not wait for them to say they're done or to "
                "say \"thank you,\" and do not ask a fourth question even if there's more "
                "you could explore.\n\n"
                "Once the candidate has answered your third question, wrap up the "
                "interview in your final reply, in two short parts: first, ONE brief, "
                "warm sentence reacting to what they just said (e.g. \"That's a really "
                "thoughtful point,\" or \"Great, that gives me a good sense of your "
                "approach,\") - not a summary, just a quick natural acknowledgment. Then "
                "immediately follow with the closing line: \"That's all my questions. "
                "Thank you so much for your time today - this concludes our interview, "
                "and best of luck.\" Do not ask any further questions after that, even if "
                "the candidate thanks you or keeps talking. Ending the interview itself "
                "happens as a background action - you don't need to mention a function, "
                "tool, or action, just speak the acknowledgment and closing naturally as "
                "your last turn.\n\n"
                "Your spoken replies should only ever be natural interview conversation. "
                "Never write out code, JSON, or any programming syntax in your reply."
            )
        )

    async def on_enter(self) -> None:
        logger.info("========== STAGE 2: EXPERIENCE (project deep-dive) ==========")
        # Explicit instructions here (not just relying on chat history) matter
        # for the fallback-timer path: intro_complete's tool call inserts a
        # natural transition line into the chat context before this runs, but
        # session.update_agent() from the timeout path does not - without a
        # clear instruction, the model has been seen narrating confusion
        # about the transition instead of just continuing.
        self.session.generate_reply(
            instructions=(
                "Continue the interview smoothly from here - do not restart, "
                "re-greet, or repeat earlier questions, and do not mention or "
                "narrate a stage change, a function/tool call, a timer, or an "
                "error. Briefly acknowledge the conversation so far in a few "
                "natural words, then ask the candidate to pick a project or "
                "role they'd like to talk about."
            )
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        """Same belt-and-suspenders guard as RapportAgent.llm_node, for a
        leaked `interview_complete` call - see _filter_leaked_tool_calls()
        for the shared buffering logic.
        """
        async for chunk in _filter_leaked_tool_calls(
            Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        ):
            yield chunk

    @function_tool
    async def interview_complete(self, context: RunContext[InterviewData]) -> str:
        """Call this once the candidate has answered your question about their
        chosen project. This ends the interview - do not call it before they've
        answered, and do not call it more than once.
        """
        context.userdata.interview_ending = True
        logger.info("interview complete")
        return (
            "Wrap up now: thank the candidate warmly for their time and interest, "
            "wish them well, and end on a positive note. Do not ask any more questions."
        )


async def entrypoint(ctx: JobContext) -> None:
    userdata = InterviewData()
    session = AgentSession[InterviewData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=deepgram.TTS(model="aura-2-orion-en"),
        vad=silero.VAD.load(),
    )

    # Lightweight latency logging now so we have a number to point at later
    # when tuning for the <1.5s reply target.
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    # ExperienceAgent.interview_complete() sets userdata.interview_ending and
    # triggers a closing reply, but that reply still needs to finish playing
    # before we actually end the job. "speaking" -> anything else is the
    # signal that the closing statement has finished, so this is the one
    # spot that's actually safe to shut down from.
    shutdown_requested = False

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
        nonlocal shutdown_requested
        if shutdown_requested or not userdata.interview_ending:
            return
        if ev.old_state == "speaking":
            shutdown_requested = True
            logger.info("closing statement finished playing - ending session")
            ctx.shutdown(reason="interview complete")

    # Connect before starting the session: RapportAgent's on_enter() fires a
    # greeting as soon as the session starts, so the room needs to already be
    # connected for that first reply to have somewhere to go.
    await ctx.connect()

    # Tavus renders the avatar as its own room participant and plays the
    # audio itself, so the agent's own audio output must be disabled below
    # (room_output_options) to avoid double-playback.
    avatar = tavus.AvatarSession(
        replica_id=os.getenv("TAVUS_REPLICA_ID"),
        # persona_id omitted on purpose - let the plugin use its default LiveKit PAL
    )
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=RapportAgent(),
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(audio_enabled=False),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))