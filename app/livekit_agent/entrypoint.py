import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import deepgram, elevenlabs

from app.livekit_agent.workers import AGENT_NAME
from app.livekit_agent.livekit_llm import LiveKitLLM

load_dotenv()

server = AgentServer()

@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(),
        llm=LiveKitLLM(),
        tts=elevenlabs.TTS(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="You are a helpful voice AI assistant."
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)