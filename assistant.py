from dotenv import load_dotenv
import os
import asyncio
from typing import Annotated
from PIL import ImageGrab
import pytesseract
from deep_translator import GoogleTranslator

load_dotenv()  # Load environment variables from .env file

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

class EnhancedAssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    def __init__(self):
        super().__init__()
        self.latest_image = None

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        try:
            print(f"Message triggering vision capabilities: {user_msg}")
            if self.latest_image:
                # Convert the frame to a PIL Image if necessary
                pil_image = self.latest_image
                if not isinstance(pil_image, ImageGrab.Image):
                    pil_image = ImageGrab.grab(bbox=self.latest_image.bounds())
                
                # Perform OCR to extract text from the image
                text = pytesseract.image_to_string(pil_image)
                
                # Basic image description
                description = f"I can see an image. Here's what I can tell about it: {text[:100]}..."
                return description
            else:
                return "I'm sorry, but I don't have access to any image at the moment."
        except Exception as e:
            print(f"Error in image processing: {e}")
            return "I encountered an error while trying to process the image. Could you please try again?"

    @agents.llm.ai_callable(
        description="Capture and analyze the current screen content"
    )
    async def capture_screen(self):
        try:
            # Capture the screen
            screenshot = ImageGrab.grab()
            
            # Perform OCR to extract text from the image
            text = pytesseract.image_to_string(screenshot)
            
            # Analyze the extracted text (you may want to use your LLM for this)
            analysis = f"Screen content: {text[:100]}..."  # Truncated for brevity
            
            return analysis
        except Exception as e:
            print(f"Error in screen capture: {e}")
            return "I encountered an error while trying to capture the screen. Could you please try again?"

    @agents.llm.ai_callable(
        description="Handle Thai language input and generate Thai responses"
    )
    async def process_thai(
        self,
        thai_input: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The Thai language input from the user"
            ),
        ],
    ):
        try:
            translator = GoogleTranslator(source='th', target='en')
            
            # Translate Thai to English
            english_input = translator.translate(thai_input)
            
            # Process the English input using your existing pipeline
            english_response = await self.process_input(english_input)
            
            # Translate the response back to Thai
            translator = GoogleTranslator(source='en', target='th')
            thai_response = translator.translate(english_response)
            
            return thai_response
        except Exception as e:
            print(f"Error in Thai processing: {e}")
            return "เกิดข้อผิดพลาดในการประมวลผลภาษาไทย โปรดลองอีกครั้ง"  # "An error occurred in Thai processing. Please try again."

    async def process_input(self, input_text: str) -> str:
        # This is a placeholder for your existing English processing pipeline
        # Replace this with your actual implementation
        return f"Processed: {input_text}"

    def set_latest_image(self, image):
        self.latest_image = image

async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a funny, witty bot. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    # Update the model to use GPT-4 Optimized
    gpt = openai.LLM(model="gpt-4-0613")

    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    enhanced_assistant_function = EnhancedAssistantFunction()

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=enhanced_assistant_function,
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        content: list[str | ChatImage] = [text]
        if use_image and enhanced_assistant_function.latest_image:
            content.append(ChatImage(image=enhanced_assistant_function.latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        for func in called_functions:
            # Safely get the function name
            func_name = getattr(func.call_info, 'function', None)
            if func_name is None:
                func_name = getattr(func.call_info, 'name', 'unknown')

            if func_name == "image":
                user_msg = func.call_info.arguments.get("user_msg") if hasattr(func.call_info, 'arguments') else None
                if user_msg:
                    asyncio.create_task(_answer(user_msg, use_image=True))
            elif func_name == "capture_screen":
                asyncio.create_task(_answer(func.result or "Screen capture failed", use_image=False))
            elif func_name == "process_thai":
                asyncio.create_task(_answer(func.result or "Thai processing failed", use_image=False))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        try:
            video_track = await get_video_track(ctx.room)
            async for event in rtc.VideoStream(video_track):
                enhanced_assistant_function.set_latest_image(event.frame)
        except Exception as e:
            print(f"Error in video processing: {e}")
            await asyncio.sleep(5)  # Wait before retrying

    # Keep the assistant running even if there's an error
    await asyncio.sleep(3600)  # Keep running for an hour, adjust as needed

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))