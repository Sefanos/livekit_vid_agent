import asyncio
from typing import Annotated, Dict
import time

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pre-load the VAD model to improve startup time
print("Pre-loading Silero VAD model...")
vad_model = silero.VAD.load()
print("VAD model loaded")


class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

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
        print(f"Message triggering vision capabilities: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""
    print("Looking for video tracks...")
    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break
                
    # Check if a track was found
    if not video_track.done():
        print("No video track found, setting up listener for new tracks")
        # Create a listener for new tracks
        def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if (publication.track is not None and 
                isinstance(publication.track, rtc.RemoteVideoTrack) and
                not video_track.done()):
                video_track.set_result(publication.track)
                print(f"Using newly published video track {publication.track.sid}")
        
        # Add listener for new tracks
        for _, participant in room.remote_participants.items():
            participant.on("track_published", on_track_published)
        
        # Also listen for future track publications
        room.on("track_published", on_track_published)

    return await video_track


# Create a video processor that updates a shared state dictionary - now with better performance
async def process_video(room: rtc.Room, state: Dict):
    """Process video frames in a separate task"""
    while room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        try:
            video_track = await get_video_track(room)
            print("Got video track, starting video stream")
            
            # Process video frames with rate limiting
            async for event in rtc.VideoStream(video_track):
                current_time = time.time()
                # Only process frames at the specified interval
                if current_time - state["last_frame_time"] >= state["frame_interval"]:
                    state["latest_image"] = event.frame
                    state["has_valid_image"] = True
                    state["last_frame_time"] = current_time
                    
                    # Only print debug info occasionally to reduce console spam
                    if int(current_time) % 5 == 0:  # Only print every ~5 seconds
                        print(f"Captured new video frame at {current_time:.2f}")
                    
                # Critical fix: Add a small delay to prevent video processing from monopolizing the CPU
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            print("Video processing cancelled")
            break
        except Exception as e:
            print(f"Error processing video: {e}")
            state["has_valid_image"] = False
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)  # Wait before retrying


async def entrypoint(ctx: JobContext):
    print("Starting assistant initialization...")
    start_time = time.time()
    
    await ctx.connect()
    print(f"Room name: {ctx.room.name}, connected in {time.time() - start_time:.2f}s")

    # Create a shared state dictionary for video processing
    video_state = {
        "latest_image": None,
        "last_frame_time": 0,
        "frame_interval": 1.0,  # Increased to 1.0 seconds to reduce resource usage
        "has_valid_image": False,  # Flag to track if we have a valid image
        "processing_response": False  # Flag to prevent multiple simultaneous responses
    }

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a funny, witty bot with visual capabilities. "
                    "You can see and describe what's happening in the video feed. "
                    "When users ask you about what you can see, provide detailed descriptions. "
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    # Initialize components with performance optimization in mind
    print("Setting up LLM and TTS...")
    gpt = openai.LLM(model="gpt-4o", timeout=30)  # Add timeout to prevent hanging

    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    print("Setting up voice assistant...")
    assistant = VoiceAssistant(
        vad=vad_model,  # Use the pre-loaded VAD model
        stt=deepgram.STT(),  
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )
    
    # Track ongoing tasks to prevent multiple simultaneous responses
    response_lock = asyncio.Lock()
    
    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        # Use a lock to prevent multiple simultaneous responses
        async with response_lock:
            try:
                print(f"Answering: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                # Set flag to indicate we're processing a response
                video_state["processing_response"] = True
                
                content: list[str | ChatImage] = [text]
                
                # Check if we're supposed to use the image and if we have a valid image to use
                if use_image or any(keyword in text.lower() for keyword in ["see", "look", "show", "camera"]):
                    if video_state["has_valid_image"] and video_state["latest_image"]:
                        print("Including image in request")
                        content.append(ChatImage(image=video_state["latest_image"]))
                    else:
                        print("No valid image available to include")
                        content.append("I'm supposed to look at something, but I don't have access to the camera feed right now.")

                chat_context.messages.append(ChatMessage(role="user", content=content))

                # Fixed: Don't await the gpt.chat() call inside wait_for
                try:
                    # Get the stream directly - it's not awaitable
                    stream = gpt.chat(chat_ctx=chat_context)
                    
                    # Create a future to track completion
                    completion_future = asyncio.Future()
                    
                    # Start saying the response in a separate task
                    say_task = asyncio.create_task(assistant.say(stream, allow_interruptions=True))
                    
                    # Add a done callback to resolve the future
                    def on_say_done(task):
                        if not completion_future.done():
                            if task.exception():
                                completion_future.set_exception(task.exception())
                            else:
                                completion_future.set_result(task.result())
                    
                    say_task.add_done_callback(on_say_done)
                    
                    # Wait for the say task with timeout
                    try:
                        await asyncio.wait_for(completion_future, 15.0)
                    except asyncio.TimeoutError:
                        print("Response generation timed out")
                        # Don't cancel say_task - let it continue in the background
                        # Just exit this function to allow new requests
                except Exception as e:
                    print(f"Error generating response: {e}")
                    await assistant.say("I'm having trouble processing that. Let me try again.", allow_interruptions=True)
            finally:
                # Clear the processing flag
                video_state["processing_response"] = False

    # Handle data messages
    @ctx.room.on("data_received")
    def on_data_received(packet):
        """Handle data messages received from participants."""
        try:
            # If already processing a response, ignore new requests
            if video_state["processing_response"]:
                print("Already processing a response, ignoring new data message")
                return
                
            data = packet.data
            message = data.decode('utf-8')
            print(f"Received message: {message}")
            
            if message:
                # Determine if this message is likely asking about the video feed
                vision_keywords = ["see", "look", "show", "camera", "watching", "screen", "image", "what's there"]
                use_vision = any(keyword in message.lower() for keyword in vision_keywords)
                
                asyncio.create_task(_answer(message, use_image=use_vision))
        except Exception as e:
            print(f"Error processing data message: {e}")
            import traceback
            traceback.print_exc()

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""
        if len(called_functions) == 0:
            return

        # If already processing a response, ignore new function calls
        if video_state["processing_response"]:
            print("Already processing a response, ignoring function call")
            return
            
        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            # Always include image when the image function is called
            asyncio.create_task(_answer(user_msg, use_image=True))
    
    # Define a synchronous dispatcher and async handler for RPC operations
    async def handle_rpc_operations(participant, rpc):
        try:
            print(f"Processing RPC: {rpc.method}")
            
            # If already processing a response and this is an interactive RPC, ignore it
            if video_state["processing_response"] and rpc.method == "describe_image":
                return "Currently busy processing another response. Please try again in a moment."
                
            # Handle different RPC methods
            if rpc.method == "change_voice":
                voice = rpc.payload or "alloy"
                nonlocal openai_tts
                openai_tts = tts.StreamAdapter(
                    tts=openai.TTS(voice=voice),
                    sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
                )
                assistant.tts = openai_tts
                return f"Voice changed to {voice}"
                
            elif rpc.method == "get_info":
                return {
                    "assistant_name": "Alloy",
                    "model": "gpt-4o",
                    "voice": "alloy",
                    "has_image": video_state["has_valid_image"],
                    "busy": video_state["processing_response"]
                }
                
            elif rpc.method == "describe_image":
                if video_state["latest_image"] and video_state["has_valid_image"]:
                    asyncio.create_task(_answer("Please describe what you can see in the camera feed", use_image=True))
                    return "Describing current video feed"
                else:
                    return "No image available to describe"
                
            else:
                return f"Unknown method: {rpc.method}"
                
        except Exception as e:
            print(f"Error handling RPC: {e}")
            return f"Error: {str(e)}"
            
    @ctx.room.on("participant_rpc")
    def on_rpc(participant: rtc.RemoteParticipant, rpc):
        print(f"Received RPC from {participant.identity}: {rpc.method}")
        # Create an async task to handle the RPC operation
        asyncio.create_task(handle_rpc_operations(participant, rpc))

    print("Starting assistant...")
    assistant.start(ctx.room)

    print("Setup completed in {:.2f}s".format(time.time() - start_time))
    await asyncio.sleep(1)
    await assistant.say("Hi there! I can see and hear you. How can I help?", allow_interruptions=True)

    # Start video processing in a separate task to not block the main flow
    video_task = asyncio.create_task(process_video(ctx.room, video_state))
    
    # Keep the main task running as long as we're connected
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)
    
    # Cancel video processing when we're done
    if not video_task.done():
        video_task.cancel()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
