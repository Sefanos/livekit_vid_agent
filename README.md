# LiveKit Assistant

LiveKit Assistant is a Python application that enables real-time interaction with a Large Language Model (LLM). It allows users to communicate with the LLM via speech, receive spoken responses, and integrate video capabilities using LiveKit.

## Installation and Setup

### 1. Create a Virtual Environment
To ensure a clean environment, create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Update Pip and Install Dependencies
Upgrade `pip` and install the required packages:

```sh
pip install -U pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Before running the assistant, set up the required environment variables:

```sh
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

Make sure to replace `...` with the actual values for each variable.

## Running the Assistant

### Step 1: Download Required Files
Run the following command to download any necessary resources:

```sh
python3 assistant.py download-files
```

### Step 2: Start the Assistant
Launch the assistant with:

```sh
python3 assistant.py start
```

## Features
- **Voice Interaction:** Speak to the LLM and receive spoken responses.
- **LLM Integration:** Utilize OpenAI's API to generate responses.
- **Live Video Support:** Leverages LiveKit for real-time video streaming.

## Testing the UI
To test the assistant using the UI, refer to:
- [LiveKit Agents Playground Repository](https://github.com/livekit/agents-playground)
- [LiveKit Agents Playground UI](https://agents-playground.livekit.io/#cam=1&mic=1&video=1&audio=1&chat=1&theme_color=cyan)

## Notes
- Ensure all required API keys and credentials are correctly configured.
- The assistant requires an internet connection to interact with the LLM and LiveKit services.

Feel free to contribute or report issues! 🚀

