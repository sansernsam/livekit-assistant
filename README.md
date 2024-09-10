# LiveKit Assistant

This project implements a voice and vision-capable assistant using LiveKit.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/livekit-assistant.git
   cd livekit-assistant
   ```

2. Create a virtual environment and activate it:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Update pip and install dependencies:
   ```
   pip install -U pip
   pip install -r requirements.txt
   ```

4. Set up your `.env` file with the following environment variables:
   ```
   LIVEKIT_URL=...
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   DEEPGRAM_API_KEY=...
   OPENAI_API_KEY=...
   ```

## Running the Assistant

1. Download necessary files:
   ```
   python3 assistant.py download-files
   ```

2. Start the assistant:
   ```
   python3 assistant.py start
   ```

3. Connect to the assistant using the [hosted playground](https://agents-playground.livekit.io/).

## Features

- Voice interaction
- Image processing capabilities
- Thai language support

## Deployment

This project is set up for deployment on Render.com. See the Procfile for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

If you encounter any issues, please check that all environment variables are correctly set and that you have the latest versions of all dependencies.
