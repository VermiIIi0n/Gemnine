# Simple Wrapper For The Gemini AI API

This module is designed for asynchronous usage and provides a simple interface to the Google Gemini API.

It also provides a command-line interface for interactive usage.

While it does provide synchronous methods, it is recommended to use the asynchronous methods for better performance.

## **NOTICE**

Images are not tested and may not work as expected.

## Installation

```bash
pip install -U gemnine
```

## Usage

### Command Line

#### Configuration

`gemnine` requires a model and an API key to be set.

You can use config file, command parameters or interactive input to set these values.

By default, `gemnine` will look for files named `config.json` and `session.json` in the user cache directory.

```bash
gemnine -c /path/to/config.json  # Use a config file
gemnine -m "models/gemini-pro"  # Set model, input API key interactively
gemnine -l  # list available models
```

#### Parameters

- `-c`, `--config`: Path to a JSON file
- `-m`, `--model`: Model name
- `-s`, `--session`: Session history file
- `-V`, `--version`: Show version

**_Precedence_**: Interactive input > Command parameters > Config file

#### Interactive Mode

This mode mimics a chat interface. You can type your message and get a response.

Commands are started with a `/` and are **case-sensitive**.

- `/exit`: Exit the program
- `/save <path>`: Save the session history to a file
- `/load`: Load a session history from a file
- `/rollback <step>`: Rollback to the previous state
- `/clear`: Clear the session history
- `/role`: Switch role
- `/model`: Switch model
- `/help`: Show help

To embed an image, insert `#image(<url>)` in your message.
Use double `#` to escape embedding.

```Python
await bot.send("""
What are these?
#image(https://example.com/image.png)
#image(file:///path/to/image.png)
#image(base64://<base64-encoded-image>)
""")
```

All URLs will be downloaded and embedded as base64-encoded images.

### API

```python
import asyncio
import gemnine

bot = gemnine.Bot(model="models/gemini-pro", api_key="your-api-key")

async def main():
    response = await bot.send("Hello, how are you?")
    print(response)

    async for r in bot.stream("I'm fine, thank you."):
        print(r, end='')

asyncio.run(main())
```
