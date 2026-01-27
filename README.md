# Local Vision Bridge (OpenWebUI Function)

**Give your text-only LLMs the ability to "see" using a secondary local Vision model.**

This OpenWebUI Function intercepts image uploads, sends them to a local Vision-Language Model (like Qwen2.5-VL running on `llama.cpp`), and seamlessly injects detailed text descriptions into the chat context.

This allows you to use massive, high-intelligence text-only models (like large MOE) while still enjoying multi-modal capabilities via a smaller, faster dedicated vision model.

## Features

* **Zero-Latency Caching:** Hashes images so you only pay the "GPU tax" once. Subsequent turns in the chat are instant.
* **History Aware:** Scans the full conversation context to ensure the model doesn't "forget" images in multi-turn chats.
* **Model Agnostic:** Works with any text-only model in OpenWebUI.
* **Universal Compatibility:** Handles both modern OpenAI-format image uploads and legacy/Ollama formats.
* **System Framing:** Injects descriptions as "System Tool Output" so the model knows *it* is seeing the image, rather than thinking the user typed the description.
