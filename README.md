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
* **Azure OpenAI Ready:** Can call a GPT-5 deployment on Azure AI Foundry / Azure OpenAI for image understanding.
* **Balanced Default Prompt:** Transcribes text-centric images in Markdown, renders formulas as LaTeX, and uses neutral 5W descriptions for non-text-centric images.

## How to use

1. In OpenWebUI, go to `Admin Panel` -> `Functions`.
2. Create a new function and paste the contents of `local-vision-bridge.py`.
3. Enable the function for the models or chats where you want image-to-text bridging.
4. If you are using a local OpenAI-compatible vision endpoint, set `vision_backend` to `openai_compatible`, then configure `vision_server_url` and `vision_model`.
5. If you are using Azure OpenAI, set `vision_backend` to `azure_openai` and fill in the Azure-specific valves listed below.
6. Upload an image in chat. The function will send the image to the configured vision model, cache the result, and inject the OCR/caption output back into the conversation as system tool output.
7. For text-only models, keep the function enabled. For native multimodal models, disable it unless you explicitly want the image converted into text before inference.

## Azure AI Foundry / Azure OpenAI Setup

To use a `gpt-5-mini` deployment hosted in Azure, set these valves in OpenWebUI:

* `vision_backend`: `azure_openai`
* `azure_openai_endpoint`: your Azure OpenAI endpoint, for example `https://YOUR_RESOURCE.openai.azure.com`
* `azure_openai_deployment`: your deployment name, for example `gpt-5-mini`
* `azure_openai_api_version`: API version, default is `2024-10-21`
* `azure_openai_api_key`: your API key

If you prefer Microsoft Entra auth instead of an API key, leave `azure_openai_api_key` empty and provide `azure_openai_auth_token` or the `AZURE_OPENAI_AUTH_TOKEN` environment variable.

When `vision_backend` stays as `openai_compatible`, the function behaves like the original version and sends images to `vision_server_url`.
