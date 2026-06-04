"""
title: Local Vision Bridge
author: feliscat
upstream_repository_url: https://github.com/feliscat/local-vision-bridge
maintainer: ykpaoschool
repository_url: https://github.com/ykpaoschool/local-vision-bridge
version: 0.2.0
description: Scans chat for images, caches OCR/captions to avoid re-processing, and injects the results as system output.
"""

import hashlib
import json
import os
import textwrap
import requests
from pydantic import BaseModel, Field
from typing import Optional, List, Union

class Filter:
    class Valves(BaseModel):
        vision_backend: str = Field(
            default="openai_compatible",
            description="Backend type: openai_compatible or azure_openai",
        )
        vision_server_url: str = Field(
            default="http://host.docker.internal:8081/v1/chat/completions",
            description="URL of the OpenAI-compatible vision endpoint",
        )
        vision_model: str = Field(
            default="qwen2.5-vl-7b",
            description="Model name for openai_compatible backends",
        )
        azure_openai_endpoint: str = Field(
            default="",
            description="Azure OpenAI endpoint, e.g. https://YOUR_RESOURCE.openai.azure.com",
        )
        azure_openai_deployment: str = Field(
            default="",
            description="Azure OpenAI deployment name, e.g. gpt-5-mini",
        )
        azure_openai_api_version: str = Field(
            default="2024-10-21",
            description="Azure OpenAI API version",
        )
        azure_openai_max_completion_tokens: int = Field(
            default=2048,
            description="Max completion tokens for Azure OpenAI vision requests",
        )
        azure_openai_api_key: str = Field(
            default="",
            description="Azure OpenAI API key; falls back to AZURE_OPENAI_API_KEY",
        )
        azure_openai_auth_token: str = Field(
            default="",
            description="Microsoft Entra bearer token; falls back to AZURE_OPENAI_AUTH_TOKEN",
        )
        vision_prompt: str = Field(
            default=textwrap.dedent(
                """
                # Vision Task

                Analyze this image and produce an output that follows these rules exactly.

                ## 1. Output language
                - Use the language of the visible text in the image if there is a clear dominant language.
                - If there is little or no visible text, or no single dominant language, use the user's language.

                ## 2. Decide image type
                A text-centric image is a document page, slide, poster, screenshot, form, table, chart with substantial labels, book page, receipt, whiteboard, handwritten note, or any image where reading the text is the main task.

                A non-text-centric image is a photo, illustration, painting, scene, object-focused image, person-focused image, interface snapshot with little text, or any image where visual content is the main task.

                ## 3. If the image is text-centric
                - Transcribe the visible text as completely and faithfully as possible.
                - Preserve document structure using Markdown.
                - Use headings, lists, tables, and paragraphs when they are visually justified.
                - Quote text exactly as it appears, preserving wording, capitalization, punctuation, and line order when important.
                - If mathematical expressions or formulas are present, write them in LaTeX.
                - Do not summarize when full transcription is feasible.
                - If some text is unreadable, mark it clearly as [illegible].
                - If non-text visual elements matter, briefly note them after the transcription.

                ## 4. If the image is non-text-centric
                - Write a concise, neutral, factual description using the 5W method when applicable.
                - Cover who is present, what is happening, when is visible or inferable only from explicit evidence such as a timestamp, where is visible from the scene, and why only if directly shown by explicit text or universally obvious function.
                - Focus on observable subjects, objects, actions, layout, colors, shapes, textures, and spatial relationships.
                - Use definite, objective language.
                - Do not speculate about mood, intent, identity, cause, or anything not directly visible.
                - If visible text exists, quote it exactly.
                - If a watermark, signature, timestamp, or compression artifact is visible, mention it briefly.

                ## 5. General rules
                - Do not mention what is absent.
                - Do not mention image resolution or technical metadata unless it is itself visible in the image.
                - Do not begin with phrases like "This image is".
                - Be precise, concrete, and concise.
                """
            ).strip(),
            description="Prompt for Vision Model to generate caption",
        )
        debug_mode: bool = Field(
            default=True, description="Print debug logs to console"
        )

    def __init__(self):
        self.valves = self.Valves()
        # Simple in-memory cache: { "hash_of_image": "Description string" }
        self.cache = {}

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        if self.valves.debug_mode:
            print(f"[Vision Bridge] Scanning history ({len(messages)} msgs)...")

        images_processed_total = 0
        cache_hits = 0

        for i, message in enumerate(messages):
            if "content" not in message:
                continue

            images_found = []

            # images (OpenAI Format)
            content = message.get("content", "")
            if isinstance(content, list):
                new_content_list = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if "base64," in url:
                            b64 = url.split("base64,")[1]
                            images_found.append(b64)
                    else:
                        new_content_list.append(item)
                message["content"] = new_content_list

            # images (Legacy Format)
            if "images" in message and message["images"]:
                images_found.extend(message["images"])
                message["images"] = []

            # cache
            if images_found:
                descriptions = []
                for idx, img_b64 in enumerate(images_found):
                    # Generate a unique hash for this image
                    img_hash = hashlib.md5(img_b64.encode()).hexdigest()

                    if img_hash in self.cache:
                        # HIT: Use stored description
                        desc = self.cache[img_hash]
                        cache_hits += 1
                    else:
                        # MISS: Ask model
                        if self.valves.debug_mode:
                            print(f"[Vision Bridge] Cache Miss. Sending to GPU...")
                        desc = self._get_description(img_b64)
                        self.cache[img_hash] = desc
                        images_processed_total += 1

                    descriptions.append(f"IMAGE {idx+1}: {desc}")

                vision_block = "\n".join(descriptions)

                # Reassemble Text
                existing_text = ""
                if isinstance(message["content"], list):
                    for item in message["content"]:
                        if item.get("type") == "text":
                            existing_text += item["text"] + "\n"
                else:
                    existing_text = str(message["content"])

                # Inject System Block
                final_text = (
                    f"--- [SYSTEM TOOL OUTPUT: VISUAL CORTEX] ---\n"
                    f"{vision_block}\n"
                    f"--- [END TOOL OUTPUT] ---\n\n"
                    f"{existing_text}"
                )

                message["content"] = final_text

                # get rid of images in the message before passing to text llm
                if "images" in message:
                    del message["images"]

        if self.valves.debug_mode:
            print(
                f"[Vision Bridge] Done. New: {images_processed_total}, Cached: {cache_hits}"
            )

        return body

    def _get_description(self, base64_image: str) -> str:
        if self.valves.vision_backend == "azure_openai":
            return self._get_description_from_azure(base64_image)
        return self._get_description_from_openai_compatible(base64_image)

    def _build_vision_user_content(self, base64_image: str) -> list:
        return [
            {
                "type": "text",
                "text": self.valves.vision_prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        ]

    def _post_json(self, url: str, payload: dict, headers: Optional[dict] = None) -> dict:
        effective_headers = headers or {}
        response = requests.post(
            url,
            headers=effective_headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _build_openai_compatible_payload(self, base64_image: str) -> dict:
        return {
            "model": self.valves.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_vision_user_content(base64_image),
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }

    def _build_openai_compatible_headers(self) -> dict:
        return {}

    def _get_description_from_openai_compatible(self, base64_image: str) -> str:
        payload = self._build_openai_compatible_payload(base64_image)
        headers = self._build_openai_compatible_headers()
        try:
            data = self._post_json(self.valves.vision_server_url, payload, headers)
            return self._extract_text_from_response(data)
        except Exception as e:
            return f"(Error reading image: {str(e)})"

    def _build_azure_payload(self, base64_image: str) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": self._build_vision_user_content(base64_image),
                }
            ],
            "max_completion_tokens": self.valves.azure_openai_max_completion_tokens,
        }

    def _build_azure_headers(self, api_key: str, auth_token: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        else:
            headers["api-key"] = api_key
        return headers

    def _get_description_from_azure(self, base64_image: str) -> str:
        endpoint = self.valves.azure_openai_endpoint.rstrip("/")
        deployment = self.valves.azure_openai_deployment
        api_version = self.valves.azure_openai_api_version
        api_key = self.valves.azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        auth_token = self.valves.azure_openai_auth_token or os.getenv(
            "AZURE_OPENAI_AUTH_TOKEN", ""
        )

        if not endpoint or not deployment:
            return "(Error reading image: Azure endpoint or deployment is not configured)"

        if not api_key and not auth_token:
            return "(Error reading image: Azure API key or auth token is not configured)"

        payload = self._build_azure_payload(base64_image)

        url = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )
        headers = self._build_azure_headers(api_key, auth_token)

        try:
            data = self._post_json(url, payload, headers)
            return self._extract_text_from_response(data)
        except Exception as e:
            return f"(Error reading image: {str(e)})"

    def _extract_text_from_response(self, data: dict) -> str:
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and item.get("text"):
                        text_parts.append(item["text"])
                    elif item.get("type") == "output_text" and item.get("text"):
                        text_parts.append(item["text"])
                if text_parts:
                    return "\n".join(text_parts).strip()

            refusal = message.get("refusal")
            finish_reason = choice.get("finish_reason")

            diagnostics = []
            if refusal:
                diagnostics.append(f"refusal={refusal}")
            if finish_reason:
                diagnostics.append(f"finish_reason={finish_reason}")

            content_filter_results = choice.get("content_filter_results")
            if content_filter_results:
                diagnostics.append(
                    "content_filter_results="
                    + json.dumps(content_filter_results, ensure_ascii=True)
                )

            if self.valves.debug_mode:
                print(
                    "[Vision Bridge] Empty model response: "
                    + json.dumps(data, ensure_ascii=True)
                )

            if diagnostics:
                return "(Image description was empty: " + "; ".join(diagnostics) + ")"

            return "(Image description was empty: no text returned by model)"
        except Exception as e:
            return f"(Error parsing model response: {str(e)})"
