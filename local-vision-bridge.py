"""
title: Local Vision Bridge
author: feliscat
description: Scans chat for images, caches descriptions to avoid re-processing, and injects them as system output.
"""

import requests
import hashlib
from pydantic import BaseModel, Field
from typing import Optional, List, Union

class Filter:
    class Valves(BaseModel):
        vision_server_url: str = Field(
            default="http://host.docker.internal:8081/v1/chat/completions",
            description="URL of the local vision llm",
        )
        vision_model: str = Field(
            default="qwen2.5-vl-7b",
            description="Model name",
        )
        vision_prompt: str = Field(
            default='Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what\'s absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.',
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
        payload = {
            "model": self.valves.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
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
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        try:
            response = requests.post(
                self.valves.vision_server_url, json=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"(Error reading image: {str(e)})"
