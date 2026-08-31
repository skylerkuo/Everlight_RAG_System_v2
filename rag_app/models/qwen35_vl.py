from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class Qwen35VL:
    """Hugging Face Transformers inference for Qwen/Qwen3.5-4B."""

    def __init__(
        self,
        model_id: str,
        image_scale: float,
    ) -> None:
        self.model_id = model_id
        self.image_scale = image_scale

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
        )

        self.model.eval()

    def _resize_image(
        self,
        image_path: str | Path,
        output_dir: Path,
    ) -> Path:
        image_path = Path(image_path)

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            new_width = max(1, int(img.width * self.image_scale))
            new_height = max(1, int(img.height * self.image_scale))

            img = img.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            output_path = output_dir / f"{image_path.stem}_resized.jpg"

            img.save(
                output_path,
                format="JPEG",
                quality=90,
            )

        return output_path

    def _get_eos_token_ids(self) -> set[int]:
        """
        Collect possible EOS token IDs from model generation config
        and tokenizer.
        """
        eos_ids: set[int] = set()

        generation_eos = getattr(
            self.model.generation_config,
            "eos_token_id",
            None,
        )

        if generation_eos is not None:
            if isinstance(generation_eos, int):
                eos_ids.add(generation_eos)
            else:
                eos_ids.update(int(x) for x in generation_eos)

        tokenizer = getattr(self.processor, "tokenizer", None)

        if tokenizer is not None:
            tokenizer_eos = getattr(
                tokenizer,
                "eos_token_id",
                None,
            )

            if tokenizer_eos is not None:
                eos_ids.add(int(tokenizer_eos))

        return eos_ids

    def _clear_cuda_cache(self) -> None:
        """
        Release Python references / CUDA cached memory before fallback.
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] | None = None,
        image_labels: Sequence[str] | None = None,
        system: str | None = None,
        max_new_tokens: int = 1024,
        enable_thinking: bool = False,
    ) -> str:

        paths = list(image_paths or [])

        if image_labels is not None and len(image_labels) != len(paths):
            raise ValueError(
                "image_labels must have the same length as image_paths"
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            resized_paths = [
                self._resize_image(path, tmp_dir)
                for path in paths
            ]

            # ---------------------------------------------------------
            # Build messages.
            # The message content itself is identical between thinking
            # and non-thinking. enable_thinking is controlled when
            # apply_chat_template() is called.
            # ---------------------------------------------------------
            messages: list[dict] = []

            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system,
                            }
                        ],
                    }
                )

            content: list[dict] = []

            for i, path in enumerate(resized_paths):

                if image_labels is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": image_labels[i],
                        }
                    )

                content.append(
                    {
                        "type": "image",
                        "path": str(path.resolve()),
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

            eos_token_ids = self._get_eos_token_ids()

            # ---------------------------------------------------------
            # Run one generation.
            # ---------------------------------------------------------
            def run_once(
                thinking: bool,
            ) -> tuple[str, int, bool]:

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=thinking,
                ).to(self.model.device)

                input_length = inputs["input_ids"].shape[-1]

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                generated_ids = output_ids[
                    0,
                    input_length:
                ]

                generated_token_count = int(
                    generated_ids.shape[-1]
                )

                last_token_id: int | None = None

                if generated_token_count > 0:
                    last_token_id = int(
                        generated_ids[-1].item()
                    )

                ended_with_eos = (
                    last_token_id is not None
                    and last_token_id in eos_token_ids
                )

                # If max_new_tokens was exhausted and the final token
                # was not EOS, generation was most likely truncated.
                hit_max_tokens = (
                    generated_token_count >= max_new_tokens
                    and not ended_with_eos
                )

                decoded_text = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

                # Qwen thinking mode may return:
                #
                # reasoning text...
                # </think>
                # final answer...
                #
                # Return only the final answer.
                if thinking and "</think>" in decoded_text:
                    decoded_text = decoded_text.rsplit(
                        "</think>",
                        1,
                    )[-1].strip()

                # Explicitly release the large generation tensors
                # before returning / retrying.
                del generated_ids
                del output_ids
                del inputs

                return (
                    decoded_text,
                    generated_token_count,
                    hit_max_tokens,
                )

            # =========================================================
            # First attempt
            # =========================================================
            text, token_count, hit_max_tokens = run_once(
                thinking=enable_thinking,
            )

            # ---------------------------------------------------------
            # Normal path:
            #
            # 1. Non-thinking calls behave exactly as before.
            # 2. Thinking completed before max_new_tokens.
            # ---------------------------------------------------------
            if not enable_thinking:
                return text

            if not hit_max_tokens:
                return text

            # =========================================================
            # Thinking reached max_new_tokens before EOS.
            #
            # Discard the entire thinking result and retry using
            # non-thinking mode.
            # =========================================================
            print(
                "[WARN] Thinking generation reached "
                f"max_new_tokens={max_new_tokens} "
                f"(generated={token_count})."
            )

            print(
                "[WARN] Discarding the thinking result and "
                "retrying with enable_thinking=False."
            )

            # Drop reference to the rejected answer as well.
            del text

            self._clear_cuda_cache()

            # =========================================================
            # Fallback:
            # same prompt
            # same images
            # same system prompt
            # same max_new_tokens
            # only thinking is disabled
            # =========================================================
            fallback_text, fallback_token_count, _ = run_once(
                thinking=False,
            )

            print(
                "[INFO] Non-thinking fallback completed. "
                f"generated_tokens={fallback_token_count}"
            )

            return fallback_text