from __future__ import annotations

import os
import json
import re
from copy import deepcopy
from typing import Any

import torch
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer


class LocalLLMJudge:
    def __init__(self, model_path: str, device: str, batch_size: int = 1, rank: int = 0) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = max(int(batch_size or 1), 1)
        self.rank = rank

        eval_logger.info(f"开始加载本地 LLM Judge；path={model_path} device={device} rank={rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            model_kwargs.update({"torch_dtype": torch.bfloat16, "device_map": device})
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs).eval()
        if not (torch.cuda.is_available() and str(device).startswith("cuda")):
            self.model.to(device)
        eval_logger.info(f"本地 LLM Judge 加载完成；path={model_path} device={device} rank={rank}")

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_samples(
        self,
        task_name: str,
        samples: list[dict[str, Any]],
        task,
        progress_desc: str | None = None,
    ) -> list[dict[str, Any]]:
        if task_name == "robovqa":
            scorer = self._score_robovqa_batch
        elif task_name == "openeqa-emeqa":
            scorer = self._score_openeqa_batch
        elif task_name == "unieqa":
            scorer = self._score_unieqa_batch
        else:
            raise ValueError(f"Unsupported local LLM Judge task: {task_name}")

        progress_bar = tqdm(
            total=len(samples),
            desc=progress_desc,
            disable=(self.rank != 0),
        )
        scored_samples = []
        for start in range(0, len(samples), self.batch_size):
            batch = samples[start : start + self.batch_size]
            scored_batch = scorer(batch, task)
            scored_samples.extend(scored_batch)
            progress_bar.update(len(batch))
        progress_bar.close()
        return scored_samples

    def _score_robovqa_batch(self, samples: list[dict[str, Any]], task) -> list[dict[str, Any]]:
        self._ensure_openai_import_env()
        from embodied_eval.tasks.robovqa import process as robovqa_process

        contexts = []
        messages_batch = []
        prompt = robovqa_process.load_prompt()
        for sample in samples:
            doc = task.eval_docs[int(sample["doc_id"])]
            prediction = self._extract_prediction(sample)
            contexts.append((sample, doc, prediction))
            messages_batch.append(
                robovqa_process.prepare_openai_messages(
                    prompt.format(
                        question=doc["question"],
                        answer=doc["answer"],
                        prediction=prediction,
                        extra_answers=None,
                    )
                )
            )

        try:
            outputs = self.generate_chat(messages_batch, max_new_tokens=128)
            self._validate_output_count(outputs, contexts)
        except Exception as e:
            return [
                self._make_robovqa_result(sample, doc, 1, str(e))
                for sample, doc, _ in contexts
            ]

        scored_samples = []
        for (sample, doc, _), output in zip(contexts, outputs):
            try:
                score = robovqa_process.parse_score(output)
                if score < 1 or score > 5:
                    raise ValueError(f"Invalid judge score {score} from output: {output!r}")
                score = robovqa_process.clamp_llm_score(score)
                scored_samples.append(self._make_robovqa_result(sample, doc, score, None))
            except Exception as e:
                scored_samples.append(self._make_robovqa_result(sample, doc, 1, str(e)))
        return scored_samples

    def _make_robovqa_result(
        self,
        sample: dict[str, Any],
        doc: dict[str, Any],
        score: int,
        judge_error: str | None,
    ) -> dict[str, Any]:
        result = deepcopy(sample)
        result.update(
            {
                "target": doc.get("answer", ""),
                "question_type": doc.get("question_type", "unknown"),
                "llm_match_score": {"llm_match_score": score},
            }
        )
        if judge_error:
            result["judge_failed"] = True
            result["judge_error"] = judge_error
        return result

    def _score_openeqa_batch(self, samples: list[dict[str, Any]], task) -> list[dict[str, Any]]:
        self._ensure_openai_import_env()
        from embodied_eval.tasks.openeqa import process as openeqa_process

        contexts = []
        messages_batch = []
        prompt = openeqa_process.load_prompt("mmbench")
        for sample in samples:
            doc = task.eval_docs[int(sample["doc_id"])]
            prediction = self._extract_prediction(sample)
            contexts.append((sample, doc, prediction))
            messages_batch.append(
                openeqa_process.prepare_openai_messages(
                    prompt.format(
                        question=doc["question"],
                        answer=doc["answer"],
                        prediction=prediction,
                        extra_answers=None,
                    )
                )
            )

        try:
            outputs = self.generate_chat(messages_batch, max_new_tokens=32)
            self._validate_output_count(outputs, contexts)
        except Exception as e:
            return [
                self._make_openeqa_result(sample, doc, 1, str(e))
                for sample, doc, _ in contexts
            ]

        scored_samples = []
        for (sample, doc, _), output in zip(contexts, outputs):
            try:
                score = openeqa_process.parse_score(output)
                if score < 1 or score > 5:
                    raise ValueError(f"Invalid judge score {score} from output: {output!r}")
                score = openeqa_process.clamp_llm_score(score)
                scored_samples.append(self._make_openeqa_result(sample, doc, score, None))
            except Exception as e:
                scored_samples.append(self._make_openeqa_result(sample, doc, 1, str(e)))
        return scored_samples

    def _make_openeqa_result(
        self,
        sample: dict[str, Any],
        doc: dict[str, Any],
        score: int,
        judge_error: str | None,
    ) -> dict[str, Any]:
        result = deepcopy(sample)
        sample_result = {
            "target": doc.get("answer", ""),
            "question_type": doc.get("category", "unknown"),
            "llm_match_score": {"llm_match_score": score},
        }
        if doc.get("extra_answers"):
            sample_result["extra_answers"] = doc.get("extra_answers")
        result.update(sample_result)
        if judge_error:
            result["judge_failed"] = True
            result["judge_error"] = judge_error
        return result

    def _score_unieqa_batch(self, samples: list[dict[str, Any]], task) -> list[dict[str, Any]]:
        self._ensure_openai_import_env()
        from embodied_eval.tasks.unieqa import process as unieqa_process

        contexts = []
        messages_batch = []
        for sample in samples:
            doc = task.eval_docs[int(sample["doc_id"])]
            prediction = self._extract_prediction(sample)
            target = doc.get("answer", "")
            dimension = doc.get("capability_dimension", "default")
            question_type = doc.get("question_type", "open-ended")
            prompt_template = unieqa_process._resolve_prompt(dimension, question_type)
            prompt = prompt_template + (
                f"\nThe Student's Answer is:\"{prediction}\"\n"
                f"The Correct Answer is:\"{target}\""
            )
            contexts.append((sample, doc, prediction, target, dimension, question_type))
            messages_batch.append(
                [
                    {"role": "system", "content": unieqa_process.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )

        try:
            outputs = self.generate_chat(messages_batch, max_new_tokens=256)
            self._validate_output_count(outputs, contexts)
        except Exception as e:
            return [
                self._make_unieqa_result(sample, target, prediction, dimension, question_type, 0, f"Error: {e}", str(e))
                for sample, _, prediction, target, dimension, question_type in contexts
            ]

        scored_samples = []
        for (sample, _, prediction, target, dimension, question_type), output in zip(contexts, outputs):
            try:
                normalized = self._normalize_json_object_response(output)
                content = json.loads(normalized)
                score_str = str(content.get("Final Score", "0"))
                score_match = re.search(r"([0-1](\.\d+)?)", score_str)
                if not score_match:
                    raise ValueError(f"Invalid judge score from output: {output!r}")
                score = float(score_match.group(1))
                assessment = content.get("Assessment", "")
                assessment_blob = json.dumps({"scores": [score], "assessments": [assessment]}, ensure_ascii=False)
                scored_samples.append(
                    self._make_unieqa_result(
                        sample,
                        target,
                        prediction,
                        dimension,
                        question_type,
                        score,
                        assessment_blob,
                        None,
                    )
                )
            except Exception as e:
                error = str(e)
                scored_samples.append(
                    self._make_unieqa_result(
                        sample,
                        target,
                        prediction,
                        dimension,
                        question_type,
                        0,
                        f"Error: {error}",
                        error,
                    )
                )
        return scored_samples

    def _make_unieqa_result(
        self,
        sample: dict[str, Any],
        target: str,
        prediction: str,
        dimension: str,
        question_type: str,
        score: float,
        assessment: str,
        judge_error: str | None,
    ) -> dict[str, Any]:
        result = deepcopy(sample)
        result.update(
            {
                "target": target,
                "prediction": prediction,
                "dimension": dimension,
                "question_type": question_type,
                "score": score,
                "assessment": assessment,
            }
        )
        if judge_error:
            result["judge_failed"] = True
            result["judge_error"] = judge_error
        return result

    @staticmethod
    def _ensure_openai_import_env() -> None:
        os.environ.setdefault("OPENAI_API_KEY", "local-llm-judge")

    @staticmethod
    def _validate_output_count(outputs: list[str], contexts: list) -> None:
        if len(outputs) != len(contexts):
            raise RuntimeError(f"Local LLM Judge returned {len(outputs)} outputs for {len(contexts)} samples")

    def generate_chat(self, messages_or_batch, max_new_tokens: int) -> list[str]:
        if not messages_or_batch:
            return []
        if isinstance(messages_or_batch[0], dict):
            batch_messages = [messages_or_batch]
        else:
            batch_messages = list(messages_or_batch)

        outputs: list[str] = []
        for start in range(0, len(batch_messages), self.batch_size):
            batch = batch_messages[start : start + self.batch_size]
            texts = [self._apply_chat_template(messages) for messages in batch]
            inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

            with torch.no_grad():
                generated = self.model.generate(**inputs, **generation_kwargs)
            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            outputs.extend(
                self.tokenizer.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return [output.strip() for output in outputs]

    @staticmethod
    def _normalize_json_object_response(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                return json.dumps(json.loads(candidate), ensure_ascii=False)
            except json.JSONDecodeError:
                continue
        return content

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"

    @staticmethod
    def _extract_prediction(sample: dict[str, Any]) -> str:
        prediction = sample.get("prediction")
        if prediction:
            return str(prediction)
        resps = sample.get("resps")
        if isinstance(resps, list) and resps:
            first = resps[0]
            if isinstance(first, list) and first:
                return str(first[0] or "")
            return str(first or "")
        return ""
