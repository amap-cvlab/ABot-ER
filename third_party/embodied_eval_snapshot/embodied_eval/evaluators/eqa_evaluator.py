import collections
import gc
import itertools
import json
import random
import math
import os
import pickle
import time
from pathlib import Path
import numpy as np
import torch

from loguru import logger as eval_logger
from tqdm import tqdm

from embodied_eval.evaluators.fast_artifacts import (
    LLM_JUDGE_TASKS,
    expected_sample_count,
    fast_task_dir_name,
    task_artifact_status,
)
from embodied_eval.tasks import TaskManager, get_task_dict, get_task_list
from embodied_eval.models import get_model
from embodied_eval.utils import create_iterator, get_datetime_str


class EQAEvaluator:
    def __init__(self, config):
        self._config = config

        # ========== Initialize TaskManager ==========
        self.task_manager = TaskManager()
        self.task_list = self.config.tasks.split(",")
        self._preserve_task_order = bool(getattr(self.config, "preserve_task_order", False))
        if self._preserve_task_order:
            self.task_names = self._match_tasks_in_request_order(self.task_list)
        else:
            self.task_names = self.task_manager.match_tasks(self.task_list)
        eval_logger.info(f"Selected Tasks: {self.task_names}")
        self.llm_judge_task_names = set(self.task_names).intersection(LLM_JUDGE_TASKS)
        if self._preserve_task_order:
            self.task_dict = self._load_tasks_in_order(self.task_names)
        else:
            self.task_dict = get_task_dict(self.task_list, self.task_manager)

        # ========== Initialize Model ==========
        model_name = self.config.model
        model_args = "" if self.config.model_args is None else self.config.model_args
        model_load_start = time.monotonic()
        self.model = get_model(model_name=model_name).create_from_arg_string(
            model_args,
            additional_config={
            }
        )
        eval_logger.info(f"TIMING model_load_seconds={time.monotonic() - model_load_start:.3f}")

        # Set task_dict for model
        for task_name, task_obj in self.task_dict.items():
            self.model.task_dict[task_name] = task_obj.dataset

        # ========== Initialize Config ==========

    @property
    def config(self):
        return self._config

    def _match_tasks_in_request_order(self, task_list):
        task_names = []
        seen = set()
        for pattern in task_list:
            for task_name in self.task_manager.match_tasks([pattern]):
                if task_name in seen:
                    continue
                task_names.append(task_name)
                seen.add(task_name)
        return task_names

    def _load_tasks_in_order(self, task_names):
        task_dict = {}
        for task_name in task_names:
            task_dict.update(self.task_manager._load_individual_task(task_name))
        return task_dict

    def inference(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        eval_logger.info(
            f"Setting random seed to {self.config.seed}  \
            Setting numpy seed to {self.config.seed}  \
            Setting torch manual seed to {self.config.seed}s"
         )
        requests = collections.defaultdict(list)
        padding_requests = collections.defaultdict(int)

        eval_tasks = get_task_list(self.task_dict)
        for task_output in eval_tasks:
            task = task_output.task
            task_name = task_output.task_name

            sample_doc_ids = self._sample_doc_ids_for_task(task_name)
            limit = None if sample_doc_ids is not None else self._parse_limit_for_fast_eval(task)

            task_build_start = time.monotonic()
            task.build_all_requests(
                limit=limit,
                rank=self.model.rank,
                world_size=self.model.world_size,
                sample_doc_ids=sample_doc_ids,
            )
            eval_logger.info(
                f"TIMING task_build_seconds task={task_name} "
                f"rank={self.model.rank} requests={len(task._instances)} "
                f"seconds={time.monotonic() - task_build_start:.3f}"
            )
            eval_logger.debug(f"Task: {task_name}; number of requests on this rank: {len(task._instances)}")

            for instance in task.instances:
                reqtype = instance.request_type
                requests[reqtype].append(instance)

                # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            if self.model.world_size > 1:
                instances_rnk = torch.tensor(len(task._instances), device=self.model.device)
                gathered_item = self.model.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
                numpad = max(gathered_item) - gathered_item[self.model.rank]
                padding_requests[task.output_type] += numpad

        for reqtype, reqs in requests.items():
            eval_logger.info(f"Running {reqtype} requests")
            cloned_reqs = []
            for req in reqs:
                cloned_reqs.extend([req] * req.repeats)

            # (FSDP/DDP require even batches among ranks)
            if (self.model.world_size > 1) and (padding_requests[reqtype] > 0):
                for _ in range(padding_requests[reqtype]):
                    cloned_reqs.extend([req] * req.repeats)

            resps = self.single_inference(cloned_reqs)

            # put responses from model into a list of length K for each request.
            for x, req in zip(resps, cloned_reqs):
                req.resps.append(x)

            if self.model.world_size > 1:
                self.model.accelerator.wait_for_everyone()

        return eval_tasks

    def _make_padding_request(self, task):
        original_instances = task._instances
        try:
            task.build_all_requests(limit=1, rank=0, world_size=1)
            if not task.instances:
                return None
            return task.instances[0]
        finally:
            task._instances = original_instances

    def _parse_limit_for_fast_eval(self, task):
        if self.config.limit is None:
            return None
        try:
            limit = float(self.config.limit)
            if limit < 1:
                return int(len(task.eval_docs) * limit)
            return int(limit)
        except (ValueError, TypeError):
            eval_logger.warning(f"Invalid limit value: {self.config.limit}, ignoring limit")
            return None

    def _sample_doc_ids_for_task(self, task_name: str) -> list[int] | None:
        sample_doc_ids_by_task = getattr(self.config, "sample_doc_ids_by_task", None) or {}
        sample_doc_ids = sample_doc_ids_by_task.get(task_name)
        if sample_doc_ids is None:
            return None
        return list(sample_doc_ids)

    def _expected_sample_count_for_task(self, task_output) -> int:
        sample_doc_ids = self._sample_doc_ids_for_task(task_output.task_name)
        if sample_doc_ids is not None:
            return len(sample_doc_ids)
        return expected_sample_count(task_output.task, self.config.limit)

    def _build_requests_for_fast_eval(self, eval_tasks):
        requests = collections.defaultdict(list)
        padding_requests = collections.defaultdict(int)
        padding_sources = {}

        for task_output in eval_tasks:
            task = task_output.task
            task_name = task_output.task_name

            limit = self._parse_limit_for_fast_eval(task)
            sample_doc_ids = self._sample_doc_ids_for_task(task_name)
            if sample_doc_ids is not None:
                limit = None
            task.build_all_requests(
                limit=limit,
                rank=self.model.rank,
                world_size=self.model.world_size,
                sample_doc_ids=sample_doc_ids,
            )
            eval_logger.debug(f"Task: {task_name}; number of requests on this rank: {len(task._instances)}")

            local_instances = list(task.instances)
            for instance in task.instances:
                reqtype = instance.request_type
                requests[reqtype].append(instance)

            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            if self.model.world_size > 1:
                instances_rnk = torch.tensor(len(task._instances), device=self.model.device)
                gathered_item = self.model.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
                numpad = max(gathered_item) - gathered_item[self.model.rank]
                padding_requests[task.output_type] += numpad
                if numpad > 0 and task.output_type not in padding_sources:
                    padding_sources[task.output_type] = (
                        local_instances[-1] if local_instances else self._make_padding_request(task)
                    )

        return requests, padding_requests, padding_sources

    def _run_fast_requests(self, requests, padding_requests, padding_sources):
        for reqtype, reqs in requests.items():
            eval_logger.info(f"Running {reqtype} requests")
            request_entries = []
            for req in reqs:
                request_entries.extend((req, False) for _ in range(req.repeats))

            # (FSDP/DDP require even batches among ranks)
            if (self.model.world_size > 1) and (padding_requests[reqtype] > 0):
                padding_source = padding_sources.get(reqtype)
                if padding_source is None:
                    raise RuntimeError(f"Unable to build padding request for request type {reqtype}")
                for _ in range(padding_requests[reqtype]):
                    request_entries.extend((padding_source, True) for _ in range(padding_source.repeats))

            cloned_reqs = [req for req, _ in request_entries]
            resps = self.single_inference_batched(cloned_reqs)

            # put responses from model into a list of length K for each request.
            for x, (req, is_padding) in zip(resps, request_entries):
                if not is_padding:
                    req.resps.append(x)

            if self.model.world_size > 1:
                self.model.accelerator.wait_for_everyone()

    def _wait_for_workers(self):
        accelerator = getattr(self.model, "accelerator", None)
        if accelerator is not None:
            accelerator.wait_for_everyone()

    def _progress_desc(self, stage: str) -> str:
        task_progress = getattr(self, "_current_task_progress", None)
        if task_progress:
            return f"{task_progress} {stage}"
        return stage

    def _is_llm_judge_task(self, task_name: str) -> bool:
        return task_name in self.llm_judge_task_names

    def _task_artifact_status(self, task_output, llm_judge: str):
        if not self.config.output_path:
            return None
        return task_artifact_status(
            Path(self.config.output_path),
            task_output.task_name,
            llm_judge,
            self._expected_sample_count_for_task(task_output),
        )

    def _skip_completed_task(self, task_output, task_index: int, total_tasks: int, llm_judge: str) -> bool:
        status = self._task_artifact_status(task_output, llm_judge)
        if status is None:
            return False

        task_name = task_output.task_name
        rank = getattr(self.model, "rank", 0)
        if status.complete:
            if rank == 0:
                eval_logger.info(
                    f"Skipping completed task [{task_index}/{total_tasks}] {task_name}: {status.reason}"
                )
                self._notify_task_result_saved(task_name)
            return True

        if rank == 0 and status.saw_any_artifact:
            eval_logger.warning(
                f"Re-running incomplete task [{task_index}/{total_tasks}] {task_name}: {status.reason}"
            )
            self._remove_fast_scored_artifacts(task_name)
        return False

    def _notify_task_result_saved(self, task_name: str) -> None:
        callback = getattr(self.config, "task_result_saved_callback", None)
        if callable(callback):
            callback(task_name)
            eval_logger.info(f"Published task result to external CSV: task={task_name}")

    def _should_fallback_llm_judge(self, task_name: str) -> bool:
        return getattr(self, "_allow_llm_judge_fallback", False) and self._is_llm_judge_task(task_name)

    def _fallback_llm_judge_result(self, task_name, doc, requests, error):
        prediction = ""
        if requests and requests[0].resps and requests[0].resps[0]:
            prediction = requests[0].resps[0]

        eval_logger.error(
            f"LLM Judge failed for task={task_name} doc_id={requests[0].doc_id if requests else 'unknown'}; "
            f"using fallback score. error={error}"
        )

        if task_name == "robovqa":
            return {
                "target": doc.get("answer", ""),
                "question_type": doc.get("question_type", "unknown"),
                "prediction": prediction,
                "llm_match_score": {"llm_match_score": 1},
                "judge_failed": True,
                "judge_error": str(error),
            }
        if task_name == "openeqa-emeqa":
            result = {
                "target": doc.get("answer", ""),
                "question_type": doc.get("category", "unknown"),
                "prediction": prediction,
                "llm_match_score": {"llm_match_score": 1},
                "judge_failed": True,
                "judge_error": str(error),
            }
            if doc.get("extra_answers"):
                result["extra_answers"] = doc.get("extra_answers")
            return result
        if task_name == "unieqa":
            return {
                "target": doc.get("answer", ""),
                "prediction": prediction,
                "dimension": doc.get("capability_dimension", "default"),
                "question_type": doc.get("question_type", "open-ended"),
                "score": 0,
                "assessment": f"Judge failed: {error}",
                "judge_failed": True,
                "judge_error": str(error),
            }
        raise RuntimeError(f"Unsupported LLM Judge fallback task: {task_name}") from error

    def _run_fast_model_task(self, task_output, task_index: int, total_tasks: int, llm_judge: str):
        task_name = task_output.task_name
        self._current_task_progress = f"[{task_index}/{total_tasks}] {task_name}"
        eval_logger.info(f"Starting task {self._current_task_progress}")
        set_active_task = getattr(self.model, "set_active_task", None)
        if callable(set_active_task):
            set_active_task(task_name)

        try:
            if self._skip_completed_task(task_output, task_index, total_tasks, llm_judge):
                return None

            single_task_outputs = [task_output]
            requests, padding_requests, padding_sources = self._build_requests_for_fast_eval(single_task_outputs)
            self._run_fast_requests(requests, padding_requests, padding_sources)

            skip_llm_judge = self._is_llm_judge_task(task_name) and llm_judge == "no"
            if skip_llm_judge:
                eval_logger.info(f"Skipping LLM Judge for {task_name}; saving model samples only.")
                if self.config.save_results:
                    self.save_fast_samples(single_task_outputs)
                return None

            if self._is_llm_judge_task(task_name) and self.config.save_results:
                self.save_fast_samples(single_task_outputs)
            results_dict = self.evaluate(single_task_outputs)
            if results_dict is not None:
                self.print_results(results_dict)
                if self.config.save_results:
                    self.save_fast_results(results_dict)
                    if getattr(self.model, "rank", 0) == 0:
                        self._notify_task_result_saved(task_name)
            return results_dict
        finally:
            if callable(set_active_task):
                set_active_task(None)
            self._current_task_progress = None

    def _run_fast_llm_judge_inference_only(self, task_output, task_index: int, total_tasks: int):
        task_name = task_output.task_name
        self._current_task_progress = f"[{task_index}/{total_tasks}] {task_name}"
        eval_logger.info(f"Starting local-judge sample generation {self._current_task_progress}")
        set_active_task = getattr(self.model, "set_active_task", None)
        if callable(set_active_task):
            set_active_task(task_name)

        try:
            single_task_outputs = [task_output]
            requests, padding_requests, padding_sources = self._build_requests_for_fast_eval(single_task_outputs)
            self._run_fast_requests(requests, padding_requests, padding_sources)
            local_samples = self._collect_model_samples(task_output, gather=False)
            if self.config.save_results:
                self._save_gathered_fast_samples(task_output, local_samples)
            return local_samples
        finally:
            if callable(set_active_task):
                set_active_task(None)
            self._current_task_progress = None

    def _make_local_results_dict(self, task_output):
        results_dict = collections.defaultdict(dict)
        results_dict["results"] = {task_output.task_name: task_output.calculate_aggregate_metric()}
        results_dict["samples"] = {task_output.task_name: task_output.logged_samples}
        results_dict["configs"] = {task_output.task_name: task_output.task_config}
        return results_dict

    def _release_model_for_local_judge(self):
        rank = self.model.rank
        world_size = self.model.world_size
        device = str(self.model.device)
        accelerator = getattr(self.model, "accelerator", None)

        if accelerator is not None:
            accelerator.wait_for_everyone()
        if hasattr(self.model, "_model"):
            del self.model._model
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        elif world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        eval_logger.info(f"Released evaluated checkpoint before local LLM Judge load on rank={rank}")
        return rank, world_size, device

    def _gather_scored_samples(self, local_samples, rank: int, world_size: int):
        if world_size <= 1:
            return local_samples

        full_samples = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(
            obj=local_samples,
            object_gather_list=full_samples,
            dst=0,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if rank == 0:
            return list(itertools.chain.from_iterable(full_samples))
        return []

    def _run_local_llm_judge_scoring(
        self,
        llm_tasks,
        local_samples_by_task,
        llm_judge_model_path: str,
        rank: int,
        world_size: int,
        device: str,
    ) -> None:
        from embodied_eval.evaluators.local_llm_judge import LocalLLMJudge

        judge = LocalLLMJudge(
            model_path=llm_judge_model_path,
            device=device,
            batch_size=max(int(getattr(self.config, "batch_size", 1) or 1), 1),
            rank=rank,
        )
        try:
            for task_index, total_tasks, task_output in llm_tasks:
                task_name = task_output.task_name
                self._current_task_progress = f"[{task_index}/{total_tasks}] {task_name}"
                local_samples = local_samples_by_task.get(task_name, [])
                eval_logger.info(
                    f"Scoring {len(local_samples)} local samples for {task_name} "
                    f"with local LLM Judge on rank={rank}/{world_size}"
                )
                try:
                    scored_local = judge.score_samples(
                        task_name,
                        local_samples,
                        task_output.task,
                        progress_desc=self._progress_desc("Postprocessing"),
                    )
                finally:
                    self._current_task_progress = None
                scored_samples = self._gather_scored_samples(scored_local, rank, world_size)
                if rank == 0:
                    task_output.logged_samples = scored_samples
                    results_dict = self._make_local_results_dict(task_output)
                    self.print_results(results_dict)
                    if self.config.save_results:
                        self.save_fast_results(results_dict)
                        self._notify_task_result_saved(task_name)
        finally:
            judge.close()
            if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

    def _run_sequential_tasks_with_local_judge(self, eval_tasks, llm_judge_model_path: str):
        total_tasks = len(eval_tasks)
        llm_tasks = []
        local_samples_by_task = {}
        for task_index, task_output in enumerate(eval_tasks, start=1):
            if self._is_llm_judge_task(task_output.task_name):
                if self._skip_completed_task(task_output, task_index, total_tasks, llm_judge="yes"):
                    self._wait_for_workers()
                    continue
                llm_tasks.append((task_index, total_tasks, task_output))
                local_samples_by_task[task_output.task_name] = self._run_fast_llm_judge_inference_only(
                    task_output,
                    task_index,
                    total_tasks,
                )
            else:
                self._run_fast_model_task(task_output, task_index, total_tasks, llm_judge="yes")
            self._wait_for_workers()

        if not llm_tasks:
            self._maybe_update_fast_results_summary()
            return

        rank, world_size, device = self._release_model_for_local_judge()
        self._run_local_llm_judge_scoring(
            llm_tasks,
            local_samples_by_task,
            llm_judge_model_path,
            rank,
            world_size,
            device,
        )

    def run_sequential_tasks(
        self,
        llm_judge: str = "yes",
        llm_judge_model: str = "api",
        llm_judge_model_path: str | None = None,
    ):
        self._keep_model_loaded = True
        self._allow_llm_judge_fallback = True
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        eval_logger.info(
            f"Setting random seed to {self.config.seed}  \
            Setting numpy seed to {self.config.seed}  \
            Setting torch manual seed to {self.config.seed}s"
         )
        eval_tasks = get_task_list(self.task_dict)
        if llm_judge == "yes" and llm_judge_model == "local":
            if not llm_judge_model_path:
                raise ValueError("llm_judge_model_path is required when llm_judge_model='local'")
            self._run_sequential_tasks_with_local_judge(eval_tasks, llm_judge_model_path)
            return

        total_tasks = len(eval_tasks)
        for task_index, task_output in enumerate(eval_tasks, start=1):
            self._run_fast_model_task(task_output, task_index, total_tasks, llm_judge=llm_judge)
            self._wait_for_workers()
        self._maybe_update_fast_results_summary()
    
    def single_inference(self, requests):
        """
        Generate responses for a list of requests.
        
        Args:
            requests: List of Instance objects
            
        Returns:
            List[str]: List of generated responses
        """
        responses = []

        def process_request(request) -> str:
            """
            Process a single request and return the response.
            
            Args:
                request: Instance object containing request information
                
            Returns:
                str: Generated response text
            """
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            gen_kwargs = gen_kwargs if gen_kwargs else {}
            
            # Get visuals for this request
            if split is not None:
                visuals = doc_to_visual(self.model.task_dict[task][split][doc_id])
                if isinstance(visuals, tuple):  # ([visual], [visual_index])
                    visual_list = visuals[0]
                else:
                    visual_list = visuals  # [visual]
            else:
                visuals = doc_to_visual(self.model.task_dict[task][doc_id])
                if isinstance(visuals, dict):
                    visual_list = visuals[0]
                else:
                    visual_list = visuals

            return context, visual_list, gen_kwargs
        
        eval_logger.info(f"Processing {len(requests)} requests")
        task_timings = collections.defaultdict(float)
        task_counts = collections.defaultdict(int)
        progress_bar = tqdm(total=len(requests), disable=(self.model.rank != 0), desc=self._progress_desc("Model Responding"))
        for request in requests:
            task_name = request.metadata.get("task", "unknown")
            request_start = time.monotonic()
            try:
                context, visual_list, gen_kwargs = process_request(request)
                response = self.model.respond(context, visual_list, **gen_kwargs).strip()
                responses.append(response)
            except Exception as e:
                eval_logger.error(f"Error processing request: {e}")
                responses.append("")  # Return empty string on error
            finally:
                task_timings[task_name] += time.monotonic() - request_start
                task_counts[task_name] += 1
            progress_bar.update(1)
        
        progress_bar.close()
        for task_name in sorted(task_timings):
            eval_logger.info(
                f"TIMING task_inference_seconds task={task_name} "
                f"rank={self.model.rank} requests={task_counts[task_name]} "
                f"seconds={task_timings[task_name]:.3f}"
            )
        return responses

    def single_inference_batched(self, requests):
        respond_requests_batch = getattr(self.model, "respond_requests_batch", None)
        if callable(respond_requests_batch):
            return respond_requests_batch(
                requests,
                task_dict=self.model.task_dict,
                output_path=self.config.output_path,
                progress_desc=self._progress_desc("Model Responding"),
            )

        responses = []

        def process_request(request):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            gen_kwargs = gen_kwargs if gen_kwargs else {}

            if split is not None:
                visuals = doc_to_visual(self.model.task_dict[task][split][doc_id])
                if isinstance(visuals, tuple):  # ([visual], [visual_index])
                    visual_list = visuals[0]
                else:
                    visual_list = visuals  # [visual]
            else:
                visuals = doc_to_visual(self.model.task_dict[task][doc_id])
                if isinstance(visuals, dict):
                    visual_list = visuals[0]
                else:
                    visual_list = visuals

            return context, visual_list, gen_kwargs

        def gen_kwargs_key(gen_kwargs):
            return json.dumps(gen_kwargs, sort_keys=True, default=str)

        def respond_single(payload):
            context, visual_list, gen_kwargs = payload
            return self.model.respond(context, visual_list, **gen_kwargs).strip()

        def respond_payloads(payloads):
            if len(payloads) == 1 or not hasattr(self.model, "respond_batch"):
                outputs = []
                for payload in payloads:
                    try:
                        outputs.append(respond_single(payload))
                    except Exception as e:
                        eval_logger.error(f"Error processing request: {e}")
                        outputs.append("")
                return outputs

            contexts = [payload[0] for payload in payloads]
            visuals_batch = [payload[1] for payload in payloads]
            gen_kwargs_batch = [payload[2] for payload in payloads]
            try:
                outputs = self.model.respond_batch(contexts, visuals_batch, gen_kwargs_batch)
                if len(outputs) != len(payloads):
                    raise RuntimeError(
                        f"respond_batch returned {len(outputs)} outputs for {len(payloads)} requests"
                    )
                return [output.strip() if isinstance(output, str) else str(output).strip() for output in outputs]
            except Exception as e:
                if len(payloads) == 1:
                    eval_logger.error(f"Error processing request: {e}")
                    return [""]
                eval_logger.warning(
                    f"Batch inference failed for batch_size={len(payloads)}; "
                    f"retrying with smaller batches. error={e}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mid = max(len(payloads) // 2, 1)
                return respond_payloads(payloads[:mid]) + respond_payloads(payloads[mid:])

        def flush_batch(batch):
            batch_responses = respond_payloads(batch)
            responses.extend(batch_responses)
            progress_bar.update(len(batch))

        eval_logger.info(f"Processing {len(requests)} requests")
        batch_size = max(int(getattr(self.config, "batch_size", 1) or 1), 1)
        progress_bar = tqdm(total=len(requests), disable=(self.model.rank != 0), desc=self._progress_desc("Model Responding"))
        batch = []
        batch_key = None

        for request in requests:
            try:
                context, visual_list, gen_kwargs = process_request(request)
            except Exception as e:
                if batch:
                    flush_batch(batch)
                    batch = []
                    batch_key = None
                eval_logger.error(f"Error preparing request: {e}")
                responses.append("")
                progress_bar.update(1)
                continue

            current_key = gen_kwargs_key(gen_kwargs)
            if batch and (current_key != batch_key or len(batch) >= batch_size):
                flush_batch(batch)
                batch = []

            batch.append((context, visual_list, gen_kwargs))
            batch_key = current_key

        if batch:
            flush_batch(batch)

        progress_bar.close()
        return responses

    def evaluate(self, eval_tasks):
        results_dict = collections.defaultdict(dict)
        results = collections.defaultdict(dict)
        configs = collections.defaultdict(dict)
        samples = collections.defaultdict(list)

        RANK = self.model.rank
        WORLD_SIZE = self.model.world_size

        for task_output in eval_tasks:
            task_eval_start = time.monotonic()
            task = task_output.task

            # Pre-process task.instances to group by doc_id
            instances_by_doc_id = collections.defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)

            sample_doc_ids = self._sample_doc_ids_for_task(task_output.task_name)
            limit = None if sample_doc_ids is not None else self._parse_limit_for_fast_eval(task)

            # iterate over different filters used
            doc_iterator = task.doc_iterator(
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
                sample_doc_ids=sample_doc_ids,
            )
            total_docs = len(sample_doc_ids) if sample_doc_ids is not None else len(task.eval_docs)
            doc_iterator_for_counting = create_iterator(
                range(total_docs),
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
            )
            num_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=num_docs, desc=self._progress_desc("Postprocessing"), disable=(RANK != 0))
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                try:
                    sample_result = task.process_results(doc, [req.resps for req in requests])
                except Exception as e:
                    if not self._should_fallback_llm_judge(task_output.task_name):
                        raise
                    sample_result = self._fallback_llm_judge_result(task_output.task_name, doc, requests, e)
                if sample_result.get("judge_failed"):
                    eval_logger.error(
                        f"LLM Judge sample failed task={task_output.task_name} doc_id={doc_id}; "
                        f"using fallback score. error={sample_result.get('judge_error', '<missing>')}"
                    )
                example = {
                    "doc_id": doc_id,
                    "doc": requests[0].args[0],
                    "resps": [req.resps for req in requests],
                }
                for key in ["image", "images", "video"]:
                    if doc.get(key) is not None:
                        example[key] = doc[key]
                example.update({key: value for key, value in sample_result.items()})
                task_output.logged_samples.append(example)

                pbar.update(1)
            pbar.close()
            eval_logger.info(
                f"TIMING task_evaluate_seconds task={task_output.task_name} "
                f"rank={RANK} samples={len(task_output.logged_samples)} "
                f"seconds={time.monotonic() - task_eval_start:.3f}"
            )

        if hasattr(self.model, "_model") and not getattr(self, "_keep_model_loaded", False):
            del self.model._model
            torch.cuda.empty_cache()

        if WORLD_SIZE > 1:
            # if multigpu, then gather data across all ranks to rank 0
            # first gather logged samples across all ranks
            for task_output in eval_tasks:
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = []
                for sample in task_output.logged_samples:
                    per_rank_samples.append(sample)
                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))

                eval_logger.info(f"Gathering sample across all ranks for: {task_output.task_name}")
            torch.distributed.barrier()  # Ensure all processes are synced before proceeding

        if RANK == 0:
            ### Aggregate results over all datapoints ###
            # aggregate results ; run bootstrap CIs
            for task_output in eval_tasks:
                agg_result = task_output.calculate_aggregate_metric()
                results[task_output.task_name] = agg_result
                configs[task_output.task_name] = task_output.task_config
                samples[task_output.task_name] = task_output.logged_samples

            results_dict["results"] = dict(results)
            results_dict["samples"] = dict(samples)
            results_dict["configs"] = dict(configs)

            eval_logger.info(f"Aggregating results across on ranks {RANK}")
        else:
            results_dict = None

            eval_logger.info(f"Pass on ranks {RANK}")

        if hasattr(self.model, "accelerator"):
            self.model.accelerator.wait_for_everyone()

        return results_dict

    def print_results(self, results_dict):
        def make_table(result):
            from pytablewriter import MarkdownTableWriter
            import re

            model = self.config.model

            type_to_metrics = {}
            for key, val in result.items():
                if key == "overall" or key.endswith("_stderr") or key.endswith("_average"):
                    continue
                qtype, metric_name = key.rsplit("_", 1)
                if qtype not in type_to_metrics:
                    type_to_metrics[qtype] = {}
                type_to_metrics[qtype][metric_name] = val

            all_metrics = sorted({m for v in type_to_metrics.values() for m in v})
            headers = ["Metric"] + list(type_to_metrics.keys()) + ["Average"]
            value_matrix = []

            for metric in all_metrics:
                row = [metric]
                for qtype in type_to_metrics:
                    val = type_to_metrics[qtype].get(metric, "")
                    row.append(f"{val:.4f}" if val != "" else "")
                avg_key = f"{metric}_average"
                avg_val = result.get(avg_key, "")
                row.append(f"{avg_val:.4f}" if avg_val != "" else "")
                value_matrix.append(row)

            # Add the metric averages to the original result dictionary
            md_writer = MarkdownTableWriter()
            md_writer.table_name = f"Results for {model}"
            md_writer.headers = headers
            md_writer.value_matrix = value_matrix

            eval_logger.info("Markdown Table:\n")
            eval_logger.info(f"Overall: {result['overall']}")
            md_writer.write_table()
        # self.save_results(results_dict)
        results = results_dict["results"]
        configs = results_dict["configs"]
        for task_name, _ in results.items():
            result, config = results[task_name], configs[task_name]
            info = f"task:{config['task']}, data:{config['dataset_path']}/{config.get('dataset_kwargs')},\
                            generate:{config['generation_kwargs']}"
            eval_logger.info(info)
            make_table(result)

    @staticmethod
    def _json_default(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def _fast_task_output_dir(self, task_name):
        task_dir_name = fast_task_dir_name(task_name)
        save_dir = os.path.join(self.config.output_path, task_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _fast_task_result_path(self, task_name):
        task_dir_name = fast_task_dir_name(task_name)
        return os.path.join(self.config.output_path, task_dir_name, f"results_{task_name}.json")

    def _remove_fast_scored_artifacts(self, task_name):
        if not self.config.output_path:
            return
        task_dir = Path(self.config.output_path) / fast_task_dir_name(task_name)
        for prefix in ("results", "configs"):
            path = task_dir / f"{prefix}_{task_name}.json"
            try:
                path.unlink()
                eval_logger.info(f"Removed stale {prefix} artifact before rerun: {path}")
            except FileNotFoundError:
                continue
            except OSError as exc:
                eval_logger.warning(f"Failed to remove stale {prefix} artifact before rerun: {path} ({exc})")

    @staticmethod
    def _fast_summary_key(task_name):
        if task_name.startswith("cosmos-"):
            return "cosmos"
        return task_name

    @staticmethod
    def _normalize_llm_match_score(score):
        return min(max((float(score) - 1.0) / 4.0, 0.0), 1.0)

    def _extract_final_score(self, task_name, results):
        if task_name in {"robovqa", "openeqa-emeqa"}:
            if "overall" in results and isinstance(results["overall"], (int, float, np.integer, np.floating)):
                return self._normalize_llm_match_score(results["overall"])
            if "100score_overall" in results and isinstance(results["100score_overall"], (int, float, np.integer, np.floating)):
                return min(max(float(results["100score_overall"]) / 100.0, 0.0), 1.0)
            return None

        for key in ("overall", "score_average", "accuracy_average"):
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
        return None

    def _load_fast_task_score(self, task_name):
        result_path = self._fast_task_result_path(task_name)
        if not os.path.isfile(result_path):
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            eval_logger.warning(f"Skipping invalid result file while updating summary: {result_path} ({e})")
            return None

        score = self._extract_final_score(task_name, results)
        if score is None:
            eval_logger.warning(f"No final score found while updating summary: {result_path}")
        return score

    def update_fast_results_summary(self):
        if not self.config.output_path:
            return

        summary = collections.OrderedDict()
        summary["completed_count"] = 0
        scores = []
        cosmos_scores = []
        cosmos_written = False

        for task_name in self.task_names:
            score = self._load_fast_task_score(task_name)
            if score is None:
                continue

            summary_key = self._fast_summary_key(task_name)
            if summary_key == "cosmos":
                cosmos_scores.append(score)
                if not cosmos_written:
                    summary["cosmos"] = 0.0
                    cosmos_written = True
                summary["cosmos"] = sum(cosmos_scores) / len(cosmos_scores)
                continue

            summary[summary_key] = score

        for key, value in summary.items():
            if key == "completed_count":
                continue
            scores.append(value)

        summary["completed_count"] = len(scores)
        summary["average"] = sum(scores) / len(scores) if scores else 0.0

        summary_path = os.path.join(self.config.output_path, "results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        eval_logger.info(f"Updated fast results summary: {summary_path}")

    def _maybe_update_fast_results_summary(self):
        if self.config.output_path and getattr(self.model, "rank", 0) == 0:
            self.update_fast_results_summary()

    def _write_jsonl(self, path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, default=self._json_default, ensure_ascii=False) + "\n")

    def _collect_model_samples(self, task_output, gather: bool = True):
        task = task_output.task
        RANK = self.model.rank
        WORLD_SIZE = self.model.world_size

        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)

        sample_doc_ids = self._sample_doc_ids_for_task(task_output.task_name)
        limit = None if sample_doc_ids is not None else self._parse_limit_for_fast_eval(task)

        task_samples = []
        for doc_id, doc in task.doc_iterator(
            rank=RANK,
            limit=limit,
            world_size=WORLD_SIZE,
            sample_doc_ids=sample_doc_ids,
        ):
            requests = instances_by_doc_id[doc_id]
            if not requests:
                continue

            example = {
                "doc_id": doc_id,
                "doc": requests[0].args[0],
                "resps": [req.resps for req in requests],
            }
            for key in ["image", "images", "video"]:
                if doc.get(key) is not None:
                    example[key] = doc[key]
            task_samples.append(example)

        if gather and WORLD_SIZE > 1:
            full_samples = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(
                obj=task_samples,
                object_gather_list=full_samples,
                dst=0,
            )
            if RANK == 0:
                task_samples = list(itertools.chain.from_iterable(full_samples))
            self.model.accelerator.wait_for_everyone()

        return task_samples

    def _save_gathered_fast_samples(self, task_output, local_samples):
        if not self.config.output_path:
            eval_logger.info("Output path not provided, skipping saving samples")
            return

        rank = self.model.rank
        world_size = self.model.world_size
        task_samples = local_samples
        if world_size > 1:
            full_samples = [None] * world_size if rank == 0 else None
            torch.distributed.gather_object(
                obj=local_samples,
                object_gather_list=full_samples,
                dst=0,
            )
            if rank == 0:
                task_samples = list(itertools.chain.from_iterable(full_samples))
            self.model.accelerator.wait_for_everyone()

        if rank == 0:
            task_name = task_output.task_name
            save_dir = self._fast_task_output_dir(task_name)
            eval_logger.info(f"Saving model samples for: {task_name} ({len(task_samples)} samples)")
            file_results_samples = os.path.join(save_dir, f"samples_{task_name}.json")
            self._write_jsonl(file_results_samples, task_samples)
            eval_logger.info(f"Model samples saved to: {file_results_samples}")

    def save_fast_results(self, results_dict):

        tasks_samples = results_dict.pop('samples')
        tasks_results = results_dict.pop('results')
        tasks_configs = results_dict.pop('configs')
        print("''''save result''''")
        for task_name, _ in tasks_results.items():
            samples = tasks_samples[task_name]
            results = tasks_results[task_name]
            configs = tasks_configs[task_name]

            if self.config.output_path:
                save_dir = self._fast_task_output_dir(task_name)
                eval_logger.info(f"Saving per-sample results for: {task_name}")

                file_results_samples = os.path.join(save_dir, f"samples_{task_name}.json")
                self._write_jsonl(file_results_samples, samples)

                file_results_aggregated = os.path.join(save_dir, f"results_{task_name}.json")
                result_dumped = json.dumps(results, indent=4, default=self._json_default)
                with open(file_results_aggregated, "w", encoding="utf-8") as f:
                    f.write(result_dumped)

                file_configs_aggregated = os.path.join(save_dir, f"configs_{task_name}.json")
                result_dumped = json.dumps(configs, indent=4, default=self._json_default)
                with open(file_configs_aggregated, "w", encoding="utf-8") as f:
                    f.write(result_dumped)

            else:
                eval_logger.info("Output path not provided, skipping saving results")

        if self.config.output_path:
            self.update_fast_results_summary()

    def save_fast_samples(self, eval_tasks):
        """Save model responses in the standard samples_<task>.json format."""
        if not self.config.output_path:
            eval_logger.info("Output path not provided, skipping saving samples")
            return

        RANK = self.model.rank

        for task_output in eval_tasks:
            task_name = task_output.task_name
            task_samples = self._collect_model_samples(task_output)

            if RANK == 0:
                save_dir = self._fast_task_output_dir(task_name)
                eval_logger.info(f"Saving model samples for: {task_name} ({len(task_samples)} samples)")

                file_results_samples = os.path.join(save_dir, f"samples_{task_name}.json")
                self._write_jsonl(file_results_samples, task_samples)

                eval_logger.info(f"Model samples saved to: {file_results_samples}")

    def save_results(self, results_dict):
        def handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        tasks_samples = results_dict.pop('samples')
        tasks_results = results_dict.pop('results')
        tasks_configs = results_dict.pop('configs')
        datetime_str = None if getattr(self.config, "flat_output", False) else get_datetime_str(timezone=self.config.timezone)
        print("''''save result''''")
        for task_name, _ in tasks_results.items():
            task_save_start = time.monotonic()
            samples = tasks_samples[task_name]
            results = tasks_results[task_name]
            configs = tasks_configs[task_name]

            if self.config.output_path:
                if getattr(self.config, "flat_output", False):
                    task_dir_name = task_name.removesuffix("_aboter")
                    save_dir = os.path.join(self.config.output_path, task_dir_name)
                else:
                    save_dir = os.path.join(self.config.output_path, datetime_str)
                os.makedirs(save_dir, exist_ok=True)
                eval_logger.info(f"Saving per-sample results for: {task_name}")

                file_results_samples = os.path.join(save_dir, f"samples_{task_name}.json")
                with open(file_results_samples, "w", encoding="utf-8") as f:
                    for sample in samples:
                        sample_dump = (
                                json.dumps(
                                    sample,
                                    default=handle_non_serializable,
                                    ensure_ascii=False,
                                )
                                + "\n"
                        )
                        f.write(sample_dump)

                file_results_aggregated = os.path.join(save_dir, f"results_{task_name}.json")
                result_dumped = json.dumps(results, indent=4, default=handle_non_serializable)
                with open(file_results_aggregated, "w", encoding="utf-8") as f:
                    f.write(result_dumped)

                file_configs_aggregated = os.path.join(save_dir, f"configs_{task_name}.json")
                result_dumped = json.dumps(configs, indent=4, default=handle_non_serializable)
                with open(file_configs_aggregated, "w", encoding="utf-8") as f:
                    f.write(result_dumped)
                eval_logger.info(
                    f"TIMING task_save_seconds task={task_name} samples={len(samples)} "
                    f"seconds={time.monotonic() - task_save_start:.3f}"
                )

            else:
                eval_logger.info("Output path not provided, skipping saving results")

    def save_inference_results(self, eval_tasks):
        """Save inference results without evaluation."""
        def handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)
        datetime_str = None if getattr(self.config, "flat_output", False) else get_datetime_str(timezone=self.config.timezone)

        if not self.config.output_path:
            eval_logger.info("Output path not provided, skipping saving inference results")
            return

        RANK = self.model.rank
        WORLD_SIZE = self.model.world_size

        # Process each task separately
        for task_output in eval_tasks:
            task = task_output.task
            task_name = task_output.task_name

            # Group instances by doc_id
            instances_by_doc_id = collections.defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)

            # Collect inference samples for this task
            task_samples = []
            sample_doc_ids = self._sample_doc_ids_for_task(task_output.task_name)
            limit = None if sample_doc_ids is not None else self._parse_limit_for_fast_eval(task)
            doc_iterator = task.doc_iterator(
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
                sample_doc_ids=sample_doc_ids,
            )
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                if not requests:
                    continue

                example = {
                    "doc_id": doc_id,
                    "doc": requests[0].args[0],  # context
                    "resps": [req.resps for req in requests],  # model responses
                }

                # Add visual information if available
                for key in ["image", "images", "video"]:
                    if doc.get(key) is not None:
                        example[key] = doc[key]

                task_samples.append(example)

            # Gather samples from all ranks if multi-GPU
            if WORLD_SIZE > 1:
                per_rank_samples = task_samples
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )
                if RANK == 0:
                    task_samples = list(itertools.chain.from_iterable(full_samples))
                self.model.accelerator.wait_for_everyone()

            # Save results only on rank 0
            if RANK == 0:
                if getattr(self.config, "flat_output", False):
                    task_dir_name = task_name.removesuffix("_aboter")
                    save_dir = os.path.join(self.config.output_path, task_dir_name)
                else:
                    save_dir = os.path.join(self.config.output_path, datetime_str)
                os.makedirs(save_dir, exist_ok=True)
                eval_logger.info(f"Saving inference results for: {task_name} ({len(task_samples)} samples)")

                file_inference_samples = os.path.join(save_dir, f"inference_{task_name}.json")
                with open(file_inference_samples, "w", encoding="utf-8") as f:
                    for sample in task_samples:
                        sample_dump = (
                            json.dumps(
                                sample,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        f.write(sample_dump)

                eval_logger.info(f"Inference results saved to: {file_inference_samples}")
