import copy
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Union

import datasets
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import prepare_data_loader
from mdl.distribute_dataset import DistributeDataset
from mdl.elastic_infer import ElasticInferProcess

from ..data.utils import Role
from ..data.ailake_utils import split_ailake_bytes_data
from ..extras.constants import AUDIO_PLACEHOLDER, FILEEXT2TYPE, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, MediaType
from ..extras.logging import get_logger
from ..extras.misc import is_float_tensor
from .constant import OutputType
from .eip import EIPSingleInstance


if TYPE_CHECKING:
    from ..data.template import Template


logger = get_logger(__name__)

def worker_init_fn(worker_id):
    logger.info(f"[Dataloader][Worker {worker_id}] Started. PID: {os.getpid()}")

class BaseDataloader(object):
    def __init__(
        self,
        tokenizer,
        template: "Template",
        infer_args,
        processor=None,
        model_dtype=None,
        output_type=OutputType.Sequence,
    ):

        self.tokenizer = tokenizer
        self.template = template
        self.processor = processor

        self.prompt_column = infer_args.prompt_column

        # image init process
        self.image_folder = infer_args.image_folder
        self.image_column = infer_args.image_column
        self.video = infer_args.video
        self.video_folder = infer_args.video_folder
        self.audio = infer_args.audio
        self.audio_folder = infer_args.audio_folder
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.template.image_token) if self.image_column is not None else None
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.template.video_token) if self.video is not None else None
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.template.audio_token) if self.audio is not None else None
        self.model_dtype = model_dtype
        if self.image_column and not model_dtype:
            raise Exception('if set image_column, you must set model_dtype')

        # TODO: support multiple multimodal features
        if self.image_column is not None:
            self.media_type = MediaType.IMAGE
        elif self.video is not None:
            self.media_type = MediaType.VIDEO
        elif self.audio is not None:
            self.media_type = MediaType.AUDIO
        else:
            self.media_type = None

        if not infer_args.mix_multimodel:
            if sum([self.image_column is not None, self.video is not None, self.audio is not None]) > 1:
                raise Exception("only one of image, video, audio can be set")

        self.system_column = infer_args.system_column
        self.response_column = infer_args.response_column
        self.history_column = infer_args.history_column
        self.query_column = infer_args.query_column
        self.cutoff_len = infer_args.cutoff_len
        self.input_columns = infer_args.input_columns

        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        self.dataloader = None
        self.infer_mode = infer_args.infer_mode
        self.output_type = output_type
        self.enable_thinking = infer_args.enable_thinking

    def __iter__(self):
        raise NotImplementedError

    def default_collate_fn(self, raw_samples):
        if isinstance(raw_samples[0], dict):
            samples = defaultdict(list)
            for sample in raw_samples:
                for key, value in sample.items():
                    samples[key].append(value)
        elif isinstance(raw_samples[0], list):
            samples = [list(sublist) for sublist in zip(*raw_samples)]
        else:
            raise Exception(f"Unknown dataset item type:{type(raw_samples[0])}")

        prompts = samples[self.prompt_column]
        images = samples[self.image_column] if self.image_column is not None else None
        videos = samples[self.video] if self.video is not None else None
        audios = samples[self.audio] if self.audio is not None else None
        system_prompts = samples[self.system_column] if self.system_column is not None else None
        history_prompts = samples[self.history_column] if self.history_column is not None else None
        query_prompts = samples[self.query_column] if self.query_column is not None else None
        response_prompts = samples[self.response_column] if self.response_column is not None else None

        # samples save raw data
        batch_data = {'samples': copy.deepcopy(samples), 'prompt_data': samples[self.prompt_column]}
        try:
            if self.output_type == OutputType.Embedding:
                batch_data.update(self.prompts_preprocess_for_embedding(prompts))
            elif self.output_type == OutputType.RewardScores:
                batch_data.update(self.prompts_preprocess_for_reward_model(prompts, response_prompts))
            else:
                batch_data.update(
                    self.prompts_preprocess(
                        prompts,
                        images=images,
                        videos=videos,
                        audios=audios,
                        system_prompts=system_prompts,
                        history_prompts=history_prompts,
                        query_prompts=query_prompts,
                    )
                )

        except Exception as e:
            # ailake场景过滤掉表中的mm数据，防止打印刷屏
            # TODO:父类代码中包含子类元素使用不当，错误透出逻辑待重构
            if hasattr(self, 'input_type') and self.input_type == 'ailake':
                bytes_index_list = [self.image_column, self.video, self.audio]
                bytes_index_list = [item for item in bytes_index_list if item is not None]
                samples = [item for index, item in enumerate(samples) if index not in bytes_index_list]
            logger.error(f"prompts_preprocess error: {e} samples: {samples} skip!")
            return None
        return batch_data

    def get_valid_multimodals(
        self,
        input_ids: List[List[int]],
        images: List[List[Union[str, bytes]]] = None,
        videos: List[List[Union[str, bytes]]] = None,
        audios: List[List[Union[str, bytes]]] = None,
    ) -> List[Dict[str, Any]]:
        prompts = [self.tokenizer.decode(item) for item in input_ids]
        batch_size = len(input_ids)

        def _get_valid_multimodals(multimodals, folder, media_token_id, media_token_flag):
            if multimodals is None:
                return [None] * batch_size
            assert media_token_id is not None or media_token_flag is not None
            assert len(multimodals) == batch_size
            if folder:
                multimodals = [
                    [
                        os.path.join(folder, item) if not item.startswith(("http://", "https://")) else item
                        for item in items
                    ]
                    for items in multimodals
                ]
            _source = input_ids if media_token_id else prompts
            _media_token = media_token_id if media_token_id else media_token_flag
            return [items[: _source[i].count(_media_token)] for i, items in enumerate(multimodals)]

        images = _get_valid_multimodals(images, self.image_folder, self.image_token_id, self.template.image_token)
        videos = _get_valid_multimodals(videos, self.video_folder, self.video_token_id, self.template.video_token)
        audios = _get_valid_multimodals(audios, self.audio_folder, self.audio_token_id, self.template.audio_token)

        return [
            {"input_ids": input_id, "images": image, "videos": video, "audios": audio}
            for input_id, image, video, audio in zip(input_ids, images, videos, audios)
        ]

    def prompts_preprocess_for_reward_model(self, prompts, response_prompts):
        conversation_str_list = []
        for index in range(len(prompts)):
            chat = [
                {"role": Role.SYSTEM, "content": self.template.default_system},
                {"role": Role.USER, "content": prompts[index]},
                {"role": Role.ASSISTANT, "content": response_prompts[index]}
            ]
            conversation_str = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False
            )
            conversation_str_list.append(conversation_str)

        input_ids = self.tokenizer(
            conversation_str_list,
            return_tensors="pt",
            padding=True,
        )
        return input_ids

    def prompts_preprocess_for_embedding(
        self,
        prompts,
        system_prompts=None,
        tools=None,
        history_prompts=None,
        query_prompts=None,
        images=None,
        videos=None,
        audios=None,
    ):
        tokenizer_kwargs = {"padding": True,
                            "truncation": True,
                            "return_tensors": "pt"}
        if self.cutoff_len and self.cutoff_len > 0:
            tokenizer_kwargs['max_length'] = self.cutoff_len
        return self.tokenizer(prompts, **tokenizer_kwargs)

    def _replace_text_placeholder(self, text, images, videos, audios):
        if text is None:
            return text
        if images is not None and self.template.image_token is not None:
            text = text.replace(IMAGE_PLACEHOLDER, self.template.image_token)
        if videos is not None and self.template.video_token is not None:
            text = text.replace(VIDEO_PLACEHOLDER, self.template.video_token)
        if audios is not None and self.template.audio_token is not None:
            text = text.replace(AUDIO_PLACEHOLDER, self.template.audio_token)
        return text

    # TODO prompts preprocess 是否可以考虑统一在一个单独的类中，后续所有的对prompts的处理都复用这个类实现就好了。
    def prompts_preprocess(
        self,
        prompts,
        system_prompts=None,
        tools=None,
        history_prompts=None,
        query_prompts=None,
        images=None,
        videos=None,
        audios=None,
    ):
        preprocess_result = {}
        input_ids = []
        for index in range(len(prompts)):
            messages = []
            prompt = prompts[index]
            system = system_prompts[index] if system_prompts else None

            history = history_prompts[index] if history_prompts else None
            if history:
                for old_prompt, old_response in history:
                    messages.append({"role": Role.USER.value, "content": old_prompt})
                    messages.append({"role": Role.ASSISTANT.value, "content": old_response})

            query = query_prompts[index] if query_prompts else None
            if query:
                prompt = f"{prompt}\n{query}"

            tool = tools[index] if tools else None
            messages.append({"role": Role.USER, "content": prompt})
            messages.append({"role": Role.ASSISTANT, "content": ""})
            if self.template.processor:
                for message in messages:
                    message["content"] = self._replace_text_placeholder(message["content"], images, videos, audios)
                system = self._replace_text_placeholder(system, images, videos, audios)
                tool = self._replace_text_placeholder(tool, images, videos, audios)
            input_id = self.template.encode_oneturn(self.tokenizer, messages=messages, system=system, tools=tool)[0]
            if self.cutoff_len and len(input_id) > self.cutoff_len:
                input_id = input_id[: self.cutoff_len]
            input_ids.append(input_id)

        if self.template.processor is not None:
            flatten_features = self.get_valid_multimodals(input_ids, images, videos, audios)
            fail_input_ids_index, batch_mm_features, new_input_ids = [], [], []
            if "vllm" not in self.infer_mode:
                for idx, features in enumerate(flatten_features):
                    processed_ids, mm_features = self.template.plugin.process_single_mm_input(failover=False, **features)
                    if processed_ids is None:
                        # TODO: delete fail_input_ids_index
                        return None
                    new_input_ids.append(processed_ids["input_ids"])
                    batch_mm_features.append(mm_features)
                batch_mm_features = self.template.plugin.mm_collate_fn(batch_mm_features, batch_input_ids=new_input_ids)
                batch_mm_features = {  # TODO: is dtype convert necessary?
                    k: v.to(self.model_dtype) if is_float_tensor(v) else v for k, v in batch_mm_features.items()
                }
                if len(batch_mm_features) > 0:
                    preprocess_result.update(batch_mm_features)
                input_ids = new_input_ids
            else:
                batch_data = []
                for idx, features in enumerate(flatten_features):
                    cur_input_ids = features.pop('input_ids')
                    mm_features = self.template.plugin.get_mm_source_features(**features)
                    self.template.plugin.validate_input_for_vllm(cur_input_ids, self.template, mm_features)
                    if hasattr(self.template.plugin, 'process_single_mm_input_for_vllm'):
                        cur_input_ids, mm_features = self.template.plugin.process_single_mm_input_for_vllm(cur_input_ids, mm_features)
                    new_input_ids.append(cur_input_ids)
                    batch_data.append(mm_features)
                preprocess_result['multi_modal_data'] = batch_data
                input_ids = new_input_ids

        preprocess_result['input_ids'] = input_ids

        return preprocess_result


class FileDataloader(BaseDataloader):
    def __init__(
        self, infer_args, tokenizer, template, processor=None, model_dtype=None, output_type=OutputType.Sequence
    ):
        super().__init__(
            tokenizer, template, infer_args, processor=processor, model_dtype=model_dtype, output_type=output_type
        )

        input_file = infer_args.inputs
        dataset = datasets.load_dataset(
            FILEEXT2TYPE.get(input_file.split(".")[-1]),
            data_files={"test": input_file},
        )["test"]
        if infer_args.input_columns:
            dataset = dataset.select_columns(infer_args.input_columns)

        self.validate_args(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            infer_args.batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn,
            collate_fn=self.default_collate_fn,
        )
        if self.world_size > 1:
            dataloader_config = DataLoaderConfiguration(dispatch_batches=False, even_batches=False)
            self.accelerator = Accelerator(dataloader_config=dataloader_config)
        self.dataloader = self.distribute_file_dataloader(dataloader)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def validate_args(self, dataset):
        column_names = dataset.column_names
        def check_column_exist(column_name):
            assert column_name in column_names and len(dataset[column_name]) > 0, f"{column_name} not in dataset columns: {column_names}"

        check_column_exist(self.prompt_column)
        if self.image_column:
            check_column_exist(self.image_column)
        if self.video:
            check_column_exist(self.video)
        if self.audio:
            check_column_exist(self.audio)
        if self.system_column:
            check_column_exist(self.system_column)
        if self.response_column:
            check_column_exist(self.response_column)
        if self.history_column:
            check_column_exist(self.history_column)
        if self.query_column:
            check_column_exist(self.query_column)

    def distribute_file_dataloader(self, dataloader):
        if self.world_size <= 1:
            return dataloader
        return prepare_data_loader(
            dataloader,
            num_processes=self.world_size,
            process_index=self.rank,
            split_batches=False,
            dispatch_batches=False,
            even_batches=False,
        )

class OpenLMDataloader(BaseDataloader):
    def __init__(
        self, infer_args, tokenizer, template, processor=None, model_dtype=None, output_type=OutputType.Sequence
    ):
        super().__init__(
            tokenizer, template, infer_args, processor=processor, model_dtype=model_dtype, output_type=output_type
        )
        dataset = datasets.load_dataset(infer_args.inputs, split=infer_args.dataset_split)
        if infer_args.input_columns:
            dataset = dataset.select_columns(infer_args.input_columns)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            infer_args.batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn,
            collate_fn=self.default_collate_fn,
        )
        if self.world_size > 1:
            dataloader_config = DataLoaderConfiguration(dispatch_batches=False, even_batches=False)
            self.accelerator = Accelerator(dataloader_config=dataloader_config)
        self.dataloader = self.distribute_file_dataloader(dataloader)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def distribute_file_dataloader(self, dataloader):
        if self.world_size <= 1:
            return dataloader
        return prepare_data_loader(
            dataloader,
            num_processes=self.world_size,
            process_index=self.rank,
            split_batches=False,
            dispatch_batches=False,
            even_batches=False,
        )

class EIPDataloader(BaseDataloader):
    def creat_eip_dataloader(self, infer_args, output_index):
        test_dataset = DistributeDataset(
            table_name=infer_args.inputs,
            num_threads=4,
            capacity=256,
            shuffle=False,
            data_convert=self.row_convert_func,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            infer_args.batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn,
            collate_fn=self.default_collate_fn
        )
        output_config = {
            "odps_table_name": infer_args.outputs,
            "col_index": output_index,
        }
        eip_args = {
            'dataloader': test_dataloader,
            'output_config': output_config,
        }
        if infer_args.infer_mode in ('vllm-async', 'vllm-multi-node'):
            eip_args['slice_mode'] = 'manual'
        try:
            return ElasticInferProcess(**eip_args)
        except TypeError:
            raise TypeError("ElasticInferProcess with slice_mode do not support in this docker image, "
                            "please use 220 and above") from None

    def create_mdp_dataloader(self, infer_args, output_index):
        from mdl.mdp import MDLDataset, MDLDataProcess

        test_dataset = MDLDataset(
            table_name=infer_args.inputs,
            num_threads=4,
            capacity=256,
            row_data_convert=self.row_convert_func,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            infer_args.batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )
        # output to odps
        io_config = {'mode': 'async',
                     'slice_size': 100,
                     'output_conf': {
                         'odps_table_name': infer_args.outputs,
                         "col_index": output_index,
                     },
                     'job_depend_on_this_instance': True}
        return MDLDataProcess(test_dataloader, io_config)

    def __init__(self, infer_args, tokenizer, template, processor=None, model_dtype=None,
                 output_type=OutputType.Sequence):
        super().__init__(
            tokenizer, template, infer_args, processor=processor, model_dtype=model_dtype, output_type=output_type
        )

        # 输出表列数为输入列 + 1（预测列） + （是否还要输出概率值）
        output_columns_cnt = len(infer_args.input_columns) + 1 + \
            int(output_type == OutputType.TokensProbabilities)

        self.input_type = "odps"
        self.row_convert_func = self.odps_data_convert
        if infer_args.tables.startswith("ailake://"):
            self.input_type = "ailake"
            self.row_convert_func = self.ailake_data_convert
            self.ailake_non_multimodel_data_columns = infer_args.ailake_non_multimodel_data_columns
            output_columns_cnt -= len(infer_args.input_columns) - len(self.ailake_non_multimodel_data_columns)

        output_index = tuple(range(output_columns_cnt))
        if len(infer_args.inputs.split(',')) > 1:
            self.dataloader = self.create_mdp_dataloader(infer_args, output_index)
        else:
            self.dataloader = self.creat_eip_dataloader(infer_args, output_index)

        EIPSingleInstance().set_EIP_instance(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)

    def default_collate_fn(self, samples):
        batch_data = super().default_collate_fn(samples)
        # 如果batch报错会返回None
        if batch_data:
            if self.input_type == 'ailake':
                batch_data['samples'] = self.ailake_post_filter(batch_data['samples'])
        return batch_data

    def odps_data_convert(self, sample, **kwargs):
        converted = []
        if self.input_columns:
            sample = [sample[_index] for _index in self.input_columns]
        for sub_sample in sample:
            if isinstance(sub_sample, bytes):
                sub_sample = sub_sample.decode("utf-8")
            elif isinstance(sub_sample, list):
                sub_sample = [
                    sub_sample_item.decode("utf-8") if isinstance(sub_sample_item, bytes) else sub_sample_item
                    for sub_sample_item in sub_sample
                ]
            converted.append(sub_sample)
        return converted

    def ailake_data_convert(self, sample, **kwargs):
        # 相比odps，lake的数据无需decode，但是对于ailake多媒体的bytes类型的数据需要split
        if self.input_columns:
            sample = [sample[_index] for _index in self.input_columns]
        if self.image_column is not None:
            sample[self.image_column] = split_ailake_bytes_data(sample[self.image_column])
        if self.video is not None:
            sample[self.video] = split_ailake_bytes_data(sample[self.video])
        if self.audio is not None:
            sample[self.audio] = split_ailake_bytes_data(sample[self.audio])
        return sample

    def ailake_post_filter(self, samples):
        samples_before_filter = copy.deepcopy(samples)
        samples_after_filter = []
        for index, column_vals in enumerate(samples_before_filter):
            if index in self.ailake_non_multimodel_data_columns:
                samples_after_filter.append(column_vals)
        return samples_after_filter


class ChatApiDataloader(BaseDataloader):
    def __init__(self, tokenizer, template, infer_args, processor=None, model_dtype=None):
        super().__init__(tokenizer, template, infer_args, processor, model_dtype)
        if self.template.image_token:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.template.image_token)
        if self.template.audio_token:
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.template.audio_token)
        if self.cutoff_len is None:
            self.cutoff_len = 2048

    def prompts_preprocess(self, prompts, system_prompts=None, tools=None, history_prompts=None, query_prompts=None,
                           images=None, videos=None, audios=None):
        preprocess_result = {}
        input_ids = []
        if (
            self.processor is not None
            and images
            and len(images) > 0
            and not hasattr(self.processor, "image_seq_length")
            and self.template.image_token not in prompts[0]["content"]
        ):  # llava-like models
            prompts[0]["content"] = self.template.image_token + prompts[0]["content"]

        paired_messages = prompts + [{"role": "assistant", "content": ""}]
        if self.template.processor:
            for message in paired_messages:
                message["content"] = self._replace_text_placeholder(message["content"], images, videos, audios)
            system_prompts = self._replace_text_placeholder(system_prompts, images, videos, audios)
            tools = self._replace_text_placeholder(tools, images, videos, audios)
        input_ids.append(
            self.template.encode_oneturn(
                self.tokenizer,
                messages=paired_messages,
                system=system_prompts,
                tools=tools,
            )[0]
        )

        if self.template.processor is not None:
            if images is not None:
                images = [images]
            if videos is not None:
                videos = [videos]
            if audios is not None:
                audios = [audios]

            flatten_features = self.get_valid_multimodals(input_ids, images, videos, audios)
            fail_input_ids_index, batch_mm_features, new_input_ids = [], [], []
            if "vllm" not in self.infer_mode:
                for idx, features in enumerate(flatten_features):
                    processed_ids, mm_features = self.template.plugin.process_single_mm_input(failover=False, **features)
                    if processed_ids is None:
                        fail_input_ids_index.append(idx)
                        continue
                    new_input_ids.append(processed_ids["input_ids"])
                    batch_mm_features.append(mm_features)
                batch_mm_features = self.template.plugin.mm_collate_fn(batch_mm_features)
                batch_mm_features = {  # TODO: is dtype convert necessary?
                    k: v.to(self.model_dtype) if is_float_tensor(v) else v for k, v in batch_mm_features.items()
                }
                if len(batch_mm_features) > 0:
                    preprocess_result.update(batch_mm_features)
                input_ids = new_input_ids
            else:
                batch_data = []
                for idx, features in enumerate(flatten_features):
                    new_input_ids.append(features.pop('input_ids'))
                    mm_features = self.template.plugin.get_mm_source_features(**features)
                    batch_data.append(mm_features)
                preprocess_result['multi_modal_data'] = batch_data

        preprocess_result["input_ids"] = input_ids
        preprocess_result["prompt_data"] = prompts
        preprocess_result["samples"] = {"fake_prompt_column": [prompts[0]["content"]]}
        return preprocess_result
