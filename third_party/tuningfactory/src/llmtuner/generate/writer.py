import copy
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List

from ..data import create_odps_table
from ..extras.logging import get_logger
from .constant import OutputType
from .eip import EIPSingleInstance
from .utils import is_external_cluster
from .mdl_metrics import MDLMetricLogger


logger = get_logger(__name__)
mdl_logger = MDLMetricLogger()


class FileWriter(object):
    def __init__(self, output):
        output_dir = os.path.dirname(output)
        os.makedirs(output_dir, exist_ok=True)
        if int(os.getenv("WORLD_SIZE", "1")) > 1:
            output += f"_{os.getenv('RANK', '0')}"
            logger.info(f"outputs_file is changed to {output}")

        self.output_file = open(output, "w", encoding="utf-8")
        self.input_tokens = 0
        self.output_tokens = 0
        self.query_cnt = 0

    def sample_iterator(self, samples: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
        # 获取batch数据每列的元素个数
        column_item_cnt = dict(zip(samples.keys(), list(map(len, list(samples.values())))))
        assert len(set(column_item_cnt.values())) == 1, (
            f"The data lengths of each column do not correspond, got {column_item_cnt}"
        )
        for sample in zip(*samples.values()):
            yield dict(zip(samples.keys(), sample))

    def write(self, data, output_data):
        data["samples"]["predict"] = output_data.predict
        self.input_tokens += output_data.input_generate_token_cnt
        self.output_tokens += output_data.output_generate_token_cnt
        self.query_cnt += len(output_data.predict)
        if output_data.generated_tokens_with_probs:
            data["samples"]["tokens_probs"] = output_data.generated_tokens_with_probs
        if output_data.audio_path is not None:
            data["samples"]["audio"] = output_data.audio_path
        for sample in self.sample_iterator(data["samples"]):
            # 如果预测为None,就不写出
            if sample["predict"] is not None:
                self.output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def close(self):
        self.output_file.close()
        mdl_logger.record_tokens(self.query_cnt, self.input_tokens, self.output_tokens)


class EIPWriter(object):
    def __init__(self, inputs, outputs, input_columns, num_return, output_type, write_buffer_size):
        if is_external_cluster():
            logger.info("is_external_cluster true, skip create_odps_table! Make sure you have created the output table ahead of time")
        else:
            create_odps_table(inputs, outputs, input_columns, num_return, output_type)
        self.output_scores = output_type == OutputType.TokensProbabilities
        self.writer = EIPSingleInstance()
        self.write_buffer_size = write_buffer_size
        self.writer_data_buffer = []
        self.writer_slice_buffer = []
        
        self.latest_write_time = time.time()
        self.input_tokens = 0
        self.output_tokens = 0
        self.query_cnt = 0

    def write(self, data, output):
        samples = data["samples"]
        slice_info = data.get("slice_info", None)
        output_data = output.predict

        if output.input_generate_token_cnt is not None:
            assert output.output_generate_token_cnt is not None
            self.input_tokens += output.input_generate_token_cnt
            self.output_tokens += output.output_generate_token_cnt
        else:
            assert output.output_generate_token_cnt is None
        
        self.query_cnt += len(output_data)
        if time.time() - self.latest_write_time > 10:
            mdl_logger.record_tokens(self.query_cnt, self.input_tokens, self.output_tokens)
            self.input_tokens = 0
            self.output_tokens = 0
            self.query_cnt = 0
            self.latest_write_time = time.time()

        # transpose samples and append responses
        if self.output_scores:
            generated_tokens_with_probs = output.generated_tokens_with_probs
            data = [
                list(subsample) + [output_data[i]] + [json.dumps(generated_tokens_with_probs[i], ensure_ascii=False)]
                for i, subsample in enumerate(zip(*samples))
                if output_data[i] is not None
            ]
        else:
            data = [
                list(subsample) + [output_data[i]]
                for i, subsample in enumerate(zip(*samples))
                if output_data[i] is not None
            ]
        if slice_info:
            self.writer_data_buffer += data
            self.writer_slice_buffer += slice_info
            if len(self.writer_data_buffer) >= self.write_buffer_size:
                logger.info(f"{datetime.now()} [EIPWriter]:trace data length:{len(self.writer_data_buffer)}")
                write_data = copy.deepcopy(self.writer_data_buffer)
                write_slice = copy.deepcopy(self.writer_slice_buffer)
                self.writer.trace(write_data, write_slice)
                self.writer_data_buffer.clear()
                self.writer_slice_buffer.clear()
        else:
            self.writer.trace(data)

    def commit_slice(self, slice_info):
        self.writer.trace(data=None, slice_info=slice_info)

    def close(self):
        if self.writer_data_buffer:
            self.writer.trace(self.writer_data_buffer, self.writer_slice_buffer)
            logger.info(f"{datetime.now()} [EIPWriter]:trace data length:{len(self.writer_data_buffer)}")
        self.writer.close()
        mdl_logger.record_tokens(self.query_cnt, self.input_tokens, self.output_tokens)



class OdpsTunnelWriter(object):
    def __init__(self, inputs, outputs, field_mapping):
        if outputs.startswith("odps://"):
            import common_io

            table = outputs
            writer = common_io.table_tunnel.TableWriter(table, slice_id=0)
            self._table_schema = writer._writer_session.schema
            odps_cols = self._table_schema.get_columns()
            self._odps_col_names = [col.name for col in odps_cols]
            logger.info(f"{table} col_names: {self._odps_col_names}")
            self._col_indices = list(range(len(odps_cols)))
            self.path = inputs
            self.field_mapping = field_mapping
            try:
                self.writer = writer
            except Exception as e:
                logger.error(f"{table} init failed: {e}")
                writer.close()
        else:
            raise NotImplementedError

    def write(self, data, output):
        samples = data["samples"]
        output_data = output.predict
        values = []
        for i, col in enumerate(self._odps_col_names):
            col_value = None
            if col in self.field_mapping:
                col_value = samples[self.field_mapping[col]]
            elif col == "prompt":
                col_value = samples["prompt"]
            elif col == "answer":
                col_value = output_data
            elif col == "dataset":
                col_value = [self.path for _ in range(len(output_data))]
            if col_value is None:
                col_value = [None for _ in range(len(output_data))]
            values.append(col_value)
        values = [list(x) for x in zip(*values)]
        self.writer.write(values, self._col_indices)

    def close(self):
        self.writer.close()
