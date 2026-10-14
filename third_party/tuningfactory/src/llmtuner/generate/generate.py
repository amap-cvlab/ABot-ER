import torch
from tqdm import tqdm
from transformers import set_seed
import asyncio
import traceback
from ..extras.misc import log_meta_to_openlm
from ..extras.logging import get_logger
from .engine import create_generate_engine
from .infer_args import get_generate_args
from .io import create_dataloader_and_writer
from .engine import VllmAsyncGenerateEngine
from .dataloader import EIPDataloader
from .mdl_metrics import set_model_name, set_architectures
logger = get_logger(__name__)

def run():
    model_args, finetuning_args, generating_args, infer_args = get_generate_args()
    set_model_name(model_args.model_name_or_path)
    infer_engine = create_generate_engine(model_args, infer_args, finetuning_args, generating_args)
    set_architectures(infer_engine.config)
    log_meta_to_openlm(finetuning_args=finetuning_args,infer_args=infer_args)
    dataloader, writer = create_dataloader_and_writer(infer_args,
                                                      infer_engine.get_tokenizer(),
                                                      infer_engine.get_template(),
                                                      infer_engine.get_processor(),
                                                      infer_engine.get_model_dtype(),
                                                      infer_engine.get_num_return(),
                                                      infer_engine.get_output_type())
    set_seed(infer_args.seed)
    try:
        predict(dataloader, writer, infer_engine)
    finally:
        writer.close()


async def producer(dataloader, writer, input_queue, consumer_cnt):
    try:
        PRODUCER_PUT_TIMEOUT = 60 * 60
        for index, data in enumerate(tqdm(dataloader, desc="Generating")):
            if isinstance(dataloader, EIPDataloader):
                slice_info, source_data = data
                if source_data:
                    source_data['slice_info'] = slice_info
                    data = source_data
                else:
                    writer.commit_slice(slice_info)
                    continue
            if not data:
                continue

            await asyncio.wait_for(input_queue.put(data), timeout=PRODUCER_PUT_TIMEOUT)
        for _ in range(consumer_cnt):
            await input_queue.put(None)
    except asyncio.CancelledError:
        logger.info(f"[TuningFactory Async] producer was cancelled")   


async def pusher(writer, output_queue):
    try:
        while True:
            data = await output_queue.get()
            output_queue.task_done()
            if data is None:
                break
            output_raw_data, output_data = data
            if not output_data:
                # 如果output_data是None，表示consumer遇到了异常，那么直接上传slice_info防止任务阻塞
                logger.info(f"[TuningFactory Async] commit_slice {output_raw_data['slice_info']}")   
                writer.commit_slice(output_raw_data['slice_info'])
            elif output_raw_data['samples']:
                writer.write(output_raw_data, output_data)
    except asyncio.CancelledError:
        logger.info(f"[TuningFactory Async] pusher was cancelled")   

async def consumer(infer_engine, input_queue, output_queue, consumer_id):
    from vllm.v1.engine.exceptions import EngineDeadError
    try:
        while True:
            data = await input_queue.get()
            input_queue.task_done()
            if data is None:
                break
            try:
                processed_item = await infer_engine.predict_batch(data)
            except EngineDeadError:
                logger.info(f"[TuningFactory Async consumer] Task fail because:\n{traceback.format_exc()}\n\nstop job！\n")
                raise
            except Exception as e:
                logger.info(f"[TuningFactory Async consumer] Task fail because:\n{traceback.format_exc()}\n\npass\n")
                # 如果有异常数据，那么输出数据置为None放入output_queue中
                processed_item = (data, None)
            await output_queue.put(processed_item)
    except asyncio.CancelledError:
        logger.info(f"[TuningFactory Async] consumer-{consumer_id} was cancelled")   

async def predict_in_asyncio(dataloader, writer, infer_engine):
    VLLM_PARALLELISM_WORKER_CNT = 128
    PRODUCER_BUFFER_SIZE = 40
    input_queue = asyncio.Queue(maxsize=PRODUCER_BUFFER_SIZE)
    output_queue = asyncio.Queue()

    producer_task = asyncio.create_task(producer(dataloader, writer, input_queue, VLLM_PARALLELISM_WORKER_CNT))
    pusher_task = asyncio.create_task(pusher(writer, output_queue))
    consumers = [asyncio.create_task(consumer(infer_engine, input_queue, output_queue, consumer_id)) for consumer_id in range(VLLM_PARALLELISM_WORKER_CNT)]

    # await producer_task

    try:
        await asyncio.gather(producer_task, *consumers)
    except Exception as e:
        logger.info(f"[TuningFactory Async] Task failed: {e}\n{traceback.format_exc()}")
        raise

    await output_queue.put(None)
    await pusher_task


def predict_in_sync_mode(dataloader, writer, infer_engine):
    for index, data in enumerate(tqdm(dataloader, desc="Generating")):
        if not data:
            continue
        output_data = infer_engine.predict_batch(data)
        writer.write(data, output_data)


@torch.inference_mode()
def predict(dataloader, writer, infer_engine):
    if isinstance(infer_engine, VllmAsyncGenerateEngine):
        asyncio.run(predict_in_asyncio(dataloader, writer, infer_engine))
    else:
        predict_in_sync_mode(dataloader, writer, infer_engine)
