import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ENABLE_FLOPS_STATISTICS"] = "-1"
os.environ["MODEL_PARAMS_THOP"] = "-1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"

# 防止accelerate误判当前任务为训练任务进行torch分布式初始化
# TODO:LOCAL_RANK后续存在缺失导致的问题，需要在这里加入更鲁棒的修改
os.environ.pop("LOCAL_RANK", None)
os.environ["PMI_SIZE"] = "0"

# https://github.com/vllm-project/vllm/issues/26922
os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"] = "1"

os.environ["RAY_CGRAPH_get_timeout"] = "600"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from llmtuner.generate import run

if __name__ == '__main__':
    run()
