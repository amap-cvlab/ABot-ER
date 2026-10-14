"""
Copy your model definition files under this directory, and import model class and register here.

Example:
from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_gemma2 import Gemma2Config
from .modeling_gemma2 import Gemma2ForCausalLM
AutoConfig.register(Gemma2Config.model_type, Gemma2Config)
AutoModelForCausalLM.register(Gemma2Config, Gemma2ForCausalLM)

NOTE: If you also want to customize the tokenizer or processor, you can do so in the same way:
from transformers import AutoProcessor, AutoTokenizer
AutoTokenizer.register(Gemma2Config, Gemma2Tokenizer, Gemma2TokenizerFast)
AutoProcessor.register(Gemma2Config, Gemma2Processor)
"""
