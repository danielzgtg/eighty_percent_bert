if True:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #os.environ['HF_DATASETS_OFFLINE'] = '1'
    #os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import warnings
    warnings.filterwarnings('ignore')
    import transformers
    pipeline = transformers.pipeline
    Conversation = transformers.Conversation
    MBartForConditionalGeneration = transformers.MBartForConditionalGeneration
    MBart50TokenizerFast = transformers.MBart50TokenizerFast
    AutoModel = transformers.AutoModel
    AutoConfig = transformers.AutoConfig
    AutoTokenizer = transformers.AutoTokenizer
    transformers.set_seed(42)  # Needed for GPT2 but doesn't work :(
    transformers.logging.set_verbosity_error()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    gpu_device = 0 if torch.cuda.is_available() else -1
    # TODO force offline once https://github.com/UKPLab/sentence-transformers/issues/1725 is implemented upstream
    from keybert import KeyBERT


def load_asset(name: str) -> str:
    with open(f"assets/{name}.txt") as f:
        result = f.read()
    result = result.strip()
    if '\n' in result:
        raise ValueError
    return result


def to_gpu(model):
    if gpu_device == 0:
        return model.to("cuda:0")
    return model
