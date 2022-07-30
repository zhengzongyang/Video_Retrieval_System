import logging
import random
import os
import torch
import numpy as np
from .service.utils.utils import get_logger
from .service.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from .service.model import CLIP2Video_Encoder
from .config.ModelConfig import get_args


def set_seed_logger(args):
    """Initialize the seed and environment variable

    Args:
        args: the hyper-parameters.

    Returns:
        args: the hyper-parameters modified by the random seed.

    """

    global logger

    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # get logger 设置日志的输出地址
    logger = get_logger(os.path.join(args.output_dir))

    return args

def init_device(args, local_rank):
    """Initialize device to determine CPU or GPU

     Args:
         args: the hyper-parameters
         local_rank: GPU id

     Returns:
         devices: cuda
         n_gpu: number of gpu

     """
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def get_encoder(args, device):
    """
    函数功能：构建编码器
    """
    # load model
    model_file = os.path.join(args.checkpoint, "pytorch_model.bin.{}".format(args.model_num))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
    # Prepare model
    encoder = CLIP2Video_Encoder.from_pretrained(args.cross_model, cache_dir=None, state_dict=model_state_dict,
                                       task_config=args)
    encoder.to(device)
    encoder.eval()
    return encoder

args = get_args()
print(args)
# 设置随机数种子
args = set_seed_logger(args)
# 初始化GPU
device, n_gpu = init_device(args, args.local_rank)
# 设置CLIP 的 tokenizer（分词器）
tokenizer = ClipTokenizer()
# 初始化模型
encoder = get_encoder(args, device)