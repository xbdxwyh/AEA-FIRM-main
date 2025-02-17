import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.proj_token_num = 24
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    args.nmb_prototypes = 1000 
    args.prot_temp = 0.1
    args.prot_eps = 0.5
    args.sink_iter = 3
    args.crops_for_assign = [0,1,2]
    args.cont_queue_size=args.batch_size*3
    args.queue_size=0
    args.queue_epoch=3
    args.decoder_depth=1
    args.atr_length = 14
    # args.
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    #if args.dataset_name != "AGTBPRSG":
    height = args.proj_token_num // (args.img_size[1]//args.stride_size)
    args.enl_img_size = (args.img_size[0] + height * args.stride_size, args.img_size[1])
    model = build_model(args, num_classes)
    logger.info(model.proj_dec_cfg)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if model.linear_proj is not None:
        logger.info(model.linear_proj)
    
    model.set_sent_list(train_loader.dataset.tokenizer,text_length=train_loader.dataset.text_length, truncate=train_loader.dataset.truncate)
    model.to(device)
    #model.text_pe = model.text_pe.to(device)
    save_train_configs(args.output_dir, args)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)