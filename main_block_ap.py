import os
import sys
import random
import numpy as np
import torch
import time
from datautils_block import get_loaders, test_ppl
import torch.nn as nn
from quantize.block_ap import block_ap
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model
from block_weights_inputs import save_layer_weights, attach_hooks_for_dataset

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    layer_names = [
        "model.layers.28.self_attn.q_proj.scales",
        "model.layers.28.self_attn.k_proj.scales",
        "model.layers.28.self_attn.v_proj.scales",
        "model.layers.28.self_attn.o_proj.scales",
        "model.layers.28.mlp.gate_proj.scales",
        "model.layers.28.mlp.up_proj.scales",
        "model.layers.28.mlp.down_proj.scales",
        "model.layers.28.input_layernorm.weight"
    ]
    # Save weights of specified layers
    save_layer_weights(model, layer_names, save_path="./layer_weights.pth")

    results = {}

    # Run evaluation on the specified dataset
    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    
    results = {}

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache directory for faster debugging")
    parser.add_argument("--output_dir", default="./log/", type=str, help="log directory")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="directory for saving quantized model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization to reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,  help="path of resumed quantized model")
    parser.add_argument("--calib_dataset", type=str, default="redpajama",
                        choices=["wikitext2", "ptb", "c4", "mix", "redpajama"],
                        help="Dataset for calibration data.")
    parser.add_argument("--train_size", type=int, default=4096, help="number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="training sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str, default="", help="evaluation tasks list, e.g., piqa,arc_easy,arc_challenge")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=4, help="weight quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weight quantization group size")
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="learning rate for quantization parameters")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="learning rate for full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr / min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--net", type=str, default=None, help="model family name for cache saving")
    parser.add_argument("--max_memory", type=str, default="70GiB", help="maximum memory per GPU")
    parser.add_argument("--early_stop", type=int, default=0, help="early stop after validation loss stops decreasing")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training data to disk to save CPU memory")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    

    # Initialize logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    logger = utils.create_logger(output_dir)
    logger.info(args)
    
     # Check if CUDA is available, and set the device accordingly
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    '''
    if args.resume_quant:
        model, tokenizer = load_quantized_model(args.resume_quant, args.wbits, args.group_size)
        model.to(device)
        logger.info(f"Memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB" if device.type == "cuda" else "Loaded model on CPU")

        # Print the model structure and quantized layers
        print("Model Architecture:\n", model)
        print("Quantized Layers:")
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:  # Assuming quantized layers aren't float32
                print(f"Layer: {name} | Shape: {param.shape} | dtype: {param.dtype}")
        
        # Summary of total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    else:
        # If loading from scratch (non-quantized), follow similar steps here
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config).half().to(device)
        
        # Print model structure
        print("Model Architecture:\n", model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")
    '''



    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"net is None, setting as {args.net}")
    if args.resume_quant:
        model, tokenizer = load_quantized_model(args.resume_quant, args.wbits, args.group_size)
        model.to(device)  # Ensure the model is moved to the appropriate device
        logger.info(f"Memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB" if device.type == "cuda" else "Loaded model on CPU")
    else:
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config).half().to(device)

        for param in model.parameters():
            param.requires_grad = False

        # Quantization
        if args.wbits < 16:
            logger.info("=== start quantization ===")
            tick = time.time()
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                valloader = torch.load(cache_valloader)
                logger.info(f"Loaded dataloaders from cache.")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)
            block_ap(
                model,
                args,
                trainloader,
                valloader,
                logger,
            )
            logger.info(f"Quantization completed in {time.time() - tick:.2f}s")
    torch.cuda.empty_cache()
    if args.save_quant_dir:
        model.save_pretrained(args.save_quant_dir)
        tokenizer.save_pretrained(args.save_quant_dir)
        logger.info("Model saved successfully")
    evaluate(model, tokenizer, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
