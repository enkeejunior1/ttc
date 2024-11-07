import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
import tqdm
import inspect
import logging

from models.teacher import Teacher
from models.configuration_teacher import TeacherConfig
from data import CoTDataset, CoTDataCollator, extract_answer

from utils import get_sep_position
from transformers import AutoModelForMaskedLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pretrained_model(args):
    if args.base_model == "sedd":
        # load model
        from ddms.sedd import SEDD
        model = SEDD.from_pretrained("louaaron/sedd-small")

        # load config
        args.num_vocabs = model.config.tokens
        args.length = model.config.model.length
        args.noise_schedule = model.config.noise.type
        args.graph = 'absorb'
    
    if args.base_model == "mdlm":
        model = AutoModelForMaskedLM.from_pretrained("kuleshov-group/mdlm-owt", trust_remote_code=True)
        
        # load config
        args.num_vocabs = model.config.vocab_size - 1
        args.length = model.config.model_length
        args.noise_schedule = 'loglinear'
        args.graph = 'absorb'
    
    return model, args

def load_diffusion_scheduler(args):
    if args.base_model == "sedd":
        pass
    if args.base_model == "mdlm":
        from ddms import mdlm
        if args.scheduler_name == "euler":
            scheduler = mdlm.EulerScheduler(args)
        if args.scheduler_name == "maskgit":
            scheduler = mdlm.MaskGITScheduler(args)
    return scheduler

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, scheduler, num_inf, loss_fn=None):
    teacher.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all
        input_ids[:, :sep_positions.max()+1] = scheduler.mask_idx
        batch_size = input_ids.shape[0]
        if loss_fn:
            with ctx:
                outputs = loss_fn(input_ids=input_ids_all, labels=labels)
            total_loss += outputs.total_loss.item()
            total_correct_tokens += outputs.total_correct.item()
            total_tokens += outputs.total_tokens
            total_instances += batch_size

        # Generate
        gen_output = scheduler.euler_sample(
            teacher, xt=input_ids, 
            t=1, s=1e-5, num_inference_steps=num_inf
        )
        # gen_output = scheduler.generate(
        #     input_ids=input_ids,
        #     num_inf=num_inf,
        # )
        # Evaluate
        #import pdb; pdb.set_trace()
        for i, (input_ids_all_i, gen_output_i) in enumerate(zip(input_ids_all, gen_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(gen_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            if i == 0:
                print(f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print(f'Target: {tgt_text}')
                print(f'Predicted: {pred_text}')
                print('')
    accuracy = total_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return accuracy, token_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--scheduler_name', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    print(args)

    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print(ptdtype, dtype, device, torch.cuda.current_device())

    # Create Teacher 
    teacher, args = load_pretrained_model(args)
    scheduler = load_diffusion_scheduler(args)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    test_dataset = CoTDataset(tokenizer, args.test_path, 1024)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    # Eval
    teacher.eval()
    accuracy, token_accuracy, ppl = evaluate(test_dataloader, tokenizer, ctx, teacher, scheduler, args.max_new_tokens)
    print(f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')

if __name__ == "__main__":
    main()
