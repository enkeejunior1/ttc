import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2TokenizerFast
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

def load_ddm_loss_fn(args, model, scheduler):
    if args.base_model == "sedd":
        from ddms import sedd
        loss_fn = sedd.Loss(model, scheduler)
    if args.base_model == "mdlm":
        from ddms import mdlm
        loss_fn = mdlm.Loss(model, scheduler)
    return loss_fn

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, max_new_tokens):
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
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]
        with ctx:
            outputs = teacher.compute_loss(input_ids=input_ids_all, labels=labels)
        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        beam_output = teacher.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        # Evaluate
        #import pdb; pdb.set_trace()
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            if i == 0:
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
    accuracy = total_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return accuracy, token_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--base_model', type=str, default='sedd')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--scheduler_name', type=str, default='euler')
    args = parser.parse_args()

    print (args)

    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Teacher 
    teacher, args = load_pretrained_model(args)
    scheduler = load_diffusion_scheduler(args)
    loss_fn = load_ddm_loss_fn(args, teacher, scheduler)

    # Load data
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    eos_token_idx = tokenizer.encode(tokenizer.eos_token)[0]
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(teacher.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    teacher.train()
    teacher = teacher.to(device, ptdtype)

    # Train
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        teacher.train()
        for batch in tqdm.tqdm(train_dataloader):
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            conds = batch['input_ids_only'].to(device) == eos_token_idx
            with ctx:
                outputs = loss_fn(input_ids=input_ids, labels=labels, conds=conds)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ppl = loss.exp().item()
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
            step += 1
        teacher.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))
        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, scheduler, args.max_new_tokens)
        print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        
if __name__ == "__main__":
    main()
