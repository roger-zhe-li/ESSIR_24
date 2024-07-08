from prettytable import PrettyTable
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval
import argparse
import torch

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Transformers' model name or path")
    args = parser.parse_args()

    # Load the LLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                device_map='auto',
                                                output_hidden_states=True,
                                                trust_remote_code=True,)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0  # Set the padding token.
    tokenizer.padding_side = "left"  # Allow batched inference

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Batcher is the KEY function that you need to modify to develop you own embedding extraction method
    ####################################################################################################
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states
            last_layer = hidden_states[-1]
            outputs = ((last_layer * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1))
            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

        return outputs.cpu()
    ####################################################################################################

    # Obtain the embeddings and evaluate them
    results = {}
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    PATH_TO_DATA = "./data/STS/"
    params = dotdict({'task_path': PATH_TO_DATA, 'batch_size':16})
    eval_dict = {"STS12": STS12Eval, "STS13": STS13Eval, "STS14": STS14Eval, "STS15": STS15Eval, "STS16": STS16Eval}
    for task in tasks:
        fpath = task + '-en-test'
        evaluation = eval_dict[task](params.task_path + fpath) # Load the dataset
        result = evaluation.run(params, batcher) # Obtain the embeddings and evaluate them
        results[task] = result

    # Print results using print_table()
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        task_names.append(task)
        if task in results:
            scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(tasks)))
    print_table(task_names, scores)
    
    # Write results to file
    with open('./sts-org-results', 'a') as f:
        model_name = args.model_name_or_path.split('/')[-1]
        f.write(model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')

if __name__ == "__main__":
    main()