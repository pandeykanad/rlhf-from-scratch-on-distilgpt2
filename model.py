"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer

def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    return AutoTokenizer.from_pretrained(model_name)

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM
def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    return AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).eval()

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    # TODO: tokenize prompt, generate continuation greedily, decode and return as a string
    tok = tokenizer.encode(prompt, return_tensors="pt")
    out = model.generate(tok, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0])

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    return torch.argmax(logits).item()

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    d = torch.softmax(logits/temperature, -1)
    return torch.multinomial(d, num_samples=1).item()

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # TODO: keep the k largest entries of logits and set the rest to -inf.
    k=min(k, logits.size(-1))
    threshold = torch.topk(logits, k).values[-1]
    return torch.where(logits >= threshold,logits, float('-inf'))

# Step 8 - top_p_filter
def top_p_filter(logits, p):
    # TODO: mask logits outside the smallest cumulative-probability nucleus of size p.
    logits = np.array(logits)
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    sort_idx = np.argsort(probs.tolist())[::-1]
    sort_probs = np.array(probs)[sort_idx]
    cum = np.cumsum(sort_probs)
    mask = cum > p
    mask[1:] = mask[:-1]
    mask[0] = False
    rem_idx = sort_idx[mask]
    logits[rem_idx] = float('-inf')
    return torch.tensor(logits)

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    # TODO: return a small in-memory list of {'prompt', 'response'} dicts for SFT
    prompt = "prompt"
    response = "response"
    dic = [
        {prompt: "Hey", response: "Hello"}, 
        {prompt: "Thank You", response: "You're welcome!"},
        {prompt: "I am sorry", response: "It's okay!"},
        {prompt: "Goodbye", response: "See you later!"}
    ]
    return dic

# Step 10 - format_example
def format_example(example):
    # TODO: render {'prompt','response'} into one training string with role markers
    return f"### Instruction:\n{example["prompt"]}\n\n### Response:\n{example["response"]}"

# Step 11 - apply_template
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
    strs = []
    for example in examples:
        strs.append(format_example(example))
    return strs

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    return encoded['input_ids']

# Step 13 - build_labels
def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels
    return input_ids.copy()

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # TODO: replace the first prompt_length entries of labels with -100 and return the new list
    prompt_length = min(len(labels), prompt_length)
    masked_labels = [-100] * prompt_length + labels[prompt_length:]
    return masked_labels

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    # TODO: right-pad a list of token id sequences to the longest length using pad_id
    maxlen = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [pad_id]*(maxlen - len(seq)) for seq in sequences]
    return padded_sequences

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    # TODO: return a same-shape 0/1 mask with 1 where token != pad_id else 0
    return [[ (0 if idx == pad_id else 1) for idx in seq] for seq in padded_ids]

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    # TODO: pad input_ids and labels, build attention mask, return dict of LongTensors
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    padded_input_ids = pad_batch(input_ids, pad_id)
    padded_labels = pad_batch(labels, -100)

    mask = make_attention_mask(padded_input_ids, pad_id)
    return {
        'input_ids': torch.tensor(padded_input_ids),
        'labels': torch.tensor(padded_labels),
        'attention_mask': torch.tensor(mask)
    }

# Step 18 - iterate_minibatches
import random
def iterate_minibatches(examples, batch_size, seed=0):
    # TODO: yield shuffled minibatches of size batch_size from examples (deterministic per seed).
    rng = random.Random(seed)
    shuffled_examples = list(examples)
    rng.shuffle(shuffled_examples)
    mini_batch = []
    for i in range(0, len(shuffled_examples), batch_size):
        yield shuffled_examples[i:i+batch_size]

# Step 19 - train_val_split
def train_val_split(examples, val_ratio=0.2, seed=0):
    # TODO: deterministically split examples into (train, val) using seed and val_ratio
    val_size = int(len(examples)*val_ratio)
    rng = random.Random(seed)
    shuffled_examples = examples.copy()
    rng.shuffle(shuffled_examples)
    val = shuffled_examples[:val_size]
    train = shuffled_examples[val_size:]
    return (train,val)

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    # TODO: drop the last logit position and the first label position so token t predicts t+1
    shifted_logits = logits[:,:-1,:]
    shifted_labels = labels[:,1:]
    return shifted_logits, shifted_labels

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # TODO: reduce (B, T-1, V) logits and (B, T-1) labels to a scalar loss tensor.
    logits_flat = shift_logits.reshape(-1, shift_logits.size(-1))
    labels_flat = shift_labels.reshape(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
    return loss

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # TODO: initialize state on first call, then update moments and apply the decoupled AdamW step
    if not state:
        state['step']=0
        state['m']=torch.zeros_like(param)
        state['v']=torch.zeros_like(param)
    
    state['step'] += 1
    param.mul_(1.0 - lr*weight_decay)
    state['m'].mul_(betas[0]).add_(grad, alpha=1.0 - betas[0])
    state['v'].mul_(betas[1]).addcmul_(grad, grad, value=1.0 - betas[1])

    bias1 = 1.0 - betas[0] ** state['step']
    bias2 = 1.0 - betas[1] ** state['step']

    m_hat = state['m'] / bias1
    v_hat = state['v'] / bias2

    param.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)

    return state

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    # TODO: return a linear warmup multiplier in [0, 1] given the current step and warmup window.
    if warmup_steps == 0:
        return 1.0
    return min(1, float(step/warmup_steps))

# Step 24 - clip_grad_norm
def clip_grad_norm(grads, max_norm):
    # TODO: compute the global L2 norm of grads and rescale in place if it exceeds max_norm.
    global_norm = 0.0
    for g in grads:
        global_norm += (g**2).sum().item()
    global_norm = global_norm ** 0.5
    if global_norm <= max_norm:
        return global_norm
    for g in grads:
        g.mul_(max_norm/global_norm)
    return global_norm

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor
    return torch.stack(grad_list).mean(dim=0)

# Step 26 - sft_train_step
import torch

def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # TODO: forward the batch, compute shifted cross-entropy loss, backprop, step optimizer
    optimizer.zero_grad()
    outputs = model(
        input_ids = batch['input_ids'],
        attention_mask = batch.get('attention_mask')
    )
    logits = outputs['logits']
    
    shift_logits, shift_labels = shift_logits_and_labels(logits, batch['labels'])
    loss = cross_entropy_loss(shift_logits, shift_labels)

    loss.backward()
    optimizer.step()

    return loss.item()

# Step 27 - evaluate_loss
import torch

def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    # TODO: iterate batches under no_grad, shift logits/labels, average cross-entropy.
    total_loss = 0.0
    if len(batches) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            outputs = model(
                input_ids = batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs['logits']
            shift_logits, shift_labels = shift_logits_and_labels(logits, batch['labels'])

            loss = cross_entropy_loss(shift_logits, shift_labels)
            total_loss += loss.item()
        
    return total_loss/len(batches)

# Step 28 - lora_delta
def lora_delta(A, B, alpha, r):
    # TODO: build the scaled low-rank weight update from factors A and B.
    return (alpha/r)*torch.matmul(B, A)

# Step 29 - lora_linear_forward
def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    out = x @ (base_weight + lora_delta(A, B, alpha, r)).T
    if bias is not None:
        out += bias
    return out

# Step 30 - init_lora_weights
import torch

def init_lora_weights(in_features, out_features, r, seed=0):
    """Return (A, B) LoRA factors with random A and zero B so the initial delta is zero."""
    # TODO: seed torch, build A of shape (r, in_features) and B of shape (out_features, r)
    torch.manual_seed(seed)
    A = torch.randn(r, in_features, dtype=torch.float32) * 0.01
    B = torch.zeros(out_features, r, dtype = torch.float32)
    return A,B

# Step 31 - freeze_base_params
def freeze_base_params(model):
    # TODO: set requires_grad=False on every base parameter, leaving LoRA adapters trainable.
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    return model

# Step 32 - count_trainable_params
def count_trainable_params(model):
    # TODO: sum p.numel() over parameters with requires_grad=True
    trainable = 0
    for p in model.parameters():
        if p.requires_grad == True:
            trainable += p.numel()
    return trainable

# Step 33 - merge_lora
def merge_lora(base_weight, lora_a, lora_b, scaling):
    return base_weight + scaling * lora_b @ lora_a

# Step 34 - build_synthetic_preference_dataset
def build_synthetic_preference_dataset(num_examples=8, seed=0):
    # TODO: return a list of {'prompt','chosen','rejected'} dicts of length num_examples.
    rng = random.Random(seed)
    candidates = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "I do not know."
        },
        {
            "prompt": "What is 2 + 2?",
            "chosen": "2 + 2 equals 4.",
            "rejected": "2 + 2 equals 5."
        },
        {
            "prompt": "How do you say hello in Spanish?",
            "chosen": "Hello in Spanish is 'Hola'.",
            "rejected": "Hello in Spanish is 'Bonjour'."
        },
        {
            "prompt": "What color is the sky on a clear day?",
            "chosen": "The sky is blue on a clear day.",
            "rejected": "The sky is green on a clear day."
        }
    ]
    dataset = [candidates[(i+seed) % len(candidates)].copy() for i in range(num_examples)]
    return dataset

# Step 35 - format_preference
def format_preference(example):
    # TODO: return a dict with 'chosen_text' and 'rejected_text' built from the prompt and each answer.
    return {
        'chosen_text': example['prompt'] + ' ' + example['chosen'],
        'rejected_text': example['prompt'] + ' ' + example['rejected']
    }

# Step 36 - reward_head_forward
import torch

def reward_head_forward(hidden_state, weight, bias):
    """Map a final hidden state to a scalar reward via a linear projection."""
    # TODO: project hidden_state (B, D) through weight (D,) plus scalar bias to get (B,) rewards
    out = hidden_state @ weight.flatten() + bias
    return out

# Step 37 - pairwise_reward_loss
import torch
import torch.nn.functional as F

def pairwise_reward_loss(chosen_reward, rejected_reward):
    """Bradley-Terry pairwise loss: mean(-log_sigmoid(chosen - rejected))."""
    # TODO: return the mean negative log-sigmoid of (chosen_reward - rejected_reward)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()

# Step 38 - reward_bce_loss
import numpy as np

def reward_bce_loss(chosen_reward, rejected_reward):
    # TODO: BCE-style reward loss with chosen as positives and rejected as negatives.
    chosen_loss = np.logaddexp(0, -chosen_reward)
    rejected_loss = np.logaddexp(0, rejected_reward)
    return np.mean(chosen_loss + rejected_loss) / 2.0

# Step 39 - pairwise_accuracy
import torch

def pairwise_accuracy(chosen_reward, rejected_reward):
    """Fraction of pairs where chosen_reward > rejected_reward."""
    return (chosen_reward > rejected_reward).float().mean().item()

# Step 40 - reward_train_step
import torch

def reward_train_step(model, reward_head, batch, optimizer):
    # TODO: forward chosen+rejected, score last token, compute loss/acc, step optimizer
    optimizer.zero_grad()

    def forward_and_score(input_ids, attention_mask):
        hidden_states = model(input_ids, attention_mask)
        last_token_idx = (attention_mask.sum(dim=1) - 1)
        D = hidden_states.size(-1)
        gather_idx = last_token_idx.view(-1, 1, 1).expand(-1,1,D).long()
        last_hidden = torch.gather(hidden_states, dim=1, index=gather_idx).squeeze(1)
        return reward_head_forward(last_hidden, reward_head.weight, reward_head.bias)
    
    chosen_rewards = forward_and_score(batch['chosen_input_ids'], batch['chosen_attention_mask'])
    rejected_rewards = forward_and_score(batch['rejected_input_ids'], batch['rejected_attention_mask'])

    loss = pairwise_reward_loss(chosen_rewards, rejected_rewards)
    accuracy = pairwise_accuracy(chosen_rewards, rejected_rewards)

    loss.backward()
    optimizer.step()

    return {
        'loss': float(loss),
        'accuracy': float(accuracy)
    }

# Step 41 - sequence_logprob
import torch
import torch.nn.functional as F

def sequence_logprob(logits, token_ids):
    """Sum log probabilities of the selected tokens along the sequence dimension."""
    # TODO: return a scalar tensor equal to sum_t log_softmax(logits)[t, token_ids[t]]
    log_probs = F.log_softmax(logits, dim=-1)
    token_ids_2d = token_ids.unsqueeze(-1)
    selected_log_probs = log_probs.gather(dim=-1, index=token_ids_2d).squeeze(-1)
    return selected_log_probs.sum()

# Step 42 - per_token_kl
import numpy as np

def per_token_kl(policy_logprobs, ref_logprobs):
    """Per-token KL estimate between policy and reference log-probs."""
    # TODO: return the per-token KL contribution used in the PPO penalty
    return (policy_logprobs - ref_logprobs)

# Step 43 - compute_returns
import numpy as np

def compute_returns(rewards, gamma=0.99):
    """Return the discounted return at each timestep as a 1D numpy array."""
    # TODO: turn a per-timestep reward sequence into discounted returns
    T=len(rewards)
    discounted_rewards = np.zeros(T+1)
    for t in reversed(range(T)):
        discounted_rewards[t] = discounted_rewards[t+1] * gamma + rewards[t]
    return discounted_rewards[:T]

# Step 44 - gae_advantages
def gae_advantages(rewards, values, gamma=0.99, lam=0.95):
    # TODO: compute GAE advantages of shape (T,) from rewards (T,) and values (T+1,)
    T = len(rewards)
    gae = np.zeros(T)
    A = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        A = delta + gamma * lam * A
        gae[t] = A
    return gae

# Step 45 - policy_ratio
import torch

def policy_ratio(new_logprobs, old_logprobs):
    """Return the PPO importance ratio exp(new - old) elementwise."""
    # TODO: exponentiate the difference between new and old log probabilities
    return torch.exp(new_logprobs - old_logprobs)

# Step 46 - clipped_surrogate
import torch

def clipped_surrogate(ratio, advantages, clip_eps=0.2):
    """PPO clipped surrogate loss (scalar tensor to minimize)."""
    # TODO: combine ratio and advantages via the PPO clipped objective and return a scalar loss
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    clipped = clipped_ratio * advantages
    unclipped = ratio * advantages
    return -torch.min(unclipped, clipped).mean()

# Step 47 - value_function_loss
import torch

def value_function_loss(values, returns):
    """Mean squared error between predicted values and target returns."""
    # TODO: compute mean((values - returns) ** 2) as a scalar tensor
    return torch.mean((values - returns) ** 2)

# Step 48 - entropy_bonus
import torch

def entropy_bonus(logits):
    """Return mean categorical entropy of the distribution defined by `logits` over the last axis."""
    # TODO: softmax over the vocab axis, compute -sum(p * log p), then average.
    return -torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1).mean()

# Step 49 - ppo_loss
import torch

def ppo_loss(ratio, advantages, values, returns, logits, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    # TODO: combine clipped surrogate, value loss, and entropy bonus into the full PPO loss dict.
    policy_loss = clipped_surrogate(ratio, advantages, clip_eps)
    value_loss = value_function_loss(values, returns)
    entropy = entropy_bonus(logits)

    total_loss = policy_loss + (vf_coef * value_loss) - (ent_coef * entropy)
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy
    }

# Step 50 - kl_penalized_reward
import torch

def kl_penalized_reward(reward, kl, beta=0.1):
    """Return reward shaped by a KL penalty against a reference policy."""
    # TODO: combine the reward model score with a beta-weighted KL penalty
    return (reward - beta * kl)

# Step 51 - batch_sequence_logprob
import torch
import torch.nn.functional as F

def batch_sequence_logprob(logits, token_ids, attention_mask=None):
    # TODO: return a (B,) tensor of summed token log probabilities, respecting attention_mask.
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        token_log_probs = token_log_probs * attention_mask

    return token_log_probs.sum(dim=1)

# Step 52 - dpo_logratios
import torch

def dpo_logratios(policy_chosen_logps, policy_rejected_logps):
    """Return policy_chosen_logps - policy_rejected_logps elementwise."""
    return policy_chosen_logps - policy_rejected_logps

# Step 53 - dpo_ref_logratios
import torch

def dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps):
    # TODO: return per-example chosen minus rejected reference log probabilities
    return (ref_chosen_logps - ref_rejected_logps)

# Step 54 - dpo_loss
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Return the DPO loss as a scalar torch tensor."""
    return -F.logsigmoid(beta * (dpo_logratios(policy_chosen_logps, policy_rejected_logps) - dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps))).mean()

# Step 55 - ipo_loss
import torch

def ipo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # TODO: regress (policy_logratios - ref_logratios) toward the IPO target 1/(2*beta)
    policy_logratios = dpo_logratios(policy_chosen_logps, policy_rejected_logps)
    ref_logratios = dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps)
    h = policy_logratios - ref_logratios
    return torch.mean((h - (1.0/(2.0*beta))) ** 2)

# Step 56 - kto_loss
import torch

def kto_loss(policy_logps, ref_logps, labels, beta=0.1):
    # TODO: implement KTO loss for unpaired desirable/undesirable examples.
    r = beta * (policy_logps - ref_logps)
    loss_d = 1 - torch.sigmoid(r)
    loss_ud = 1 - torch.sigmoid(-r)
    loss = labels * loss_d + (1 - labels) * loss_ud
    return loss.mean()

# Step 57 - orpo_loss
def orpo_loss(policy_chosen_logps, policy_rejected_logps, sft_loss, lambda_or=0.1):
    # TODO: return sft_loss + lambda_or * mean(-log_sigmoid(log_odds_chosen - log_odds_rejected))
    log_odds_chosen = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps))
    log_odds_rejected = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps))

    log_odds_diff = log_odds_chosen - log_odds_rejected
    or_penalty = -F.logsigmoid(log_odds_diff).mean()
    return sft_loss + lambda_or * or_penalty

# Step 58 - simpo_loss
import torch
import torch.nn.functional as F

def simpo_loss(policy_chosen_logps, policy_rejected_logps, chosen_lengths, rejected_lengths, beta=2.0, gamma=1.0):
    """Return the mean SimPO loss as a scalar tensor."""
    # TODO: form length-normalized implicit rewards and apply the beta/gamma margin loss
    r_chosen = policy_chosen_logps / chosen_lengths
    r_rejected = policy_rejected_logps / rejected_lengths
    logits = beta * (r_chosen - r_rejected) - gamma
    return -F.logsigmoid(logits).mean()

# Step 59 - build_eval_prompt_set
def build_eval_prompt_set():
    # TODO: return a held-out list of at least 4 short instruction-style eval prompts
    prompts = [
        "Write a short poem about a rainy day in the city.",
        "Explain the difference between a list and a dictionary in Python.",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Give me a step-by-step guide on how to boil a perfect egg.",
        "What are the main causes of the French Revolution?"
    ]
    return prompts

# Step 60 - generate_completions
def generate_completions(model, tokenizer, prompts, max_new_tokens=16):
    """Return a list of greedy completions, one per prompt."""
    completions = []
    for prompt in prompts:
        decoded_output = generate_and_decode(
            model,
            tokenizer,
            prompt,
            max_new_tokens = max_new_tokens
        )
        completions.append(decoded_output)
    return completions

# Step 61 - score_with_reward
def score_with_reward(reward_model, tokenizer, prompt, completion):
    """Return a scalar reward float for the prompt+completion pair."""
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = reward_model['model'](
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True
        )
        # outputs.hidden_states[-1] gets the final layer's hidden states (shape: B, T, D)
        last_hidden_state = outputs.hidden_states[-1][0, -1]
    
    score = reward_head_forward(
        last_hidden_state,
        reward_model['weight'],
        reward_model['bias']
    )
    
    return float(score.item() if hasattr(score, 'item') else score)

# Step 62 - win_rate
def win_rate(reward_model, tokenizer, prompts, completions_a, completions_b):
    """Fraction of prompts where A's completion outscores B's under the reward model.

    Ties count as 0.5. Returns a float in [0, 1].
    """
    wins = 0.0
    for i in range(len(prompts)):
        ra = score_with_reward(reward_model, tokenizer, prompts[i], completions_a[i])
        rb = score_with_reward(reward_model, tokenizer, prompts[i], completions_b[i])
        if ra > rb:
            wins += 1.0
        elif ra == rb:
            wins += 0.5
    
    return float(wins / len(prompts))

# Step 63 - stream_tokens
def stream_tokens(model, tokenizer, prompt, max_new_tokens):
    # TODO: yield one decoded text piece per greedy-decoded new token, up to max_new_tokens.
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    current_text = tokenizer.decode(input_ids[0])
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
        next_token_logits = outputs.logits[0,-1,:]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        new_text = tokenizer.decode(input_ids[0])
        new_piece = new_text[len(current_text):]
        yield new_piece
        current_text = new_text

# Step 64 - apply_stop_tokens
def apply_stop_tokens(text, stop_tokens, eos_token):
    # TODO: truncate text at the earliest occurrence of any stop token or the eos token
    if eos_token is not None:
        stop_tokens.append(eos_token)
    idxf = None
    for token in stop_tokens:
        idx = text.find(token)
        if idx != -1:
            if idxf is None or idx < idxf:
                idxf = idx
    if idxf is not None:
        text = text[:idxf]
    return text

# Step 65 - chat
def chat(model, tokenizer, user_message, system_prompt=None, max_new_tokens=32, stop_tokens=None):
    # TODO: build a chat-style prompt, generate a reply, and trim it at stop tokens / EOS.
    if max_new_tokens <= 0:
        return ""
    prompt = f"User: {user_message}\nAssistant:"
    if system_prompt:
        prompt = f"System: {system_prompt}\n"+prompt
    
    set_pad_token_to_eos(tokenizer)
    raw_reply = generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if stop_tokens is None:
        stop_tokens = ["\nUser:", "\nSystem:"]
    trimmed_reply = apply_stop_tokens(raw_reply, stop_tokens, tokenizer.eos_token)
    return trimmed_reply

