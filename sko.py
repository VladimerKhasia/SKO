# SKO
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

def set_seed(seed=42):
    """Locks all random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Setup] Random seed set to {seed}")

# =============================================================================
# 1. MATHEMATICAL CORE: O'Dowd's Spherical Kernel 
# =============================================================================


class OSKAPolynomialKernelExplicit(nn.Module):
    def __init__(self, n_heads, head_degrees, manifold_dim_q=64):
        super().__init__()
        self.n_heads = n_heads
        self.q = manifold_dim_q
        self.lam = (self.q - 1) / 2.0
        
        # Ensure the user provided exactly one degree per head
        assert len(head_degrees) == n_heads, f"Provided {len(head_degrees)} degrees for {n_heads} heads!"
        
        # The maximum degree dictates how many loop iterations we must run
        self.max_deg = math.ceil(max(head_degrees))
        
        # Polynomial weights per head (These are still learnable!)
        self.poly_weights = nn.Parameter(torch.ones(n_heads, self.max_deg + 1))
        
        # --- EXPLICIT DEGREES ---
        # Shape: (1, n_heads, 1, 1) so it broadcasts across Batch and Sequence dims
        degrees_tensor = torch.tensor(head_degrees, dtype=torch.float32).view(1, n_heads, 1, 1)
        self.register_buffer("head_degrees", degrees_tensor)

    def forward(self, x):
        # x shape: (Batch, Heads, Seq_q, Seq_k)
        B, H, S_q, S_k = x.shape
        
        # Read the fixed degrees
        d = self.head_degrees

        R = [torch.ones_like(x), x]
        
        # k = 0
        w0 = self.poly_weights[:, 0].view(1, H, 1, 1)
        gate0 = torch.clamp(d - 0 + 1, min=0.0, max=1.0)
        Phi_n_q = (w0 * gate0) * R[0]
        
        # k = 1
        w1 = self.poly_weights[:, 1].view(1, H, 1, 1)
        gate1 = torch.clamp(d - 1 + 1, min=0.0, max=1.0)
        Phi_n_q = Phi_n_q + (w1 * gate1) * R[1]

        # Calculate for higher degrees up to max_deg
        for k in range(2, self.max_deg + 1):
            term1_coeff = 2 * (k + self.lam - 1) / (k + 2 * self.lam - 1)
            term2_coeff = (k - 1) / (k + 2 * self.lam - 1)

            R_k = term1_coeff * x * R[k-1] - term2_coeff * R[k-2]
            R.append(R_k)

            w_k = self.poly_weights[:, k].view(1, H, 1, 1)
            gate_k = torch.clamp(d - k + 1, min=0.0, max=1.0)
            
            Phi_n_q = Phi_n_q + (w_k * gate_k) * R_k

        return Phi_n_q

# =============================================================================
# 2. ATTENTION REPLACEMENT: OSKA
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class OSKAKERNEL(nn.Module):
    def __init__(self, d_model, n_heads, head_degrees, q_dim, use_post_rms_norm=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_post_rms_norm = use_post_rms_norm

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # USING THE EXPLICIT KERNEL
        self.kernel = OSKAPolynomialKernelExplicit(
            n_heads=n_heads, head_degrees=head_degrees, manifold_dim_q=q_dim
        )

        if self.use_post_rms_norm:
            self.post_norm = RMSNorm(self.d_head)

    def forward(self, x):
        B, seq_len, _ = x.shape

        q = self.q_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        sim = torch.matmul(q, k.transpose(-2, -1))

        weights = self.kernel(sim)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        weights = weights.masked_fill(mask == 0, 0.0)

        M_divisor = torch.arange(1, seq_len + 1, device=x.device).view(1, 1, seq_len, 1)
        out = torch.matmul(weights, v) / M_divisor

        if self.use_post_rms_norm:
            out = self.post_norm(out)

        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.out_proj(out)

# =============================================================================
# 3. LLM ARCHITECTURE
# =============================================================================

class OSKABlock(nn.Module):
    def __init__(self, d_model, n_heads, head_degrees, q_dim, use_post_rms_norm, attention_type="oska"):
        super().__init__()
        if attention_type == "oska":
            self.attn = OSKAKERNEL(d_model, n_heads, head_degrees, q_dim, use_post_rms_norm)
        elif attention_type == "baseline":
            self.attn = StandardAttention(d_model, n_heads)
        else:
            raise ValueError("attention_type must be 'oska' or 'baseline'")
            
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp_up = nn.Linear(d_model, d_model * 4, bias=False)
        self.mlp_gate = nn.Linear(d_model, d_model * 4, bias=False)
        self.mlp_down = nn.Linear(d_model * 4, d_model, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        norm_x = self.norm2(x)
        mlp_out = F.silu(self.mlp_gate(norm_x)) * self.mlp_up(norm_x)
        x = x + self.mlp_down(mlp_out)
        return x

class OSKALanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, head_degrees=None, q_dim=64, use_post_rms_norm=True, attention_type="oska"):
        super().__init__()
        
        # Fallback just in case user forgets to provide the list
        if head_degrees is None:
            head_degrees = [5.0] * n_heads
            
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)

        self.layers = nn.ModuleList([
            OSKABlock(d_model, n_heads, head_degrees, q_dim, use_post_rms_norm, attention_type)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_emb.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        B, seq_len = input_ids.shape
        pos = torch.arange(0, seq_len, device=input_ids.device)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# =============================================================================
# 4. Baseline
# =============================================================================

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, seq_len, _ = x.shape

        q = self.q_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Standard scaled dot-product
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Standard causal mask (using -inf for Softmax)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        sim = sim.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(sim, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.out_proj(out)

# =============================================================================
# 5. EXPERIMENT & HARNESS UTILITIES 
# =============================================================================

def create_data_generators(dataset_name, dataset_subset, streaming, train_samples, val_samples):
    """Creates isolated training and validation data generators, handling streaming properly."""
    if streaming:
        ds = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)
        # In streaming, we take the first N elements for strictly fixed validation, and skip them for training
        val_ds = ds.take(val_samples)
        train_ds = ds.skip(val_samples)

        def cyclic_gen(dataset):
            while True:
                for example in dataset:
                    yield example

        return iter(cyclic_gen(train_ds)), iter(cyclic_gen(val_ds))
    else:
        # If not streaming, we slice and split using standard HF tools
        ds = load_dataset(dataset_name, name=dataset_subset, split=f"train[:{train_samples + val_samples}]")
        split = ds.train_test_split(test_size=val_samples, seed=42)

        def cyclic_gen(dataset):
            while True:
                for example in dataset:
                    yield example

        return iter(cyclic_gen(split['train'])), iter(cyclic_gen(split['test']))

def get_batch(data_iterator, tokenizer, seq_len, batch_size, device):
    """Fetches and tokenizes a batch from the generic text data iterator."""
    texts = [next(data_iterator)['text'] for _ in range(batch_size)]
    encodings = tokenizer(texts, truncation=True, padding="max_length",
                          max_length=seq_len+1, return_tensors="pt")

    x = encodings['input_ids'][:, :-1].to(device)
    y = encodings['input_ids'][:, 1:].to(device)
    return x, y

@torch.no_grad()
def evaluate(model, val_iterator, tokenizer, seq_len, batch_size, eval_steps, device):
    """Computes the validation loss rigorously over strictly separated validation data."""
    model.eval()
    total_loss = 0.0
    for _ in range(eval_steps):
        x, y = get_batch(val_iterator, tokenizer, seq_len, batch_size, device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
    model.train()
    return total_loss / eval_steps

def plot_training_results(train_losses, val_losses, save_path="oska_experiment_loss.png"):
    """Generates a matplotlib graph of the training vs validation loss curves."""
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))

    # Unpack tuples (step, loss)
    t_steps, t_loss = zip(*train_losses)
    v_steps, v_loss = zip(*val_losses)

    plt.plot(t_steps, t_loss, label='Training Loss', alpha=0.6, color='#1f77b4', linewidth=1.5)
    plt.plot(v_steps, v_loss, label='Validation Loss', marker='o', color='#d62728', linewidth=2.5)

    plt.title("OSKA Model: Training vs Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visuals] Saved loss plot to {save_path}")
    plt.close()

@torch.no_grad()
def demo_inference(model, tokenizer, prompt, max_new_tokens, seq_len, device):
    """Generates text autoregressively using the provided trained OSKA model."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"\n--- OSKA Generative Inference Demo ---")
    print(f"Prompt: '{prompt}'")

    for _ in range(max_new_tokens):
        # Truncate to context window seq_len
        idx_cond = input_ids[:, -seq_len:]

        logits, _ = model(idx_cond)
        # Pluck the logits at the final step
        next_token_logits = logits[:, -1, :]

        # Greedy decoding (Argmax) for deterministic demo stability
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    print(f"Result: {generated_text}")
    print(f"--------------------------------------\n")

# =============================================================================
# 6. EXPERIMENT PIPELINE WITH USER-CONTROLLED VARIABLES
# =============================================================================
def run_experiment(
    # --- Data Parameters ---
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_subset="sample-10BT",
    streaming=True,
    train_samples=9_500_000,   # Used strictly as partition size in streaming
    val_samples=5000,      # Strict validation separation size

    # --- Model Hyperparameters ---
    q_dim=64,              # Intrinsic manifold dimension assumption
    use_rms=True,          # User-controlled normalization mitigation
    d_model=256,           # Scaled down for rapid testing (Standard=768)
    n_layers=4,
    n_heads=4,
    head_degrees=[2.0, 3.0, 4.0, 5.0],  #[5.0, 5.0, 5.0, 5.0],
    seq_len=256,

    # --- Training Loop Setup ---
    batch_size=32, #8,
    max_steps=5000,
    val_interval=500,
    eval_steps=32,         # Batches to compute over per validation phase
    lr=6e-4,
    weight_decay=0.1,
    checkpoint_dir="./oska_checkpoints",

    seed=42,              
    attention_type="oska", 
    use_scheduler=True,    
    min_lr=1e-5,           
):

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Running on Device: {device}")

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"best_{attention_type}_model.pt")

    # 1. Load Tokenizer & Datasets (Strict Separation)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Creating isolated data streams...")
    train_iter, val_iter = create_data_generators(
        dataset_name, dataset_subset, streaming, train_samples, val_samples
    )

    # 2. Setup Models
    model = OSKALanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_degrees=head_degrees, 
        q_dim=q_dim,
        use_post_rms_norm=use_rms,
        attention_type=attention_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=min_lr
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 3. Macro-Training & Validation Loop
    model.train()

    train_history = []
    val_history = []
    best_val_loss = float('inf')

    progress_bar = tqdm(total=max_steps, desc=f"Training {'OSKA' if attention_type=='oska' else 'Baseline'}")

    for step in range(1, max_steps + 1):
        x, y = get_batch(train_iter, tokenizer, seq_len, batch_size, device)

        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if use_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr

        # # Track training loss
        train_history.append((step, loss.item()))
        progress_bar.update(1)

        # Update your progress bar to show LR
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "LR": f"{current_lr:.2e}"
        })

        # --- Validation & Checkpointing Phase ---
        if step % val_interval == 0 or step == max_steps:
            val_loss = evaluate(model, val_iter, tokenizer, seq_len, batch_size, eval_steps, device)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf') # Cap exp for printing
            val_history.append((step, val_loss))

            tqdm.write(f"\nStep {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

            # Save Checkpoint if Validation Improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt_path)
                tqdm.write(f" ╰─> [Checkpoint Saved] New best validation loss achieved!")

    progress_bar.close()
    print("\nTraining complete!")

    # 4. Generate Graphical Report
    plot_training_results(train_history, val_history, save_path=f"{attention_type}_validation_report.png")

    # 5. Load Best Checkpoint & Demo Inference
    if attention_type=="oska":
        print("\nLoading best checkpoint for final generative inference demo...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        
        demo_inference(
            model=model,
            tokenizer=tokenizer,
            prompt="The most important scientific discovery",
            max_new_tokens=40,
            seq_len=seq_len,
            device=device
        )

    return val_history

def plot_model_comparison(oska_val_losses, baseline_val_losses, save_path="attention_comparison.png"):
    """Generates a dual-line plot comparing OSKA and Baseline validation loss."""
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))

    # Unpack OSKA steps and losses
    if oska_val_losses:
        o_steps, o_loss = zip(*oska_val_losses)
        plt.plot(o_steps, o_loss, label='OSKA Validation Loss', marker='o', color='#1f77b4', linewidth=2.5)

    # Unpack Baseline steps and losses
    if baseline_val_losses:
        b_steps, b_loss = zip(*baseline_val_losses)
        plt.plot(b_steps, b_loss, label='Baseline Validation Loss', marker='s', color='#ff7f0e', linewidth=2.5, linestyle='--')

    plt.title("Validation Loss Comparison: OSKA vs Standard Attention", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Cross-Entropy Validation Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visuals] Saved comparison plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    # # You can easily override hyperparameters here:
    # run_experiment(
    #     dataset_name="HuggingFaceFW/fineweb-edu",
    #     dataset_subset="sample-10BT",
    #     streaming=True,             # Set to False to download locally before slicing
    #     batch_size=8,               # Increase based on GPU RAM
    #     max_steps=500,              # Adjust length of experiment
    #     val_interval=50,            # How frequently to validate & checkpoint
    #     n_degree=5,                 # Degree of OSKA kernel
    #     q_dim=64,                   # Manifold dim of OSKA kernel
    #     use_rms=True                # Feature flag mitigation
    # )

    print("=== RUNNING BASELINE (STANDARD ATTENTION) ===")
    baseline_val_history = run_experiment(
        attention_type="baseline",
        seed=42,    
        use_scheduler=True
        # ... specify your other args here ...
    )

    print("\n=== RUNNING OSKA ATTENTION ===")
    oska_val_history = run_experiment(
        attention_type="oska",
        seed=42,
        use_scheduler=True,
        n_heads=4, 
        head_degrees=[2.0, 3.0, 4.0, 5.0],               # [5.0, 5.0, 5.0, 5.0],           
        # ... specify your other args here ...
    )

    # Generate the comparison plot!
    plot_model_comparison(
        oska_val_losses=oska_val_history,
        baseline_val_losses=baseline_val_history,
        save_path="oska_vs_baseline_validation.png"
    )
