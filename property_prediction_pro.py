import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast, AdamW, get_linear_schedule_with_warmup
from src.dataset import ecloud_dataset
from src.model import T5WithContrastive
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    # random.seed(seed)

def compute_metrics(y_true, y_pred_logits):
    # y_pred_logits: shape (N, 2)
    probs = torch.softmax(torch.tensor(y_pred_logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {}
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, probs)
    except Exception:
        metrics['roc_auc'] = float('nan')
    try:
        metrics['pr_auc'] = average_precision_score(y_true, probs)
    except Exception:
        metrics['pr_auc'] = float('nan')
    metrics['f1'] = f1_score(y_true, preds)
    metrics['acc'] = accuracy_score(y_true, preds)
    return metrics

seeds = [0, 10, 42, 66, 88, 100, 666, 2025, 2026, 666666]
for seed in seeds: 
    print(f'now seed: {seed}')
    setup_seed(seed) 
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    pretrain_base = "ckpt/"
    tokenizer = T5TokenizerFast.from_pretrained("tokenizer")
    model = T5WithContrastive("laituan245/molt5-small", tokenizer)
    state_dict = torch.load(pretrain_base+"db4/best_model.pt", map_location="cuda") 
    model.load_state_dict(state_dict, strict=False) 
    model.to(device)

    all_data = ecloud_dataset("data/protac-fine4.h5", rotate=True, addp = 'data/fine4_protein_emb.npy')
    trainval_data, test_data = torch.utils.data.random_split(all_data, [len(all_data)-400, 400])
    train_data, val_data = torch.utils.data.random_split(trainval_data, [len(all_data)-800, 400])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    eval_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # optimizer with layerwise lr: smaller for encoder, larger for head
    no_decay = ["bias", "LayerNorm.weight"]
    encoder_params = []
    head_params = []
    for n, p in model.named_parameters():
        if n.startswith("encoder."):
            encoder_params.append((n, p))
        else:
            head_params.append((n, p))

    optimizer_grouped_parameters = [
        {"params": [p for n, p in encoder_params if p.requires_grad], "lr": 1e-5, "weight_decay": 0.01}, 
        {"params": [p for n, p in head_params if p.requires_grad], "lr": 1e-4, "weight_decay": 0.01}, 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    total_steps = len(train_loader) * 20
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    loss_cla = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss() 
    output_dir = f"ckpt/base4_pro_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    best_score = -100
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_num, batch in enumerate(tqdm(train_loader)):
            eclouds = torch.Tensor(batch["eclouds"]).to(torch.float).to(device)
            eclouds += (eclouds>0.2).int() * (torch.rand_like(eclouds) * 0.4 - 0.2).to(device)
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            classes = torch.tensor(batch["classes"]).squeeze(-1).long().to(device)
            p_tgt = pad_sequence(torch.Tensor(batch["p_tgt"]), batch_first=True)
            p_tgt = p_tgt.to(torch.float).to(device)
            p_e3 = pad_sequence(torch.Tensor(batch["p_e3"]), batch_first=True)
            p_e3 = p_e3.to(torch.float).to(device)

            out_cla = model.predict_degradation( 
                eclouds, input_ids, p_tgt, p_e3
            )
            loss = loss_cla(out_cla, classes)
            loss.backward()
            running_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        model.eval()
        all_labels = []
        all_logits = []
        ypred_list = []
        ytest_list = []
        with torch.no_grad():
            for batch in eval_loader:
                eclouds = torch.Tensor(batch["eclouds"]).to(torch.float).to(device)
                input_ids = torch.tensor(batch["input_ids"]).to(device)
                classes = torch.tensor(batch["classes"]).squeeze(-1).long()
                p_tgt = pad_sequence(torch.Tensor(batch["p_tgt"]), batch_first=True)
                p_tgt = p_tgt.to(torch.float).to(device)
                p_e3 = pad_sequence(torch.Tensor(batch["p_e3"]), batch_first=True)
                p_e3 = p_e3.to(torch.float).to(device)
                out = model.predict_degradation(
                    eclouds, input_ids, p_tgt, p_e3
                )
                all_labels.extend(classes.numpy().tolist())
                all_logits.extend(out_cla.cpu().numpy().tolist())
            val_metrics = compute_metrics(all_labels, np.array(all_logits))
            val_auc = val_metrics.get('roc_auc', 0.0)

            print(f"Epoch {epoch} train_loss={running_loss / len(train_loader) :.4f} \
                val_roc_auc={val_auc:.4f} val_pr_auc={val_metrics.get('pr_auc'):.4f} \
                val_f1={val_metrics.get('f1'):.4f} val_acc={val_metrics.get('acc'):.4f}")
                
            now_score = val_auc
            # early stopping on val ROC-AUC
            if now_score > best_score + 1e-4:
                best_score = now_score
                best_epoch = epoch   
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
                print(f"Saved best model at epoch {epoch+1}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 5: 
                print(f"No improvement for 5 epochs. Early stopping.")
                break

    print(f"Training finished. Best val score: {best_score:.4f} at epoch {best_epoch}")
    state_dict = torch.load(os.path.join(output_dir, "best_model.pt"), map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    all_labels = []
    all_logits = []
    ypred_list = []
    ytest_list = []
    with torch.no_grad():
        for batch in test_loader:
            eclouds = torch.Tensor(batch["eclouds"]).to(torch.float).to(device)
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            classes = torch.tensor(batch["classes"]).squeeze(-1).long()
            p_tgt = pad_sequence(torch.Tensor(batch["p_tgt"]), batch_first=True)
            p_tgt = p_tgt.to(torch.float).to(device)
            p_e3 = pad_sequence(torch.Tensor(batch["p_e3"]), batch_first=True)
            p_e3 = p_e3.to(torch.float).to(device)
            out = model.predict_degradation(
                eclouds, input_ids, p_tgt, p_e3
            )
            all_labels.extend(classes.numpy().tolist())
            all_logits.extend(out_cla.cpu().numpy().tolist())

        val_metrics = compute_metrics(all_labels, np.array(all_logits))
        val_auc = val_metrics.get('roc_auc', 0.0)
        print(f"Final test performance:  \
                test_roc_auc={val_auc:.4f} test_pr_auc={val_metrics.get('pr_auc'):.4f} \
                test_f1={val_metrics.get('f1'):.4f} test_acc={val_metrics.get('acc'):.4f}")
