import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from model import GPTModel, GPT_CONFIG_124M


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(
        txt,
        tokenizer,
        max_length,
        stride
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device,
        )
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter,
):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter,
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=eval_iter,
        )

    model.train()

    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch+1} "
                    f"(Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
        )

    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]

    encoded = text_to_token_ids(
        start_context,
        tokenizer,
    ).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=0.8,
            top_k=40,
        )

    decoded_text = token_ids_to_text(
        token_ids,
        tokenizer,
    )

    print(decoded_text.replace("\n", " "))

    model.train()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}
    )
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=None,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]

            logits = torch.where(
                logits < min_val.unsqueeze(-1),
                torch.tensor(
                    float("-inf"),
                    device=logits.device
                ),
                logits,
            )

        if temperature > 0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

        else:
            idx_next = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def load_tinystories(num_stories=500):
    dataset = load_dataset("roneneldan/TinyStories")
    stories = dataset["train"]["text"][:num_stories]
    return "\n\n".join(stories)


if __name__ == "__main__":
    torch.manual_seed(123)

    tiny_stories_text = load_tinystories(num_stories=500)

    text_data = tiny_stories_text
    tokenizer = tiktoken.get_encoding("gpt2")

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1,
    )

    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=50,
        eval_iter=5,
        start_context="Once upon a time",
        tokenizer=tokenizer,
    )

    torch.save(model.state_dict(), "StoryGPT.pth")
    print("Model saved successfully!")
