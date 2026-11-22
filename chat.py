import torch
import tiktoken

from model import GPTModel, GPT_CONFIG_124M


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

        # Top-k sampling
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

        # Temperature scaling
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


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPT_CONFIG_124M)

    model.load_state_dict(
        torch.load(
            "StoryGPT.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    prompt = input("Enter a prompt: ")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=150,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.8,
        top_k=40,
    )

    generated_text = token_ids_to_text(
        token_ids,
        tokenizer
    )

    print("\nGenerated story:")
    print(generated_text)
