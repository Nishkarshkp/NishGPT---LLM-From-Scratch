Model Name: NishGPT

NishGPT is a small GPT-based language model trained on the TinyStories
dataset. It generates short stories from text prompts.

Final submission split from `LLM_nishkarsh_vfinal1.ipynb`.

Files:
- model.py — model classes and configuration.
- train.py — TinyStories loading, dataset, loss/evaluation, and training.
- chat.py — inference with temperature scaling and top-k sampling.
- generated_output.txt — final generated output recorded in the notebook.
- LLM_nishkarsh_vfinal1.ipynb — complete final notebook.

Model configuration:
- vocab_size = 50257
- context_length = 256
- emb_dim = 256
- n_heads = 4
- n_layers = 4
- drop_rate = 0.1
- qkv_bias = False

Training:
- TinyStories: first 500 training stories
- 90/10 train-validation split
- 10 epochs
- Colab T4 GPU

Round 3 sampling:
- temperature = 0.8
- top_k = 40

Inference:
Put model.py, chat.py, and StoryGPT.pth in one folder and run:
    python chat.py

Important:
The trained `StoryGPT.pth` file was created in the user's Colab runtime and was not attached to this conversation, so it is NOT inside this ZIP. Download `StoryGPT.pth` from Colab and add it to the StoryGPT folder before submitting.
