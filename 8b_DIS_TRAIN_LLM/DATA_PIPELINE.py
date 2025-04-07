import torch
import tiktoken

tokenizer = tiktoken.get_encoding("o200k_base")

def tokenize(text: str, device="cpu"):
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([id]) for id in token_ids]
    return {
        'token_ids': token_ids,
        'tokens': tokens
    }

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    passinput = input("Enter your text: ")
    results = tokenize(passinput, device=device)

    print("Tokens:", results['tokens'])
    print("\nToken IDs:", results['token_ids'])
    print("\nExample token (first token):")
    print("Token:", results['tokens'][0])
    print("Token ID:", results['token_ids'][0])