import torch
from model import ShakePT
from train import encode, decode
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    try:
        from train import vocab_size, dict_ch, dict_idx
        model = ShakePT(vocab_size).to(device)
        model.load_state_dict(torch.load('results/ShakePT.pth'))
        
        input = input("Enter the beginning of the setence : ")
        
        context = torch.tensor([encode(input, dict_ch)], dtype=torch.long, device=device)
        gen = model.generate(context, max_tokens=1000)
        res = decode(gen[0].tolist(), dict_idx)
        with open('results/output.txt', 'w+', encoding='utf-8') as file:
            file.write(res)
        print("Output register in results/output.txt")
        
    except:
        print("No model pre-trained, please train ShakePT before testing with the command 'python3 train.py'.")


