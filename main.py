import customtkinter as ctk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Chat App")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/WizardLM-7B-Uncensored")
        self.model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/WizardLM-7B-Uncensored")
        
        # Check if GPU is available and move model to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            print("Error: CUDA-enabled GPU not found. Exiting application.")
            sys.exit(1)
        
        # Set up the GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.text_area = ctk.CTkTextbox(self.root, width=500, height=400, wrap='word')
        self.text_area.pack(pady=10)
        
        self.entry = ctk.CTkEntry(self.root, width=400)
        self.entry.pack(pady=10)
        
        self.send_button = ctk.CTkButton(self.root, text="Send", command=self.send_message)
        self.send_button.pack(pady=10)
        
    def send_message(self):
        user_input = self.entry.get()
        self.display_message(f"User: {user_input}\n", "user")
        
        # Generate response from the model
        response = self.generate_response(user_input)
        self.display_message(f"Bot: {response}\n\n", "bot")
        
        # Clear the entry box
        self.entry.delete(0, ctk.END)
        
    def display_message(self, message, sender):
        if sender == "user":
            self.text_area.insert(ctk.END, message, "user")
        else:
            self.text_area.insert(ctk.END, message, "bot")
        
        # Apply tags for styling
        self.text_area.tag_configure("user", foreground="blue", font=("Helvetica", 10, "bold"))
        self.text_area.tag_configure("bot", foreground="green", font=("Helvetica", 10, "italic"))
        
    def generate_response(self, user_input):
        # Tokenize input and generate response
        inputs = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt').to(self.device)
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(self.device)  # Create an attention mask

        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,  # Pass the attention mask
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,  # Controls randomness: lower is less random
            top_k=50,         # Limits the sampling pool to top_k tokens
            top_p=0.50,       # Nucleus sampling: considers tokens with cumulative probability up to top_p
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True    # Enable sampling to use temperature and top_p
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA-enabled GPU not found. Exiting application.")
        sys.exit(1)
    
    root = ctk.CTk()
    app = ChatApp(root)
    root.mainloop()
