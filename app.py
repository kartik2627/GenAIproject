import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load CodeGen model
model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Language templates
language_templates = {
    "Python": "# Language: Python\n# Task: ",
    "JavaScript": "// Language: JavaScript\n// Task: ",
    "C++": "// Language: C++\n// Task: ",
    "Java": "// Language: Java\n// Task: ",
    "HTML": "<!-- Language: HTML -->\n<!-- Task: ",
    "SQL": "-- Language: SQL\n-- Task: ",
    "Bash": "# Language: Bash\n# Task: "
}

# Code generation function
def generate_code(prompt, language="Python", temperature=0.7, max_tokens=256):
    template = language_templates.get(language, "")
    full_prompt = template + prompt + "\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0]) + max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code[len(full_prompt):].strip()

# Chat function with history
def chat_with_codegen(user_input, language, history):
    if not user_input.strip():
        return history, "Please enter a prompt."

    generated_code = generate_code(user_input, language)
    history.append((f"[{language}] {user_input}", generated_code))
    return history, generated_code

# Gradio UI
with gr.Blocks(title="Multilingual CodeGen Chatbot with History") as demo:
    gr.Markdown("## 🤖 CodeGen Chatbot with Language Support + Prompt History")
    gr.Markdown("Describe what you want the code to do. Select a language. Click Generate!")

    with gr.Row():
        lang_choice = gr.Dropdown(choices=list(language_templates.keys()), value="Python", label="💬 Language")
        user_input = gr.Textbox(label="📝 Your Prompt", placeholder="e.g., Write a function to reverse a string")

    chatbot = gr.Chatbot(label="🧠 Chat History")
    output = gr.Code(label="🧾 Generated Code")

    state = gr.State([])  # Keeps history

    generate_btn = gr.Button("🚀 Generate Code")

    generate_btn.click(
        fn=chat_with_codegen,
        inputs=[user_input, lang_choice, state],
        outputs=[chatbot, output]
    )

demo.launch()

