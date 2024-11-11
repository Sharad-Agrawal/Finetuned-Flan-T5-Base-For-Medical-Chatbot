import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load your fine-tuned T5 model and tokenizer
model_name = "./best_model"
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(model_name)
original_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Define a function to generate answers
def generate_answer(question):
    # Tokenize the input question
    inputs = tokenizer.encode("answer the medical question: " + question, return_tensors="pt", truncation=True)
    
    # Generate the answer with the fine-tuned model
    fine_tuned_outputs = fine_tuned_model.generate(inputs, max_length=128, no_repeat_ngram_size=3, top_k=50, top_p=0.95)
    original_outputs = original_model.generate(inputs, max_length=128)
    
    # Decode the generated answers
    fine_tuned_answer = tokenizer.decode(fine_tuned_outputs[0], skip_special_tokens=True)
    original_answer = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    
    return [fine_tuned_answer, original_answer]

# Set up the Gradio interface with two outputs
iface = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=2, placeholder="Enter your medical question here...", label="Question"),
    outputs=[
        gr.Textbox(label="Fine-Tuned Model Answer"),
        gr.Textbox(label="Original Model Answer")
    ],
    title="Medical Question Answering Bot",
    description="Ask a medical question and get answers from both the original and fine-tuned models."
)

# Launch the Gradio app with share=True to get a public link
iface.launch(share=True)
