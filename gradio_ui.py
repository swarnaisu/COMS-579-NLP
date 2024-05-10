import gradio as gr
from upload import create_index_and_upload_pdf
from query import get_index_and_get_answer

def answer_question(question, history):
    answer = get_index_and_get_answer(question)
    new_history = f"Q: {question}\nA: {answer}\n\n" + history
    return answer, new_history
    # return ["This is sample Answer", "This is previous history"]

def index_pdf(file_path):
    create_index_and_upload_pdf(file_path.name)
    return "File indexed successfully."

css = """
.textbox-container {
    max-height: 300px;
    overflow-y: auto;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Enter your question")
            answer_output = gr.Text(label="Answer")
            history_output = gr.Textbox(label="Q&A History", value="", lines=10, interactive=False)
            question_button = gr.Button("Get Answer")
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF")
            pdf_output = gr.Text(label="Indexing Status")
            pdf_button = gr.Button("Index PDF")
    question_button.click(
        fn=answer_question,
        inputs=[question_input, history_output],
        outputs=[answer_output, history_output]
    )
    pdf_button.click(index_pdf, inputs=pdf_input, outputs=pdf_output)

demo.launch()
