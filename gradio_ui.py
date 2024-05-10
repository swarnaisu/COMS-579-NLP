import gradio as gr
from upload import create_index_and_upload_pdf
from query import get_index_and_get_answer

def answer_question(question, history):
    answer = get_index_and_get_answer(question)
    new_history = ""
    if len(history) != 0:
        new_history += f"{history}\n\n\n\n"
    new_history += f"You:\n{question}\n\nRAG:\n{answer}"

    return new_history

def index_pdf(file_path):
    create_index_and_upload_pdf(file_path.name)

    return "File upload done"

css = """
.textbox-container {
    max-height: 300px;
    overflow-y: auto;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Glass()) as demo:
    with gr.Row():
        with gr.Column():
            history_output = gr.Textbox(label="Results", value="", lines=10, interactive=False)
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="Upload PDF", height=30, min_width=60)
                with gr.Column():
                    pdf_output = gr.Text(label="Status")
            pdf_button = gr.Button("Upload PDF")
            question_input = gr.Textbox(label="Enter your question")
            question_button = gr.Button("Query")
    question_button.click(
        fn=answer_question,
        inputs=[question_input, history_output],
        outputs=[history_output]
    )
    pdf_button.click(index_pdf, inputs=pdf_input, outputs=pdf_output)

demo.launch()
