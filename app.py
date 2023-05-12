import gradio as gr
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

llama_embeddings_model = 'models/ggml-model-q4_0.bin'
persist_directory = 'db'

model_path = 'models/ggml-gpt4all-j-v1.3-groovy.bin'
model_n_ctx = 1000

from constants import CHROMA_SETTINGS

start_message = """<|SYSTEM|># hpcGPT
- Solve all the stupid HPC Tests!"""

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

def chat(curr_system_message, history):
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    res = qa(curr_system_message)
    answer, docs = res['result'], res['source_documents']
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    return answer   
    

with gr.Blocks() as demo:
    gr.Markdown("## hpcChat")
    gr.Markdown("### A chatbot for HPC Test")
    
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                             show_label=False).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[
               submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, [chatbot], queue=False)