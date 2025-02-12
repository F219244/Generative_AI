!pip install gradio openai-whisper transformers fpdf torch

import gradio as gr
import whisper
from transformers import pipeline
from fpdf import FPDF
import os

# Load Whisper ASR Model
model = whisper.load_model("base")
# Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def transcribe_audio(audio_file):
    """Convert speech to text using Whisper"""

    if not os.path.exists(audio_file):
        return "Error: Audio file not found!"

    try:
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Transcription failed: {str(e)}"

def summarize_text(text):
    """Summarizes long transcriptions into a short summary"""

    if len(text) < 50:
        return "Text is too short for summarization!"

    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Summarization failed: {str(e)}"

def save_as_pdf(transcription):
    """Save the transcribed text as a PDF file"""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, transcription)

    pdf_path = "/content/transcription.pdf"
    pdf.output(pdf_path)
    return pdf_path

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎤 Speech-to-Text & Summarization App")
    gr.Markdown("Convert audio into text and get a summary instantly!")

    # Upload Audio Section
    audio_input = gr.File(label="🎵 Upload Your Audio File (MP3, WAV, M4A)")

    # Buttons for Actions
    transcribe_btn = gr.Button("📝 Transcribe")
    summarize_btn = gr.Button("📑 Summarize Text")
    download_btn = gr.Button("📥 Download as PDF")

    # Output Sections
    transcription_output = gr.Textbox(label="Transcription", placeholder="Your transcription will appear here...", lines=10)
    summary_output = gr.Textbox(label="Summary", placeholder="Summary of your text...", lines=5)
    pdf_output = gr.File(label="Download PDF")

    # Button Actions
    transcribe_btn.click(transcribe_audio, inputs=audio_input, outputs=transcription_output)
    summarize_btn.click(summarize_text, inputs=transcription_output, outputs=summary_output)
    download_btn.click(save_as_pdf, inputs=transcription_output, outputs=pdf_output)

# Launch the app
app.launch()
