import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('research_paper_summarizer.log')
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class ResearchPaperSummarizer:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        device = 0 if torch.cuda.is_available() else -1 
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
        self.max_chunk_length = 2048
        logger.info("ResearchPaperSummarizer initialized")

    def clean_text(self, text):
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_into_chunks(self, text, max_chunk_length=2048):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chunk_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def summarize_section(self, text, max_length=200, min_length=60):
        if not text or len(text.strip()) < 50:
            return ""
        
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) <= self.max_chunk_length:
            summary = self.summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text'] if summary else ""
        
        chunks = self.split_into_chunks(cleaned_text)
        summaries = []
        
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                if summary and summary[0]['summary_text']:
                    summaries.append(summary[0]['summary_text'])
            except Exception as e:
                logger.warning(f"Error summarizing chunk: {str(e)}")
                continue
        
        return " ".join(summaries[:5]) 
    
    def extract_sections(self, text):
        sections = defaultdict(str)
        
        section_patterns = {
            'abstract': r'(?:abstract|summary).*?\n(.*?)(?=introduction|\n\n|$)',
            'introduction': r'(?:introduction|background).*?\n(.*?)(?=methodology|methods|materials and methods|\n\n|$)',
            'methodology': r'(?:methodology|methods|materials and methods|experimental setup).*?\n(.*?)(?=results|\n\n|$)',
            'results': r'(?:results|findings).*?\n(.*?)(?=discussion|\n\n|$)',
            'discussion': r'discussion.*?\n(.*?)(?=conclusion|\n\n|$)',
            'conclusion': r'(?:conclusion|conclusions).*?\n(.*?)(?=references|bibliography|\n\n|$)'
        }
        
        for section, pattern in section_patterns.items():
            matches = re.search(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            if matches:
                sections[section] = matches.group(1).strip()
        
        return sections
    
    def extract_figures(self, text):
        captions = re.findall(r'(?:Figure\s?\d+[:\.\-\s]?.*?)(?=\n|\Z)', text, re.IGNORECASE)
        return [caption.strip() for caption in captions]

    def visualize_word_count(self, sections):
        word_counts = {section: len(text.split()) for section, text in sections.items()}
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(word_counts.keys()), y=list(word_counts.values()), ax=ax)
        ax.set_title('Section-Wise Word Counts')
        ax.set_xlabel('Section')
        ax.set_ylabel('Word Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    def visualize_sentence_lengths(self, text):
        sentences = sent_tokenize(text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(sentence_lengths, bins=20, kde=True, ax=ax)
        ax.set_title('Sentence Length Distribution')
        ax.set_xlabel('Sentence Length (in words)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    def calculate_rouge(self, generated_summary, reference_summary):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        return scores


def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return None
    return pdf_text

def extract_title(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if "Title" in pdf_reader.metadata:
            return pdf_reader.metadata["Title"]
        
        if pdf_reader.pages:
            first_page_text = pdf_reader.pages[0].extract_text()
            title_match = re.search(
                r'^[^\n]{10,100}(?=\n.*abstract|introduction)',
                first_page_text,
                re.IGNORECASE | re.DOTALL
            )
            if title_match:
                return title_match.group(0).strip()
        
    except Exception as e:
        logger.warning(f"Unable to extract title: {str(e)}")
    
    return "Title not found"

def main():
    st.set_page_config(page_title="Research Paper Summarizer", page_icon="📚")
    st.title("📚 Research Paper Summarizer")
    st.write("Upload a research paper PDF to get a detailed summary and visualizations.")
    
    summarizer = ResearchPaperSummarizer()
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            title = extract_title(uploaded_file)
            paper_text = extract_text_from_pdf(uploaded_file)
            
            if paper_text:
                sections = summarizer.extract_sections(paper_text)
                captions = summarizer.extract_figures(paper_text)
                
                st.header("📋 Research Paper Summary")
                st.subheader("1. Title")
                st.write(title)
                
                st.subheader("2. Abstract Summary")
                abstract_summary = summarizer.summarize_section(sections.get("abstract", ""))
                st.write(abstract_summary)
                
                st.subheader("3. Conclusion Summary")
                conclusion_summary = summarizer.summarize_section(sections.get("conclusion", ""))
                st.write(conclusion_summary)
                
                st.subheader("4. Figures Captions")
                if captions:
                    for caption in captions:
                        st.write(f"- {caption}")
                else:
                    st.warning("No figure captions found.")
                
                st.subheader("5. Overall Summary")
                overall_summary = summarizer.summarize_section(paper_text, max_length=200, min_length=100)
                st.write(overall_summary)

                # Visualizations
                st.subheader("📊 Visualizations")
                summarizer.visualize_word_count(sections)
                summarizer.visualize_sentence_lengths(paper_text)
                
                st.subheader("ROUGE Score")
                rouge_scores = summarizer.calculate_rouge(abstract_summary, sections.get("abstract", ""))
                st.write(f"ROUGE-1: {rouge_scores['rouge1']}")
                st.write(f"ROUGE-2: {rouge_scores['rouge2']}")
                st.write(f"ROUGE-L: {rouge_scores['rougeL']}rch.cuda.is_available() else -1 
        self.summarizer =import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import streamlit as st
import PyPDF2
import io
import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rouge_score import rouge_scorer

class ResearchPaperSummarizer:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        device = 0 if torch.cuda.is_available() else -1 
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
        self.max_chunk_length = 2048

    def clean_text(self, text):
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_into_chunks(self, text, max_chunk_length=2048):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chunk_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def summarize_section(self, text, max_length=200, min_length=60):
        if not text or len(text.strip()) < 50:
            return ""
        
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) <= self.max_chunk_length:
            summary = self.summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text'] if summary else ""
        
        chunks = self.split_into_chunks(cleaned_text)
        summaries = []
        
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                if summary and summary[0]['summary_text']:
                    summaries.append(summary[0]['summary_text'])
            except Exception as e:
                st.warning(f"Error summarizing chunk: {str(e)}")
                continue
        
        return " ".join(summaries[:5]) 
    
    def extract_sections(self, text):
        sections = defaultdict(str)
        
        section_patterns = {
            'abstract': r'(?:abstract|summary).*?\n(.*?)(?=introduction|\n\n|$)',
            'introduction': r'(?:introduction|background).*?\n(.*?)(?=methodology|methods|materials and methods|\n\n|$)',
            'methodology': r'(?:methodology|methods|materials and methods|experimental setup).*?\n(.*?)(?=results|\n\n|$)',
            'results': r'(?:results|findings).*?\n(.*?)(?=discussion|\n\n|$)',
            'discussion': r'discussion.*?\n(.*?)(?=conclusion|\n\n|$)',
            'conclusion': r'(?:conclusion|conclusions).*?\n(.*?)(?=references|bibliography|\n\n|$)'
        }
        
        for section, pattern in section_patterns.items():
            matches = re.search(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            if matches:
                sections[section] = matches.group(1).strip()
        
        return sections
    
    def extract_figures(self, text):
        captions = re.findall(r'(?:Figure\s?\d+[:\.\-\s]?.*?)(?=\n|\Z)', text, re.IGNORECASE)
        return [caption.strip() for caption in captions]

    def visualize_word_count(self, sections):
        word_counts = {section: len(text.split()) for section, text in sections.items()}
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(word_counts.keys()), y=list(word_counts.values()), ax=ax)
        ax.set_title('Section-Wise Word Counts')
        ax.set_xlabel('Section')
        ax.set_ylabel('Word Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    def visualize_sentence_lengths(self, text):
        sentences = sent_tokenize(text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(sentence_lengths, bins=20, kde=True, ax=ax)
        ax.set_title('Sentence Length Distribution')
        ax.set_xlabel('Sentence Length (in words)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    def calculate_rouge(self, generated_summary, reference_summary):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        return scores


def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None
    return pdf_text

def extract_title(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if "Title" in pdf_reader.metadata:
            return pdf_reader.metadata["Title"]
        
        if pdf_reader.pages:
            first_page_text = pdf_reader.pages[0].extract_text()
            title_match = re.search(
                r'^[^\n]{10,100}(?=\n.*abstract|introduction)',
                first_page_text,
                re.IGNORECASE | re.DOTALL
            )
            if title_match:
                return title_match.group(0).strip()
        
    except Exception as e:
        st.warning(f"Unable to extract title: {str(e)}")
    
    return "Title not found"

def main():
    st.set_page_config(page_title="Research Paper Summarizer", page_icon="📚")
    st.title("📚 Research Paper Summarizer")
    st.write("Upload a research paper PDF to get a detailed summary and visualizations.")
    
    summarizer = ResearchPaperSummarizer()
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            title = extract_title(uploaded_file)
            paper_text = extract_text_from_pdf(uploaded_file)
            
            if paper_text:
                sections = summarizer.extract_sections(paper_text)
                captions = summarizer.extract_figures(paper_text)
                
                st.header("📋 Research Paper Summary")
                st.subheader("1. Title")
                st.write(title)
                
                st.subheader("2. Abstract Summary")
                abstract_summary = summarizer.summarize_section(sections.get("abstract", ""))
                st.write(abstract_summary)
                
                st.subheader("3. Conclusion Summary")
                conclusion_summary = summarizer.summarize_section(sections.get("conclusion", ""))
                st.write(conclusion_summary)
                
                st.subheader("4. Figures Captions")
                if captions:
                    for caption in captions:
                        st.write(f"- {caption}")
                else:
                    st.warning("No figure captions found.")
                
                st.subheader("5. Overall Summary")
                overall_summary = summarizer.summarize_section(paper_text, max_length=200, min_length=100)
                st.write(overall_summary)

                # Visualizations
                st.subheader("📊 Visualizations")
                summarizer.visualize_word_count(sections)
                summarizer.visualize_sentence_lengths(paper_text)
                
                st.subheader("ROUGE Score")
                rouge_scores = summarizer.calculate_rouge(abstract_summary, sections.get("abstract", ""))
                st.write(f"ROUGE-1: {rouge_scores['rouge1']}")
                st.write(f"ROUGE-2: {rouge_scores['rouge2']}")
                st.write(f"ROUGE-L: {rouge_scores['rougeL']}")

if __name__ == "__main__":
    main()
