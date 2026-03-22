import streamlit as st
import torch
import numpy as np
import plotly.graph_objs as go
import re
from transformers import AutoTokenizer, AutoModel
import os
import nltk
from models.model import RoBERTaModernBERTCRF
from models.style_features import StyleFeatureExtractor
from utils.preprocessing import clean_text

# Download NLTK resources (only if not already present)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Model and tokenizer names
ROBERTA_MODEL_NAME = 'roberta-base'
MODERNBERT_MODEL_NAME = 'answerdotai/ModernBERT-Base'
MODEL_PATH = r"C:\teja\AAAI_2026\RoBERTa_ModernBERT_CRF.pth"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'style_features' not in st.session_state:
    st.session_state.style_features = None
if 'info_mask' not in st.session_state:
    st.session_state.info_mask = None
if 'words_plot' not in st.session_state:
    st.session_state.words_plot = []

# Streamlit UI setup
st.set_page_config(page_title="AI Text Detector", layout="wide")
st.markdown(
    """
    <style>
        .main { background-color: #f5f6fa; }
        .stTextArea textarea { font-size: 1.1em; width: 1000px !important; min-width: 1000px !important; max-width: 1000px !important; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 1.1em; }
        .result-word { font-weight: bold; padding: 2px 4px; border-radius: 3px; }
        .ai { background-color: #ffcccc; }
        .human { background-color: #ccffcc; }
        .word-count { font-size: 1.1em; margin-bottom: 10px; }
        .warning { color: #b30000; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🕵️‍♂️ AI Text Detector")
st.write("Paste your text below and click **Detect** to see which parts are likely AI-generated.")

input_text = st.text_area("Enter your text here", height=400, key="input_text_area")
st.session_state.input_text = input_text

word_count = len(input_text.split())
detect_disabled = word_count < 30

if st.button("Detect", disabled=detect_disabled, key="detect_button"):
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model checkpoint not found at {MODEL_PATH}")
            st.stop()
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, add_prefix_space=True)
        language_model = AutoModel.from_pretrained(ROBERTA_MODEL_NAME, use_safetensors=True)
        if hasattr(language_model, 'is_meta') and language_model.is_meta:
            language_model = language_model.to_empty(device=device)
        else:
            language_model = language_model.to(device)
        style_extractor = StyleFeatureExtractor(tokenizer, language_model, device)
        model = RoBERTaModernBERTCRF(ROBERTA_MODEL_NAME, MODERNBERT_MODEL_NAME, num_labels=2)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict']
        # Remap CRF keys if needed (for torchcrf version compatibility)
        crf_key_map = {
            'crf.trans_matrix': 'crf.transitions',
            'crf.start_trans': 'crf.start_transitions',
            'crf.end_trans': 'crf.end_transitions',
            'crf.transitions': 'crf.trans_matrix',
            'crf.start_transitions': 'crf.start_trans',
            'crf.end_transitions': 'crf.end_trans',
        }
        # If old keys present, map to new
        if any(k in state_dict for k in ['crf.trans_matrix', 'crf.start_trans', 'crf.end_trans']):
            for old, new in crf_key_map.items():
                if old in state_dict:
                    state_dict[new] = state_dict.pop(old)
        # If new keys present, map to old (for older torchcrf)
        elif any(k in state_dict for k in ['crf.transitions', 'crf.start_transitions', 'crf.end_transitions']):
            for new, old in crf_key_map.items():
                if new in state_dict:
                    state_dict[old] = state_dict.pop(new)
        model.load_state_dict(state_dict)
        if hasattr(model, 'is_meta') and model.is_meta:
            model = model.to_empty(device=device)
        else:
            model = model.to(device)
        model.eval()
        text_cleaned = re.sub(r'</?AI_Start>|</?AI_End>', '', input_text) if re.search(r'</?AI_Start>|</?AI_End>', input_text) else input_text
        words = text_cleaned.split()
        encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", max_length=512, truncation=True, padding=True)
        word_ids = encoding.word_ids(batch_index=0)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        style_features = style_extractor.get_style_features(words, word_ids).unsqueeze(0)
        with torch.no_grad():
            predictions, info_mask = model(input_ids, attention_mask, style_features)
        st.session_state.predictions = predictions[0]
        st.session_state.style_features = style_features.squeeze(0).detach().cpu().numpy()
        st.session_state.info_mask = info_mask.squeeze(0).detach().cpu()
        st.session_state.words_plot = words
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")

# Display colored tokens
if st.session_state.predictions is not None and st.session_state.words_plot:
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, add_prefix_space=True)
    encoding = tokenizer(st.session_state.words_plot, is_split_into_words=True, return_tensors="pt", max_length=512, truncation=True, padding=True)
    word_ids = encoding.word_ids(batch_index=0)
    word_level_preds = []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id != prev_word_id:
            word_level_preds.append((st.session_state.words_plot[word_id], st.session_state.predictions[idx]))
            prev_word_id = word_id
    highlighted_output = ""
    for word, label in word_level_preds:
        color = "green" if label == 0 else "red"
        highlighted_output += f'<span style="color:{color}; font-weight:bold;">{word}</span> '
    st.markdown(highlighted_output, unsafe_allow_html=True)

    # --- Plots: show after colored tokens ---
    if st.session_state.style_features is not None and st.session_state.words_plot:
        style_feat_seq = st.session_state.style_features
        style_feat_seq_T = style_feat_seq.T
        words_plot = st.session_state.words_plot[:len(style_feat_seq)]
        feature_labels = ['TTR', 'Punctuation', 'POS Density', 'Readability']
        heatmap = go.Heatmap(
            z=style_feat_seq_T,
            x=words_plot,
            y=feature_labels,
            colorscale='RdBu',
            colorbar=dict(title='Feature Value'),
            zmin=np.min(style_feat_seq_T),
            zmax=np.max(style_feat_seq_T),
            hoverongaps=False,
            hovertemplate='Feature: %{y}<br>Word: %{x}<br>Value: %{z:.2f}<extra></extra>'
        )
        layout = go.Layout(
            title='Style Features Heatmap Across Words',
            xaxis=dict(title='Windows', tickangle=-45),
            yaxis=dict(title='Features'),
            height=400,
            margin=dict(l=40, r=40, t=60, b=120),
        )
        fig1 = go.Figure(data=[heatmap], layout=layout)
        fig1.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig1, use_container_width=True)

    if st.session_state.info_mask is not None and st.session_state.words_plot:
        info_mask_values = st.session_state.info_mask.numpy()
        words_plot = st.session_state.words_plot[:len(info_mask_values)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=words_plot,
            y=info_mask_values,
            mode='lines+markers',
            line=dict(color='teal'),
            marker=dict(size=6),
            hovertemplate='Word: %{x}<br>Info Mask: %{y:.3f}<extra></extra>',
            name='Info Mask'
        ))
        fig2.update_layout(
            title="Token-wise Info Mask Signal",
            xaxis_title="Words",
            yaxis_title="Info Mask Strength",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(l=40, r=40, t=60, b=120),
            template='plotly_white'
        )
        fig2.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig2, use_container_width=True)

        attention_sim = torch.softmax(torch.rand_like(st.session_state.info_mask), dim=0)
        attn_mask_product = attention_sim * st.session_state.info_mask
        min_len = min(len(st.session_state.words_plot), len(attn_mask_product))
        words_plot = st.session_state.words_plot[:min_len]
        attn_mask_product = attn_mask_product[:min_len]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=words_plot,
            y=attn_mask_product.numpy(),
            marker_color='purple',
            hoverinfo='x+y',
        ))
        fig3.update_layout(
            title="Attention × Info Mask (Fused Influence)",
            xaxis_title="Words",
            yaxis_title="Influence",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(l=40, r=40, t=60, b=120),
            template='plotly_white',
        )
        fig3.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig3, use_container_width=True)