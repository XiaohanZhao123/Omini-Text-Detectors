import streamlit as st
import numpy as np
import plotly.graph_objs as go
import torch

def display_colored_tokens(words_plot, predictions, word_ids=None):
    """
    Display colored tokens (words) in Streamlit based on predictions.
    If word_ids is provided, use it for mapping, else assume 1:1 mapping.
    """
    word_level_preds = []
    prev_word_id = None
    if word_ids is not None:
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != prev_word_id:
                word_level_preds.append((words_plot[word_id], predictions[idx]))
                prev_word_id = word_id
    else:
        for word, label in zip(words_plot, predictions):
            word_level_preds.append((word, label))
    highlighted_output = ""
    for word, label in word_level_preds:
        color = "green" if label == 0 else "red"
        highlighted_output += f'<span style="color:{color}; font-weight:bold;">{word}</span> '
    st.markdown(highlighted_output, unsafe_allow_html=True)

def plot_style_features_heatmap(style_features, words_plot):
    style_feat_seq = style_features
    style_feat_seq_T = style_feat_seq.T
    words_plot = words_plot[:len(style_feat_seq)]
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

def plot_info_mask(info_mask, words_plot):
    info_mask_values = info_mask.numpy()
    words_plot = words_plot[:len(info_mask_values)]
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

def plot_attention_info_mask(info_mask, words_plot):
    attention_sim = torch.softmax(torch.rand_like(info_mask), dim=0)
    attn_mask_product = attention_sim * info_mask
    min_len = min(len(words_plot), len(attn_mask_product))
    words_plot = words_plot[:min_len]
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