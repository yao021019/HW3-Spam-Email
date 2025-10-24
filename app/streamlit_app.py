"""Streamlit dashboard for Spam/Ham classifier (data + models).

Features:
- Select dataset CSV and column names
- Choose models directory, test size, seed, and decision threshold
- Data overview (class distribution, special token counts)
- Top-N tokens by class (using stored vectorizer if available)
- Model performance: confusion matrix and threshold sweep
- Live inference with autofill ham/spam example buttons

Run with:
    streamlit run app/streamlit_app.py
"""
import json
import math
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split


DEFAULTS = {
    "dataset": "datasets/processed/sms_spam_clean.csv",
    "label_col": "col_0",
    "text_col": "col_1",
    "model_dir": "models",
    "test_size": 0.1,
    "seed": 42,
    "threshold": 0.5,
}

# fallback dataset URL (raw CSV)
DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"


@st.cache_resource
def load_model_and_vectorizer(model_dir: str):
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    # try multiple model filenames; prefer SVM if present
    candidates = ["svm_linear.joblib", "linear_svc.joblib", "svm.joblib", "logistic_regression.joblib", "logistic.joblib", "model.joblib"]
    model = None
    vec = None
    if os.path.exists(vec_path):
        try:
            vec = joblib.load(vec_path)
        except Exception:
            vec = None
    for c in candidates:
        p = os.path.join(model_dir, c)
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception:
                model = None
    return model, vec


def safe_sigmoid(x):
    # numerically stable sigmoid
    return 1 / (1 + np.exp(-x))


def get_probs(model, X):
    # Return probability for positive class
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probs = safe_sigmoid(scores)
    else:
        preds = model.predict(X)
        probs = preds.astype(float)
    return probs


def detect_columns(df: pd.DataFrame, label_hint: str, text_hint: str):
    # Try provided hints first, then common alternatives
    label_cols = [label_hint, "label", "Label", "y", "target", "class", "category", "spam"]
    text_cols = [text_hint, "message", "text", "sms", "body", "message_text"]
    found_label = None
    found_text = None
    for c in label_cols:
        if c in df.columns:
            found_label = c
            break
    for c in text_cols:
        if c in df.columns:
            found_text = c
            break
    return found_label, found_text


def map_labels(series: pd.Series):
    # Return (binary_array, mapping) where mapping maps original values to 0/1
    vals = series.dropna().unique()
    lower_vals = [str(v).lower() for v in vals]
    mapping = {}
    if any("spam" == v for v in lower_vals):
        for v in vals:
            mapping[v] = 1 if str(v).lower() == "spam" else 0
    elif set(lower_vals) <= {"0", "1"}:
        for v in vals:
            mapping[v] = int(str(v))
    else:
        # fallback: map the minority class to 1 (assume spam is minority)
        counts = series.value_counts()
        if len(counts) >= 2:
            minority = counts.idxmin()
            for v in vals:
                mapping[v] = 1 if v == minority else 0
        else:
            # default map anything truthy to 1
            for v in vals:
                mapping[v] = 1 if str(v).strip() not in ["", "0", "false", "none"] else 0
    # apply mapping to series
    binary = series.map(mapping).fillna(0).astype(int).to_numpy()
    return binary, mapping


def pick_example_by_prob(df: pd.DataFrame, text_col: str, mask, model, vec, prefer_label: int, target_prob: float = 0.9):
    """Pick an example from df[text_col] where mask is True that the model predicts
    with probability >= target_prob for prefer_label. If none found, return the example
    with the highest (for spam) or lowest (for ham) probability.
    """
    try:
        candidates = df[mask][text_col].astype(str).tolist()
        if not candidates:
            return None
        if model is None or vec is None:
            return candidates[0]
        # compute probabilities in batches
        X = vec.transform(candidates)
        probs = get_probs(model, X)
        # if prefer_label==1 (spam) we want probs >= target_prob
        if prefer_label == 1:
            idxs = [i for i, p in enumerate(probs) if p >= target_prob]
            if idxs:
                return candidates[idxs[0]]
            # fallback to highest prob
            best = int(np.argmax(probs))
            return candidates[best]
        else:
            # prefer low prob for ham
            idxs = [i for i, p in enumerate(probs) if p <= (1 - target_prob)]
            if idxs:
                return candidates[idxs[0]]
            best = int(np.argmin(probs))
            return candidates[best]
    except Exception:
        return None


def top_tokens_by_class(vec: TfidfVectorizer, X, y, top_n: int = 20):
    # sum tf-idf per class and take top tokens
    try:
        feature_names = np.array(vec.get_feature_names_out())
    except Exception:
        return {}, {}
    X_arr = X.toarray() if hasattr(X, "toarray") else X
    tokens_ham = X_arr[y == 0].sum(axis=0)
    tokens_spam = X_arr[y == 1].sum(axis=0)
    top_ham_idx = np.argsort(tokens_ham)[-top_n:][::-1]
    top_spam_idx = np.argsort(tokens_spam)[-top_n:][::-1]
    return list(feature_names[top_ham_idx]), list(feature_names[top_spam_idx])


def compute_threshold_metrics(y_true, probs, thresholds):
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


def main():
    st.set_page_config(layout="wide", page_title="Spam/Ham Classifier — Phase 4 Visualizations")
    st.title("Spam/Ham Classifier — Phase 4 Visualizations")
    st.markdown("Interactive dashboard for data distribution, token patterns, and model performance")

    with st.sidebar.expander("Inputs"):
        uploaded = st.file_uploader("Upload CSV dataset", type=["csv"] ,help="Optional: upload a CSV to override the dataset path")
        dataset_path = st.text_input("Dataset CSV", DEFAULTS["dataset"])
        label_col = st.text_input("Label column", DEFAULTS["label_col"])
        text_col = st.text_input("Text column", DEFAULTS["text_col"])
        model_dir = st.text_input("Models dir", DEFAULTS["model_dir"])
        test_size = st.slider("Test size", 0.05, 0.5, float(DEFAULTS["test_size"]), step=0.05)
        seed = st.number_input("Seed", value=int(DEFAULTS["seed"]), step=1)
        threshold = st.slider("Decision threshold", 0.0, 1.0, float(DEFAULTS["threshold"]), step=0.01)
        invert_pred = st.checkbox("Invert model predictions for display (flip predicted class)", value=False, help="Use this if the model's positive class is opposite of the dataset mapping")

    # Load dataset: prefer uploaded file, then explicit path, then fallback to data/sms_spam.csv
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded dataset: {e}")
    else:
        chosen_paths = [dataset_path, "data/sms_spam.csv"]
        for p in chosen_paths:
            if p and os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    break
                except Exception as e:
                    st.error(f"Failed to read dataset at {p}: {e}")

        # If no local dataset found, try to download the canonical dataset automatically
        if df is None:
            try:
                os.makedirs(os.path.dirname("data/sms_spam.csv"), exist_ok=True)
                r = requests.get(DATA_URL, timeout=15)
                if r.status_code == 200 and r.content:
                    with open("data/sms_spam.csv", "wb") as fh:
                        fh.write(r.content)
                    df = pd.read_csv("data/sms_spam.csv", header=None)
                    # The canonical file may not have headers; try to normalize to 'label' and 'message'
                    if df.shape[1] >= 2:
                        df = df.rename(columns={0: 'label', 1: 'message'})
                    st.info("Downloaded dataset automatically to data/sms_spam.csv")
                else:
                    st.info("No dataset found locally and automatic download failed.")
            except Exception:
                st.info("No dataset found locally and automatic download failed.")

        # Detect likely label/text columns if provided hints are wrong
        detected_label, detected_text = (None, None)
        if df is not None:
            detected_label, detected_text = detect_columns(df, label_col, text_col)
            if detected_label and detected_label != label_col:
                st.info(f"Detected label column '{detected_label}' in dataset; using it for analysis.")
                label_col = detected_label
            if detected_text and detected_text != text_col:
                st.info(f"Detected text column '{detected_text}' in dataset; using it for analysis.")
                text_col = detected_text

    # Prepare a held-out test split for demo autofill and evaluation to avoid using the training set
    demo_test_df = None
    label_mapping_full = None
    if df is not None and label_col in df.columns and text_col in df.columns:
        try:
            y_all, label_mapping_full = map_labels(df[label_col])
            # create stratified indices split so autofill uses unseen examples
            indices = df.index.to_list()
            try:
                train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=int(seed), stratify=y_all)
                demo_test_df = df.loc[test_idx].reset_index(drop=True)
            except Exception:
                # fallback without stratify
                train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=int(seed))
                demo_test_df = df.loc[test_idx].reset_index(drop=True)
        except Exception:
            demo_test_df = None

    model, vec = load_model_and_vectorizer(model_dir)

    # Add a small control to force reloading model artifacts without restarting Streamlit
    with st.sidebar:
        if st.button("Reload model from models dir"):
            try:
                st.cache_resource.clear()
            except Exception:
                # older streamlit versions may not expose clear(), fall back to a no-op
                pass
            model, vec = load_model_and_vectorizer(model_dir)
            st.success("Reloaded model and vectorizer (from '{}' if available).".format(model_dir))

    # Layout: left = data overview, center = tokens/top tokens, right = model performance
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.header("Data Overview")
        if df is None:
            st.info("No dataset found at the selected path. Upload or set a valid path. Example: data/sms_spam.csv")
        else:
            st.subheader("Class distribution")
            if label_col not in df.columns:
                st.warning(f"Label column '{label_col}' not found in CSV columns: {list(df.columns)}")
            else:
                counts = df[label_col].value_counts().sort_index()
                fig, ax = plt.subplots()
                counts.plot(kind="bar", ax=ax)
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                st.pyplot(fig)

            st.subheader("Token replacements in cleaned text (approximate)")
            if text_col in (df.columns if df is not None else []):
                tokens = ["<URL>", "<EMAIL>", "<PHONE>", "<NUM>"]
                token_counts = {t: int(df[text_col].astype(str).str.count(t).sum()) for t in tokens}
                st.table(pd.DataFrame.from_dict(token_counts, orient="index", columns=["count"]))

    # Show detected mapping info if available
    if df is not None and 'label_mapping_full' in locals():
        try:
            mapping_preview = {str(k): int(v) for k, v in label_mapping_full.items()}
            st.sidebar.markdown("**Detected label mapping (dataset -> binary)**")
            st.sidebar.write(mapping_preview)
            pos_vals = [k for k, v in mapping_preview.items() if v == 1]
            st.sidebar.caption(f"Positive class values: {pos_vals}")
        except Exception:
            pass

    with c2:
        st.header("Top Tokens by Class")
        top_n = st.slider("Top-N tokens", 5, 100, 10)
        if df is None or text_col not in df.columns or label_col not in df.columns:
            st.info("Provide dataset and correct columns to view top tokens.")
        else:
            # Ensure vectorizer available; if not, fit a local one
            if vec is None:
                local_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
                X_local = local_vec.fit_transform(df[text_col].astype(str).tolist())
                vec_for_top = local_vec
                X_for_top = X_local
            else:
                X_for_top = vec.transform(df[text_col].astype(str).tolist())
                vec_for_top = vec
            # Map labels to binary robustly
            y, label_mapping = map_labels(df[label_col])
            ham_tokens, spam_tokens = top_tokens_by_class(vec_for_top, X_for_top, y, top_n)
            st.subheader("Class: ham")
            if ham_tokens:
                fig_h, axh = plt.subplots()
                axh.barh(range(len(ham_tokens)), [1]*len(ham_tokens))
                axh.set_yticks(range(len(ham_tokens)))
                axh.set_yticklabels(ham_tokens)
                st.pyplot(fig_h)
            else:
                st.write("No ham tokens available")
            st.subheader("Class: spam")
            if spam_tokens:
                fig_s, axs = plt.subplots()
                axs.barh(range(len(spam_tokens)), [1]*len(spam_tokens))
                axs.set_yticks(range(len(spam_tokens)))
                axs.set_yticklabels(spam_tokens)
                st.pyplot(fig_s)
            else:
                st.write("No spam tokens available")

    with c3:
        st.header("Model Performance (Test)")
        if df is None or label_col not in df.columns or text_col not in df.columns:
            st.info("Provide dataset and correct columns to evaluate model performance.")
        else:
            # Split and evaluate
            X_text = df[text_col].astype(str).tolist()
            y_full, label_mapping_full = map_labels(df[label_col])
            X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y_full, test_size=test_size, random_state=int(seed), stratify=y_full)

            # Use vectorizer if available, else fit on train
            if vec is None:
                vec_local = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
                X_train = vec_local.fit_transform(X_train_text)
                X_test = vec_local.transform(X_test_text)
            else:
                try:
                    X_train = vec.transform(X_train_text)
                    X_test = vec.transform(X_test_text)
                except Exception:
                    # fallback: fit local
                    vec_local = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
                    X_train = vec_local.fit_transform(X_train_text)
                    X_test = vec_local.transform(X_test_text)

            if model is None:
                # If no persisted model is available, fit a quick fallback model on the dataset so the demo can run
                st.warning("No trained model found in models dir. Training a small fallback model on the dataset for demo purposes...")
                try:
                    fallback_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
                    X_all = fallback_vec.fit_transform(df[text_col].astype(str).tolist())
                    y_all, _ = map_labels(df[label_col])
                    from sklearn.linear_model import LogisticRegression
                    fallback_model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
                    fallback_model.fit(X_all, y_all)
                    model = fallback_model
                    vec = fallback_vec
                    # Save artifacts to models dir if possible
                    try:
                        os.makedirs(model_dir, exist_ok=True)
                        joblib.dump(vec, os.path.join(model_dir, 'vectorizer.joblib'))
                        joblib.dump(model, os.path.join(model_dir, 'logistic_regression.joblib'))
                    except Exception:
                        pass
                    st.success("Fallback model trained and ready for demo (saved to models/ if writable).")
                    # After training the fallback model, run the same evaluation pipeline so the performance section is populated
                    try:
                        probs = get_probs(model, X_test)
                        # compute predicted class using threshold, apply invert toggle if requested
                        if invert_pred:
                            preds = (probs < threshold).astype(int)
                        else:
                            preds = (probs >= threshold).astype(int)
                        cm = confusion_matrix(y_test, preds)
                        st.subheader("Confusion matrix")
                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        st.pyplot(fig_cm)

                        # ROC curve
                        try:
                            from sklearn.metrics import roc_curve, auc

                            fpr, tpr, _ = roc_curve(y_test, probs)
                            roc_auc = auc(fpr, tpr)
                            fig_roc, ax_roc = plt.subplots()
                            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                            ax_roc.set_xlabel("False Positive Rate")
                            ax_roc.set_ylabel("True Positive Rate")
                            ax_roc.legend()
                            st.subheader("ROC curve")
                            st.pyplot(fig_roc)
                        except Exception:
                            st.info("Could not compute ROC curve for this model")

                        st.subheader("Threshold sweep (precision/recall/f1)")
                        thresholds = np.linspace(0.01, 0.99, 50)
                        df_thresh = compute_threshold_metrics(y_test, probs, thresholds)
                        st.dataframe(df_thresh)
                        st.line_chart(df_thresh.set_index("threshold")[['precision','recall','f1']])
                    except Exception:
                        st.info("Could not evaluate fallback model on test split.")
                except Exception as e:
                    st.error(f"Failed to train fallback model: {e}")
            else:
                probs = get_probs(model, X_test)
                # compute predicted class using threshold, apply invert toggle if requested
                if invert_pred:
                    preds = (probs < threshold).astype(int)
                else:
                    preds = (probs >= threshold).astype(int)
                cm = confusion_matrix(y_test, preds)
                st.subheader("Confusion matrix")
                # plot heatmap
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

                # ROC curve
                try:
                    from sklearn.metrics import roc_curve, auc

                    fpr, tpr, _ = roc_curve(y_test, probs)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.legend()
                    st.subheader("ROC curve")
                    st.pyplot(fig_roc)
                except Exception:
                    st.info("Could not compute ROC curve for this model")

                st.subheader("Threshold sweep (precision/recall/f1)")
                thresholds = np.linspace(0.01, 0.99, 50)
                df_thresh = compute_threshold_metrics(y_test, probs, thresholds)
                st.dataframe(df_thresh)
                st.line_chart(df_thresh.set_index("threshold")[['precision','recall','f1']])

    st.markdown("---")
    st.header("Live Inference")

    # Live inference with autofill buttons
    if 'inference_text' not in st.session_state:
        st.session_state['inference_text'] = ""

    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        if st.button("Fill ham example"):
            if df is not None and label_col in df.columns:
                try:
                    mask_ham = df[label_col].map(label_mapping_full) == 0
                except Exception:
                    mask_ham = df[label_col].astype(str).str.lower() != 'spam'
                # pick example with low spam probability (<=0.1) if possible
                example = pick_example_by_prob(df, text_col, mask_ham, model, vec, prefer_label=0, target_prob=0.9)
                if example:
                    st.session_state['inference_text'] = example
                else:
                    ham_example = df[mask_ham][text_col].astype(str).head(1).tolist()
                    st.session_state['inference_text'] = ham_example[0] if ham_example else "Hello, are we still on for lunch?"
            else:
                st.session_state['inference_text'] = "Hello, are we still on for lunch?"
    with col_b:
        if st.button("Fill spam example"):
            if df is not None and label_col in df.columns:
                try:
                    mask_spam = df[label_col].map(label_mapping_full) == 1
                except Exception:
                    mask_spam = df[label_col].astype(str).str.lower() == 'spam'
                # pick example with high spam probability (>=0.9) if possible
                example = pick_example_by_prob(df, text_col, mask_spam, model, vec, prefer_label=1, target_prob=0.9)
                if example:
                    st.session_state['inference_text'] = example
                else:
                    # No example reached the high-confidence threshold. Show top-K spam candidates with their predicted probs
                    if model is not None and vec is not None and mask_spam.sum() > 0:
                        try:
                            candidates = df[mask_spam][text_col].astype(str).tolist()
                            Xc = vec.transform(candidates)
                            probs = get_probs(model, Xc)
                            order = list(np.argsort(-probs))[:5]
                            rows = []
                            idxs = df[mask_spam].index.tolist()
                            for i in order:
                                rows.append({
                                    "prob": float(probs[i]),
                                    "text": candidates[i][:300]
                                })
                            st.warning("No spam example reached p>=0.9. Showing top candidates (highest predicted spam probability):")
                            st.table(pd.DataFrame(rows))
                            # set inference_text to the highest-prob candidate anyway (honest display)
                            best = order[0]
                            st.session_state['inference_text'] = candidates[best]
                        except Exception:
                            spam_example = df[mask_spam][text_col].astype(str).head(1).tolist()
                            st.session_state['inference_text'] = spam_example[0] if spam_example else "Congratulations! You've won a prize. Click here!"
                    else:
                        spam_example = df[mask_spam][text_col].astype(str).head(1).tolist()
                        st.session_state['inference_text'] = spam_example[0] if spam_example else "Congratulations! You've won a prize. Click here!"
            else:
                st.session_state['inference_text'] = "Congratulations! You've won a prize. Click here!"

    with col_c:
        st.text_area("Enter a message to classify", key='inference_text', height=120)
        if st.button("Classify message"):
            if model is None or vec is None:
                st.error("Model or vectorizer not available. Ensure models exist in the configured models dir.")
            else:
                x = vec.transform([st.session_state['inference_text']])
                prob = get_probs(model, x)[0]
                pred = int(prob >= threshold)
                label = "SPAM" if pred == 1 else "HAM"
                st.success(f"Prediction: {label} (p={prob:.3f})")


if __name__ == "__main__":
    main()
