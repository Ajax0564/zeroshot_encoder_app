import html
import warnings
import requests
import gradio as gr

# suppress internal Starlette/Gradio deprecation warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="starlette")
warnings.filterwarnings("ignore", module="gradio")

API_URL = "http://127.0.0.1:8000"

def api_post(endpoint: str, payload: dict):
    """POST request to FastAPI."""
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=300,
        )
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            return {"error": True, "status_code": response.status_code, "detail": detail}
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": True, "detail": f"Cannot connect to API at {API_URL}. Make sure Uvicorn is running."}
    except requests.exceptions.Timeout:
        return {"error": True, "detail": "Request timed out. The model may still be processing."}
    except Exception as e:
        return {"error": True, "detail": str(e)}

def api_get(endpoint: str):
    """GET request to FastAPI."""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": True, "detail": str(e)}

def api_delete(endpoint: str):
    """DELETE request to FastAPI."""
    try:
        response = requests.delete(f"{API_URL}{endpoint}", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": True, "detail": str(e)}


def parse_labels(label_text: str):
    """Convert multi-line or comma-separated labels into a list."""
    if not label_text:
        return []
    labels = []
    for line in label_text.splitlines():
        parts = line.split(",")
        for part in parts:
            label = part.strip()
            if label:
                labels.append(label)
    return labels

def parse_texts(text_text: str):
    """Convert multiline text input into a list of non-empty strings."""
    if not text_text:
        return []
    return [line.strip() for line in text_text.splitlines() if line.strip()]

def get_dynamic_colors(num_colors: int):
    """Generates an evenly spaced, visually appealing HSL color palette."""
    if num_colors == 0:
        return []
    # Generates pastel colors spanning the color wheel dynamically based on label count
    return [f"hsl({int(i * (360 / num_colors))}, 85%, 88%)" for i in range(num_colors)]


def render_highlighted_entities(results: list) -> str:
    """Renders entity extraction results as HTML with dynamically generated colors."""
    if not results:
        return "<p style='color: #6b7280; font-style: italic;'>No results returned.</p>"

    # 1. Identify unique labels to generate dynamic color palette
    unique_labels = set()
    for item in results:
        entities_dict = item.get("entities", {})
        if "entities" in entities_dict and isinstance(entities_dict["entities"], dict):
            entities_dict = entities_dict["entities"]
        if isinstance(entities_dict, dict):
            for label_key in entities_dict.keys():
                unique_labels.add(label_key)

    sorted_labels = sorted(list(unique_labels))
    dynamic_colors = get_dynamic_colors(len(sorted_labels))
    label_colors = {lbl: dynamic_colors[i] for i, lbl in enumerate(sorted_labels)}

    # 2. Build HTML Output
    html_out = "<div style='font-family: ui-sans-serif, system-ui, sans-serif; line-height: 1.8;'>"

    if label_colors:
        html_out += "<div style='margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 12px; background: #f9fafb; border-radius: 8px; border: 1px solid #f3f4f6;'>"
        html_out += "<strong style='font-size: 13px; color: #4b5563; margin-right: 8px;'>Detected Entities:</strong>"
        for lbl, col in label_colors.items():
            html_out += f"""
            <span style='background-color: {col}; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; color: #1f2937; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>
                {html.escape(lbl)}
            </span>
            """
        html_out += "</div>"

    for index, item in enumerate(results, 1):
        text = item.get("text", "")
        entities_dict = item.get("entities", {})
        if "entities" in entities_dict and isinstance(entities_dict["entities"], dict):
            entities_dict = entities_dict["entities"]

        spans = []
        if isinstance(entities_dict, dict):
            for cat_label, entity_list in entities_dict.items():
                if isinstance(entity_list, list):
                    for ent in entity_list:
                        if isinstance(ent, dict):
                            spans.append({
                                "start": ent.get("start", 0),
                                "end": ent.get("end", 0),
                                "text": ent.get("text", ""),
                                "label": cat_label,
                                "confidence": ent.get("confidence", None)
                            })

        spans.sort(key=lambda x: x["start"])

        last_idx = 0
        text_html = ""

        for span in spans:
            start, end = span["start"], span["end"]
            if start < last_idx:
                continue

            text_html += html.escape(text[last_idx:start])

            color = label_colors.get(span["label"], "#E5E7EB")
            entity_text_escaped = html.escape(text[start:end])
            conf_str = f" ({span['confidence']:.2f})" if span["confidence"] is not None else ""

            text_html += f"""<mark style='background-color: {color}; padding: 3px 6px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-weight: 500;' title='{span["label"]}{conf_str}'>{entity_text_escaped}<sup style='font-size: 9px; font-weight: bold; margin-left: 4px; color: rgba(0,0,0,0.5); text-transform: uppercase;'>{html.escape(span["label"])}</sup></mark>"""
            last_idx = end

        text_html += html.escape(text[last_idx:])

        html_out += f"""
        <div style='background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);'>
            <div style='font-size: 11px; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>Result {index}</div>
            <div style='font-size: 16px; color: #1f2937;'>{text_html}</div>
        </div>
        """

    html_out += "</div>"
    return html_out

def render_classification_markdown(results: list, task_name: str) -> str:
    """Renders text classification outputs."""
    if not results:
        return "*No classification results returned.*"

    md = ""
    for idx, item in enumerate(results, 1):
        text = item.get("text", "")
        classification_data = item.get("classification", {}).get(task_name, [])

        md += f"### Text {idx}\n> {text}\n\n**Predicted Labels:**\n\n"

        if isinstance(classification_data, dict):
            features_list = [classification_data]
        elif isinstance(classification_data, list):
            features_list = classification_data
        else:
            features_list = [str(classification_data)]

        if features_list:
            for feat in features_list:
                if isinstance(feat, dict):
                    label = feat.get("label", "N/A")
                    confidence = feat.get("confidence", None)
                    conf_badge = f" `({confidence * 100:.1f}%)`" if confidence is not None else ""
                    md += f"- **{label}**{conf_badge}\n"
                elif isinstance(feat, str):
                    md += f"- **{feat}**\n"
        else:
            md += "*No matching labels found.*\n"

        md += "\n---\n"

    return md



def run_classification(texts, labels, task_name, is_multilabel, include_confidence):
    texts_list = parse_texts(texts)
    labels_list = parse_labels(labels)

    if not texts_list:
        return {"error": True}, "⚠️ **Error:** Please enter at least one text."
    if not labels_list:
        return {"error": True}, "⚠️ **Error:** Please enter at least one label."
    if not task_name.strip():
        return {"error": True}, "⚠️ **Error:** Please enter a task name."

    payload = {
        "texts": texts_list,
        "labels": labels_list,
        "task_name": task_name.strip(),
        "is_multilabel": is_multilabel,
        "include_confidence": include_confidence,
    }

    raw_response = api_post("/classify", payload)

    if "error" in raw_response and raw_response["error"]:
        return raw_response, f"⚠️ **API Request Failed:** {raw_response.get('detail', 'Unknown error')}"

    results = raw_response.get("results", [])
    md_output = render_classification_markdown(results, task_name.strip())
    return raw_response, md_output


def run_entities(texts, labels, task_name, include_confidence, threshold):
    texts_list = parse_texts(texts)
    labels_list = parse_labels(labels)

    if not texts_list:
        return {"error": True}, "<p style='color: #ef4444; font-weight: 600;'>Error: Please enter at least one text.</p>"
    if not labels_list:
        return {"error": True}, "<p style='color: #ef4444; font-weight: 600;'>Error: Please enter at least one entity label.</p>"
    if not task_name.strip():
        return {"error": True}, "<p style='color: #ef4444; font-weight: 600;'>Error: Please enter a task name.</p>"

    payload = {
        "texts": texts_list,
        "labels": labels_list,
        "task_name": task_name.strip(),
        "include_confidence": include_confidence,
        "threshold": threshold,
    }

    raw_response = api_post("/entities", payload)

    if "error" in raw_response and raw_response["error"]:
        return raw_response, f"<p style='color: #ef4444; font-weight: 600;'>API Request Failed: {raw_response.get('detail', 'Unknown error')}</p>"

    results = raw_response.get("results", [])
    html_output = render_highlighted_entities(results)
    return raw_response, html_output


def check_api():
    return api_get("/")


def clear_cache():
    return api_delete("/cache")



CSS = """
.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
}
.app-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.app-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: 0.5rem;
}
.app-subtitle {
    color: #6b7280;
    font-size: 1.1rem;
}
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
}
"""

with gr.Blocks(title="GLiNER 2.5 Playground", css=CSS, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="gray")) as demo:

    gr.HTML("""
        <div class="app-header">
            <h1>🚀 GLiNER 2.5 Playground</h1>
            <p class="app-subtitle">Interactive testing interface for zero-shot classification and named entity extraction.</p>
        </div>
    """)


    with gr.Tab("🏷️ Classification"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    classification_task = gr.Textbox(
                        label="Task Name",
                        value="product_features",
                        placeholder="e.g. sentiment, product_features",
                    )
                    classification_text = gr.Textbox(
                        label="Texts",
                        placeholder="Enter one text per line...",
                        lines=6,
                    )
                    classification_labels = gr.Textbox(
                        label="Labels (Comma separated or one per line)",
                        value="camera, battery, price, delivery, packaging",
                        placeholder="Enter labels...",
                        lines=4,
                    )
                with gr.Row():
                    classification_multilabel = gr.Checkbox(label="Multi-label", value=True)
                    classification_confidence = gr.Checkbox(label="Return Confidence Scores", value=True)

                classify_button = gr.Button("🚀 Run Classification", variant="primary", size="lg")

            with gr.Column(scale=1):
                classification_formatted = gr.Markdown(
                    label="Formatted Classification Results",
                    value="*Results will appear here...*",
                )
                with gr.Accordion("Raw JSON Response", open=False):
                    classification_output = gr.JSON(label="JSON Payload")

        classify_button.click(
            fn=run_classification,
            inputs=[classification_text, classification_labels, classification_task, classification_multilabel, classification_confidence],
            outputs=[classification_output, classification_formatted],
        )

        # FIX: Only mapping text-based fields prevents Gradio Dataframe errors
        gr.Examples(
            label="Try an example (Click to load)",
            examples=[
                ["The phone has an excellent camera and great battery life.", "camera, battery, price, performance", "product_features"],
                ["The delivery was late and the packaging was damaged.", "delivery, packaging, quality", "customer_experience"],
                ["The movie was entertaining but the ending was disappointing.", "positive, negative, neutral", "sentiment"],
                ["The service was okay, nothing special but not bad either.", "positive, negative, neutral", "sentiment"],
                ["Apple announced new health monitoring features in their latest smartwatch, boosting their stock price.","technology,business,health,politics,sports","topics"]
            ],
            inputs=[classification_text, classification_labels, classification_task],
        )

    with gr.Tab("🔎 Entity Extraction"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    entity_task = gr.Textbox(
                        label="Task Name",
                        value="named_entities",
                        placeholder="e.g. named_entities",
                    )
                    entity_text = gr.Textbox(
                        label="Texts",
                        placeholder="Enter one text per line...",
                        lines=6,
                    )
                    entity_labels = gr.Textbox(
                        label="Entity Labels (Comma separated or one per line)",
                        value="company, product, location, person",
                        placeholder="Enter entity labels...",
                        lines=4,
                    )
                with gr.Row():
                    entity_confidence = gr.Checkbox(label="Include Confidence", value=True)
                entity_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                    label="Confidence Threshold",
                    info="Higher values return fewer but more confident entities.",
                )

                entity_button = gr.Button("🔎 Extract Entities", variant="primary", size="lg")

            with gr.Column(scale=1):
                entity_html_output = gr.HTML(
                    label="Highlighted Entities Output",
                    value="<div style='color: #6b7280; text-align: center; padding: 40px;'>Results will appear here...</div>",
                )
                with gr.Accordion("Raw JSON Response", open=False):
                    entity_output = gr.JSON(label="JSON Payload")

        entity_button.click(
            fn=run_entities,
            inputs=[entity_text, entity_labels, entity_task, entity_confidence, entity_threshold],
            outputs=[entity_output, entity_html_output],
        )

        # FIX: Only mapping text-based fields prevents Gradio Dataframe errors
        gr.Examples(
            label="Try an example (Click to load)",
            examples=[
                ["Sundar Pichai is the CEO of Google. Google is headquartered in Mountain View, California.", "person, company, location, job_title", "named_entities"],
                ["Apple Inc. CEO Tim Cook announced the new iPhone 15 in Cupertino, California on September 12, 2023.","company,person,product,location,date","named_entities"],
                ["Elon Musk founded SpaceX and Tesla. Tesla is based in Austin, Texas.", "person, company, location", "named_entities"],
                ["Apple announced the new iPhone in Cupertino on September 10.", "company, product, location, date", "named_entities"],
            ],
            inputs=[entity_text, entity_labels, entity_task],
        )

   
    with gr.Tab("⚙️ System Configuration"):
        with gr.Row():
            with gr.Column():
                check_button = gr.Button("🔄 Check API Status", variant="secondary")
                api_status = gr.JSON(label="API Status")
            with gr.Column():
                cache_button = gr.Button("🗑️ Clear Redis Cache", variant="stop")
                cache_status = gr.JSON(label="Cache Status")

        check_button.click(fn=check_api, outputs=api_status)
        cache_button.click(fn=clear_cache, outputs=cache_status)

    gr.HTML("""
        <div class="footer">
            <p>GLiNER 2.5 Explorer • Powered by FastAPI, Uvicorn & Redis</p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )