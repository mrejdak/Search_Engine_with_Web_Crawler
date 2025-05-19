import gradio as gr
from SearchEngine import SearchEngine


def get_and_format_search_results(query, k, ann):
    k = int(k) if k != '' else 0
    results = SE.search(query, k=k, ann=ann)
    html = "<div style='font-family: Arial, sans-serif;'>"
    for result in results:
        html += f"""
        <div style='margin-bottom: 20px; padding: 10px; border-bottom: 1px solid #eee;'>
            <h3><a href='{result['url']}' target='_blank'>{result['title']}</a></h3>
            <p style='color: #555;'>{result['snippet']}</p>
        </div>
        """
    html += "</div>"
    return html

# Gradio interface
with gr.Blocks(title="Search Engine") as web_ui:
    gr.Markdown("# Search Engine")
    SE = SearchEngine()
    with gr.Row():
        with gr.Column(scale=4):
            search_input = gr.Textbox(label="Search query", placeholder="Enter your search terms...")
        with gr.Column(scale=1):
            k_value = gr.Textbox(label="Set k value:", placeholder="0")
    ann_checkbox = gr.Checkbox(label="Use ANN (works only when k > 0)", value=False)

    search_button = gr.Button("Search", variant="primary")
    results_output = gr.HTML(label="Search Results")


    search_button.click(
        fn=get_and_format_search_results,
        inputs=[search_input, k_value, ann_checkbox],
        outputs=results_output
    )

web_ui.launch()


