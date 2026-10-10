"""The page layout. Every element has a CSS class starting with
`beatle_rag_`, so src/app/style.css can style any of them:

    beatle_rag_container        the centred column holding everything
    beatle_rag_header           title and subtitle
      beatle_rag_title          the <h1>, holding the logo
        beatle_rag_logo         the title image (the title text is its alt)
      beatle_rag_subtitle       the line under it
    beatle_rag_api_key_box      the border around the key and its help line
      beatle_rag_api_key        the API key box
      beatle_rag_api_key_help   the line under it
    beatle_rag_chat_box         one border around the conversation and the question
      beatle_rag_chat           the conversation
      beatle_rag_input_row      question box and button
        beatle_rag_question     the question box
        beatle_rag_ask_button   the Ask button
    beatle_rag_examples         the example questions
    beatle_rag_footer           sources, licence and a warning

Inside each answer (written by render.py): beatle_rag_status,
beatle_rag_citation, beatle_rag_sources, beatle_rag_sources_title,
beatle_rag_source, beatle_rag_source_number, beatle_rag_source_link,
beatle_rag_source_section, beatle_rag_notice and beatle_rag_answer_meta.

The chat's own bin icon clears the conversation; it also empties the
history the model sees, or the next answer would follow a conversation the
user no longer sees."""

from __future__ import annotations

import base64
import html
from collections.abc import Callable, Iterator
from pathlib import Path

import gradio as gr

from src.app.chat import Message
from src.config import AppConfig
from src.rag.agent import Turn

Respond = Callable[[str, str, list[Message], list[Turn]], Iterator[tuple[str, list[Message], list[Turn]]]]


def data_uri(path: Path, media_type: str) -> str:
    """The file inside the page itself, so Gradio does not have to be told
    which folders it may serve files from."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def font_face_css(font_path: Path) -> str:
    """Declares Inter from the bundled file, so the page never depends on the
    visitor having it installed. It holds Latin letters (Spanish, Italian,
    French... included); other scripts fall back to the system fonts listed
    after it in style.css. One variable file covers every weight."""
    return (
        "@font-face {"
        " font-family: 'Inter';"
        " font-style: normal;"
        " font-weight: 100 900;"
        " font-display: swap;"
        f" src: url({data_uri(font_path, 'font/woff2')}) format('woff2');"
        " }"
    )


def header_html(title: str, subtitle: str, logo_path: Path) -> str:
    """The title stays an <h1> with the title as alt text, so screen readers
    and search engines still read it."""
    return (
        f'<h1 class="beatle_rag_title">'
        f'<img class="beatle_rag_logo" src="{data_uri(logo_path, "image/png")}" alt="{html.escape(title)}">'
        f'</h1>'
        f'<p class="beatle_rag_subtitle">{html.escape(subtitle)}</p>'
    )


def build_ui(app: AppConfig, max_question_chars: int, respond: Respond) -> gr.Blocks:
    """`respond` is chat.respond with its context already bound (app.py)."""
    with gr.Blocks(title=app.title, analytics_enabled=False) as demo:
        turns = gr.State([])  # one list per browser session: the model's history

        with gr.Column(elem_classes="beatle_rag_container"):
            gr.HTML(header_html(app.title, app.subtitle, app.logo_path), elem_classes="beatle_rag_header")

            with gr.Group(elem_classes="beatle_rag_api_key_box"):  # one bordered box for both
                api_key = gr.Textbox(
                    show_label=False,
                    type="password",
                    placeholder=app.api_key_label,
                    elem_classes="beatle_rag_api_key",
                )
                gr.Markdown(app.api_key_help, elem_classes="beatle_rag_api_key_help")

            # One group, so the chat and the question row share one border with
            # no gap between them, whatever spacing Gradio uses on the screen.
            with gr.Group(elem_classes="beatle_rag_chat_box"):
                chat = gr.Chatbot(
                    show_label=False,
                    height=520,
                    sanitize_html=True,
                    render_markdown=True,
                    latex_delimiters=[],  # "$0.0009" is a price, not maths
                    buttons=["copy"],     # no "share": it would post the chat to Hugging Face
                    elem_classes="beatle_rag_chat",
                )

                with gr.Row(elem_classes="beatle_rag_input_row"):
                    question = gr.Textbox(
                        placeholder=app.question_placeholder,
                        show_label=False,
                        max_length=max_question_chars,
                        scale=5,
                        elem_classes="beatle_rag_question",
                    )
                    ask = gr.Button("Ask", variant="primary", scale=1, elem_classes="beatle_rag_ask_button")

            with gr.Column(elem_classes="beatle_rag_examples"):
                gr.Examples(examples=[[example] for example in app.examples], inputs=question, label="You can try one of these")

            gr.Markdown(app.footer, elem_classes="beatle_rag_footer")

        inputs = [question, api_key, chat, turns]
        outputs = [question, chat, turns]
        for trigger in (question.submit, ask.click):
            trigger(respond, inputs, outputs, show_progress="hidden", api_visibility="private")
        chat.clear(lambda: [], None, turns, api_visibility="private")

    return demo
