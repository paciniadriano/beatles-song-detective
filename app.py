"""Entry point of the Hugging Face Space (and of a local run).

    python app.py

Loads everything once at startup: the config, the Qdrant store, the
embedding models and the prompt. Each question then makes its own OpenAI
client from the key the user pasted, or locally from the key in .env when
the box is empty (src/app/chat.py)."""

import os

# Before Gradio is imported: no usage statistics are sent anywhere.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Free Spaces now run only on ZeroGPU hardware, which refuses to start an app
# with no @spaces.GPU function. This app needs no GPU (the LLM is OpenAI's;
# the embedding models run on the CPU), so it declares one empty function
# that is never called, and no GPU time is used. The `spaces` package exists
# only on the Space; a local run skips this.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    @spaces.GPU
    def zero_gpu_placeholder() -> None:
        """Never called: it only lets a ZeroGPU Space start."""

from collections.abc import Iterator  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402

from src.app.chat import ChatContext, LockedSearcher, Message, respond  # noqa: E402
from src.app.startup import prepare_store, runs_on_space  # noqa: E402
from src.app.ui import build_ui, font_face_css  # noqa: E402
from src.config import Config  # noqa: E402
from src.index.collections import collection_name  # noqa: E402
from src.rag.agent import Turn, load_prompt  # noqa: E402
from src.rag.llm import make_client  # noqa: E402
from src.rag.search import open_searcher  # noqa: E402

# The page is designed on white: a browser in dark mode is sent to the light
# version (Gradio reads the theme from this URL parameter).
FORCE_LIGHT_MODE = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get("__theme") !== "light") {
    url.searchParams.set("__theme", "light");
    window.location.replace(url.href);
  }
}
"""

# Inter comes with the app (src/app/fonts, declared by ui.font_face_css):
# nothing is downloaded from Google. Gradio puts quotes around each name,
# which turns keywords like system-ui into font names that do not exist, so
# only real font names go here; style.css sets the full list.
THEME = gr.themes.Default(
    primary_hue="red",
    neutral_hue="gray",
    font=["Inter", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    font_mono=["Consolas", "Menlo", "monospace"],
)


def main() -> None:
    config = Config.load(Path("config.yaml"))
    store = prepare_store(config)
    searcher = open_searcher(config, collection_name(config, config.chunking.active), store)

    # Locally the key in .env answers when the box is empty. Never on the
    # Space, even if a key were set there: every visitor pays with their own.
    fallback_api_key = "" if runs_on_space(config) else os.getenv(config.llm.api_key_env, "")

    ctx = ChatContext(
        searcher=LockedSearcher(searcher),
        settings=config.llm,
        prices=config.model_prices[config.llm.model],
        system_prompt=load_prompt(config.llm.prompt_path),
        make_client=partial(make_client, config.llm.base_url),
        fallback_api_key=fallback_api_key,
    )

    def answer_question(
        question: str, api_key: str, messages: list[Message], turns: list[Turn],
    ) -> Iterator[tuple[str, list[Message], list[Turn]]]:
        yield from respond(question, api_key, messages, turns, ctx)

    demo = build_ui(config.app, config.llm.max_question_chars, answer_question)
    demo.queue(max_size=config.space.queue_max_size, default_concurrency_limit=config.space.concurrency_limit)
    demo.launch(
        theme=THEME,
        css=font_face_css(config.app.font_path),
        css_paths=[config.app.css_path],
        js=FORCE_LIGHT_MODE,
        footer_links=["gradio"],  # no "Use via API": the events are private
    )


if __name__ == "__main__":
    main()
