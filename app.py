"""Gradio 6.x Web Frontend for ZentroMart VoiceOps 2026 Capstone.

Features:
- Instant static startup greeting in the chat window (zero API calls on load)
- Unified Gemini-style bottom prompt bar (Text input + inline Mic & Send buttons)
- Pop-up Voice Recorder card triggered by clicking the inline Mic button
- Full compatibility with Gradio 6.30.0 native messages format

Launch locally with:
    python app.py
"""
import os
import time
import inspect
from dotenv import load_dotenv
import gradio as gr

import logging

from src.rag import build_index
from src.speech import transcribe_audio, synthesize_speech
from src.agent import ZentroMartAgent

try:
    from tools.tools import lookup_order, create_ticket
except ImportError:
    lookup_order = None
    create_ticket = None

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Load GEMINI_API_KEY and GNANI_API_KEY from .env
load_dotenv()

# Ensure output folder exists for live web audio replies
WEB_OUTPUT_DIR = os.path.join("sample_outputs", "web_replies")
os.makedirs(WEB_OUTPUT_DIR, exist_ok=True)

print("1/2 Loading FAISS Vector Index from ./kb ...")
retriever = build_index("./kb")

print("2/2 Initializing ZentroMart Gemini Agent (gemini-3.5-flash-lite) ...")
try:
    agent = ZentroMartAgent(retriever=retriever)
except Exception as init_err:
    print(f"[Startup Warning] Agent initialization note: {init_err}")
    agent = None

# Static startup greeting rendered immediately without any LLM/TTS API call
WELCOME_MESSAGE = (
    "👋 Hi! I'm ZentroMart's AI support assistant. How can I help you today?\n\n"
    "You can type your question below or click the **🎙️ microphone icon** to speak in "
    "English, Hindi, Hinglish, or any of the languages given below."
)
INITIAL_CHAT_HISTORY = [{"role": "assistant", "content": WELCOME_MESSAGE}]


def _ensure_agent():
    """Ensure the Gemini agent is initialized once GEMINI_API_KEY is available."""
    global agent
    if agent is None:
        load_dotenv(override=True)
        agent = ZentroMartAgent(retriever=retriever)
    return agent


def _append_chat_turn(chat_history, user_msg: str, bot_msg: str):
    """Append a user/assistant turn in Gradio 6.x messages format (list of role/content dicts)."""
    normalized = []
    for msg in (chat_history or []):
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            normalized.append({"role": msg["role"], "content": msg["content"]})
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            normalized.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            normalized.append({"role": "user", "content": str(msg[0])})
            normalized.append({"role": "assistant", "content": str(msg[1])})

    normalized.append({"role": "user", "content": str(user_msg)})
    normalized.append({"role": "assistant", "content": str(bot_msg)})
    return normalized


def _format_sources(sources: list) -> str:
    if not sources:
        return "No knowledge base chunks retrieved for this turn."
    formatted = []
    for idx, src in enumerate(sources, 1):
        formatted.append(f"--- Chunk {idx} ---\n{src}")
    return "\n\n".join(formatted)


def toggle_voice_popup(is_open: bool):
    """Toggle the visibility of the pop-up voice recorder card when Mic icon is clicked."""
    new_state = not bool(is_open)
    return gr.update(visible=new_state), new_state


def close_voice_popup():
    """Explicitly close the pop-up voice recorder card."""
    return gr.update(visible=False), False


def handle_voice_turn(audio_path, chat_history, lang_code, voice_name):
    """Orchestrate a complete voice turn: Gnani STT -> Gemini RAG Agent -> Gnani TTS, then close popup."""
    if not audio_path:
        return (
            chat_history or INITIAL_CHAT_HISTORY,
            None,
            "No audio detected. Please click Record or upload a .wav file.",
            "No sources retrieved.",
            gr.update(visible=True),
            True,
            None,
        )

    # 1. Listen: Transcribe customer audio via Gnani STT
    heard_text = transcribe_audio(audio_path, lang=lang_code)
    if not heard_text or not heard_text.strip():
        fallback_reply = "Sorry, I didn't catch that. Could you please say it again?"
        out_wav = os.path.join(WEB_OUTPUT_DIR, f"reply_{int(time.time())}.wav")
        audio_out = synthesize_speech(fallback_reply, out_wav, lang=lang_code, voice=voice_name)
        chat_history = _append_chat_turn(chat_history, "🎙️ [Unclear Audio]", fallback_reply)
        return (
            chat_history,
            audio_out,
            "[Empty / Unclear Audio]",
            "STT returned an empty transcript; triggered polite recovery prompt.",
            gr.update(visible=False),
            False,
            None,
        )

    # 2. Think: Query Gemini 3.5 Flash Lite with RAG context & multi-turn memory
    try:
        active_agent = _ensure_agent()
        reply_text, sources = active_agent.chat(heard_text)
    except Exception as e:
        reply_text = f"Agent configuration error: {str(e)}"
        sources = []

    # 3. Speak: Synthesize spoken reply via Gnani TTS (timbre-v2.5)
    out_wav = os.path.join(WEB_OUTPUT_DIR, f"reply_{int(time.time())}.wav")
    audio_out = synthesize_speech(reply_text, out_wav, lang=lang_code, voice=voice_name)

    # 4. Log & Update UI state (auto-hide voice popup and reset audio input)
    log_path = os.path.join(WEB_OUTPUT_DIR, f"log_{int(time.time())}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            f"Customer STT: {heard_text}\n\nAgent Reply: {reply_text}\n\nSources Used: {sources}\n"
        )

    chat_history = _append_chat_turn(chat_history, f"🎙️ {heard_text}", reply_text)
    return (
        chat_history,
        audio_out,
        heard_text,
        _format_sources(sources),
        gr.update(visible=False),
        False,
        None,
    )


def handle_text_turn(user_text, chat_history, lang_code, voice_name):
    """Process a typed customer message with RAG & TTS playback."""
    if not user_text or not user_text.strip():
        return (
            chat_history or INITIAL_CHAT_HISTORY,
            None,
            "",
            "Please enter a question.",
            "",
        )

    try:
        active_agent = _ensure_agent()
        reply_text, sources = active_agent.chat(user_text.strip())
    except Exception as e:
        reply_text = f"Agent configuration error: {str(e)}"
        sources = []

    out_wav = os.path.join(WEB_OUTPUT_DIR, f"reply_{int(time.time())}.wav")
    audio_out = synthesize_speech(reply_text, out_wav, lang=lang_code, voice=voice_name)

    chat_history = _append_chat_turn(chat_history, user_text.strip(), reply_text)
    return chat_history, audio_out, user_text.strip(), _format_sources(sources), ""


def reset_conversation():
    """Reset the Gemini agent's multi-turn history and restore the initial greeting bubble."""
    if agent is not None:
        agent.reset_history()
    return (
        list(INITIAL_CHAT_HISTORY),
        None,
        None,
        "",
        "Session reset. Ready for a new customer call.",
        gr.update(visible=False),
        False,
    )


def test_order_lookup_ui(order_id, verification_val):
    """Direct UI helper to inspect ZentroMart's identity-checked lookup_order tool."""
    if not lookup_order:
        return "tools.py not found in project root."
    res = lookup_order(order_id or "", verification_val or None)
    return str(res)


CUSTOM_CSS = """
.zentro-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #0f766e 100%);
    color: #ffffff;
    padding: 16px 24px;
    border-radius: 14px;
    margin-bottom: 12px;
    box-shadow: 0 8px 20px -5px rgba(30, 27, 75, 0.25);
}
.zentro-header h1 {
    margin: 0 0 4px 0;
    font-size: 1.45rem;
    font-weight: 700;
    color: #ffffff !important;
}
.zentro-header p {
    margin: 0 0 8px 0;
    opacity: 0.92;
    font-size: 0.9rem;
    color: #e0e7ff !important;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.zentro-badge {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #f8fafc;
}
/* Gemini-style unified prompt pill bar */
.composer-pill-bar {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 999px !important;
    padding: 4px 10px 4px 18px !important;
    align-items: center !important;
    gap: 6px !important;
    margin-top: 6px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}
.composer-textbox textarea, .composer-textbox input {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 0.98rem !important;
}
.composer-icon-btn {
    border-radius: 999px !important;
    min-width: 44px !important;
    max-width: 48px !important;
    height: 42px !important;
    padding: 0 !important;
    font-size: 1.15rem !important;
}
/* Floating Pop-up Voice Recorder Drawer */
.voice-popup-card {
    border: 1px solid #6366f1 !important;
    background: rgba(15, 23, 42, 0.95) !important;
    border-radius: 16px !important;
    padding: 14px !important;
    margin-top: 8px !important;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35) !important;
}
footer {visibility: hidden !important;}
"""


def build_ui():
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal")
    blocks_kwargs = {"title": "ZentroMart VoiceOps AI Support"}
    blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    if "theme" in blocks_params:
        blocks_kwargs["theme"] = theme
    if "css" in blocks_params:
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        voice_popup_state = gr.State(False)

        gr.HTML(
            """
            <div class="zentro-header">
                <h1>🛒 ZentroMart VoiceOps Support Agent</h1>
                <p>Grounded Multi-Turn Customer Support with Gnani Speech Models &amp; Google Gemini</p>
                <div class="badge-row">
                    <span class="zentro-badge">🧠 LLM: Gemini 3.5 Flash Lite</span>
                    <span class="zentro-badge">🎙️ STT &amp; TTS: Gnani Vachana (timbre-v2.5)</span>
                    <span class="zentro-badge">📚 RAG: FAISS + MiniLM-L6-v2 (20 Active Docs)</span>
                    <span class="zentro-badge">📅 Reference Date: 20 Oct 2026</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Interactive Chat Window + Unified Prompt & Mic Bar
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    value=list(INITIAL_CHAT_HISTORY),
                    label="Live Customer Support Conversation",
                    height=430,
                )

                # Gemini-style Unified Input Bar (Text Input + Inline Mic Pop-up Button + Send Button)
                with gr.Row(elem_classes="composer-pill-bar"):
                    text_input = gr.Textbox(
                        placeholder="Enter a prompt here (or click 🎙️ to speak)...",
                        show_label=False,
                        container=False,
                        lines=1,
                        max_lines=3,
                        scale=9,
                        elem_classes="composer-textbox",
                    )
                    mic_toggle_btn = gr.Button(
                        "🎙️",
                        variant="secondary",
                        scale=1,
                        min_width=44,
                        elem_classes="composer-icon-btn",
                    )
                    submit_text_btn = gr.Button(
                        "➤",
                        variant="primary",
                        scale=1,
                        min_width=44,
                        elem_classes="composer-icon-btn",
                    )

                # Pop-Up Voice Recorder Modal (Appears only when customer clicks 🎙️)
                with gr.Group(visible=False, elem_classes="voice-popup-card") as voice_popup:
                    gr.Markdown("##### 🎙️ Voice Input Recorder — Record from Microphone or Upload `.wav`")
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Customer Voice Clip",
                    )
                    with gr.Row():
                        submit_voice_btn = gr.Button(
                            "🔊 Send Voice Message (STT → Gemini → TTS)",
                            variant="primary",
                            scale=3,
                        )
                        close_popup_btn = gr.Button(
                            "✖ Close",
                            variant="secondary",
                            scale=1,
                        )

                # Compact Voice Settings & Reset Toolbar
                with gr.Row():
                    lang_dropdown = gr.Dropdown(
                        choices=[
                            "en-IN",
                            "hi-IN",
                            "gu-IN",
                            "mr-IN",
                            "ta-IN",
                            "te-IN",
                            "kn-IN",
                            "bn-IN",
                            "ml-IN",
                            "pa-IN",
                        ],
                        value="en-IN",
                        label="🌐 Language (Gnani STT/TTS)",
                        scale=2,
                    )
                    voice_dropdown = gr.Dropdown(
                        choices=["Kaveri", "Trupti", "Pranav", "Shlok"],
                        value="Kaveri",
                        label="🗣️ Agent Voice (timbre-v2.5)",
                        scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ New Call / Clear",
                        variant="secondary",
                        scale=1,
                    )

            # Right Column: Spoken Reply, Transcript & RAG Grounding Inspector
            with gr.Column(scale=5):
                audio_output = gr.Audio(
                    label="🔊 ZentroMart Agent Spoken Reply (Gnani TTS)",
                    type="filepath",
                    autoplay=True,
                )
                stt_transcript_box = gr.Textbox(
                    label="📝 Customer Transcript (Gnani STT Output)",
                    placeholder="Transcribed speech or typed prompt will appear here...",
                    interactive=False,
                    lines=2,
                )
                rag_sources_box = gr.Textbox(
                    label="📚 Retrieved Knowledge Base Evidence (FAISS Top-K Chunks)",
                    placeholder="Supporting policy chunks with [SECTION-ID] will appear here...",
                    interactive=False,
                    lines=9,
                )

                with gr.Accordion("🛠️ ZentroMart Tool Inspector (lookup_order & create_ticket)", open=False):
                    gr.Markdown(
                        "Test identity-checked order lookups (`4-digit phone suffix` or `6-digit pincode`) "
                        "and human escalation tickets directly against `data/*.csv`."
                    )
                    with gr.Row():
                        tool_oid = gr.Textbox(label="Order ID", placeholder="e.g., 69520867")
                        tool_verify = gr.Textbox(
                            label="Phone Last 4 / Pincode", placeholder="e.g., 4533"
                        )
                    tool_lookup_btn = gr.Button("🔍 Run lookup_order()")
                    tool_output_box = gr.Textbox(
                        label="Tool Output", interactive=False, lines=4
                    )
                    tool_lookup_btn.click(
                        fn=test_order_lookup_ui,
                        inputs=[tool_oid, tool_verify],
                        outputs=[tool_output_box],
                    )

        # Toggle Pop-up Voice Recorder when inline Mic icon is clicked
        mic_toggle_btn.click(
            fn=toggle_voice_popup,
            inputs=[voice_popup_state],
            outputs=[voice_popup, voice_popup_state],
        )
        close_popup_btn.click(
            fn=close_voice_popup,
            inputs=[],
            outputs=[voice_popup, voice_popup_state],
        )

        # Wire up Voice Turn events (processes voice and auto-closes popup)
        submit_voice_btn.click(
            fn=handle_voice_turn,
            inputs=[audio_input, chatbot, lang_dropdown, voice_dropdown],
            outputs=[
                chatbot,
                audio_output,
                stt_transcript_box,
                rag_sources_box,
                voice_popup,
                voice_popup_state,
                audio_input,
            ],
        )

        # Wire up Text Turn events (Enter key or ➤ Send button)
        submit_text_btn.click(
            fn=handle_text_turn,
            inputs=[text_input, chatbot, lang_dropdown, voice_dropdown],
            outputs=[chatbot, audio_output, stt_transcript_box, rag_sources_box, text_input],
        )
        text_input.submit(
            fn=handle_text_turn,
            inputs=[text_input, chatbot, lang_dropdown, voice_dropdown],
            outputs=[chatbot, audio_output, stt_transcript_box, rag_sources_box, text_input],
        )

        # Wire up Clear / Reset Session button (restores initial greeting)
        clear_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=[
                chatbot,
                audio_input,
                audio_output,
                stt_transcript_box,
                rag_sources_box,
                voice_popup,
                voice_popup_state,
            ],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    launch_params = inspect.signature(ui.launch).parameters
    launch_kwargs = {"server_name": "127.0.0.1", "server_port": 7860, "share": True}
    if "theme" in launch_params:
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal")
    if "css" in launch_params:
        launch_kwargs["css"] = CUSTOM_CSS
    ui.launch(**launch_kwargs)