"""
Sniper AI Training Data Generator
Gradio-based UI for generating fine-tuning data via OpenAI / Anthropic / Gemini.
Run:  python app.py
"""

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gradio as gr

from converter import get_stats, load_existing, save_results
from generator import MODELS, estimate_cost, generate_one, validate_api_key

# ── Global stop flag ───────────────────────────────────────────────────────────
_stop_event = threading.Event()
_LOG_MAXLEN = 50


def _fmt_estimate(model: str, count: int) -> str:
    cost    = estimate_cost(model, count)
    minutes = round(count * 2 / 60, 1)
    return f"⏱️ Becsült idő: ~{minutes} perc | 💰 Becsült költség: ~${cost:.4f} USD"


def _fmt_last_conv(result: dict) -> str:
    tema = result.get("tema", "?")
    user = result.get("user_message", "")
    asst = result.get("assistant_message", "")
    return (
        f"**📌 Téma:** {tema}\n\n"
        f"**🪖 Katona:**\n\n{user}\n\n"
        f"**🎯 Kovács Őrmester:**\n\n{asst}"
    )


def _fmt_progress(done: int, total: int, success: int, errors: int,
                  elapsed: float) -> str:
    pct   = int(done / total * 100) if total else 0
    bar   = "█" * (pct // 5) + "░" * (20 - pct // 5)
    speed = success / elapsed * 60 if elapsed > 0 else 0
    return (
        f"`[{bar}] {pct}% — {done}/{total}`\n\n"
        f"✅ **{success}** sikeres · ❌ **{errors}** hiba · "
        f"⚡ **{speed:.1f}** db/perc"
    )


# ── UI callbacks ───────────────────────────────────────────────────────────────
def update_models(provider: str):
    choices = MODELS.get(provider, [])
    return gr.update(choices=choices, value=choices[0] if choices else None)


def update_estimate(model: str, count: int) -> str:
    return _fmt_estimate(model, count) if model else ""


def test_api_key(provider: str, api_key: str) -> str:
    if not api_key or not api_key.strip():
        return "❌ API kulcs nem adott meg!"
    if not provider:
        return "❌ Válassz providert!"
    ok, msg = validate_api_key(provider, api_key)
    return f"✅ Kapcsolat sikeres: {msg}" if ok else f"❌ Hiba: {msg}"


# ── Generation ─────────────────────────────────────────────────────────────────
def start_generation(provider, model, api_key, count, output_dir, resume_mode):
    """
    Generator — yields (status_md, progress_md, last_conv_md, log_str, stats_dict, session_str, files)
    every time a conversation is generated.
    NOTE: no gr.Progress() — its separate SSE stream conflicts with yielded outputs.
    """
    global _stop_event
    _stop_event.clear()

    _NO_FILES = gr.update(value=None)

    # Validation
    if not provider:
        yield "❌ Válassz providert!", "", "", "", {}, "", _NO_FILES
        return
    if not api_key or not api_key.strip():
        yield "❌ API kulcs nem adott meg!", "", "", "", {}, "", _NO_FILES
        return
    if not model:
        yield "❌ Válassz modellt!", "", "", "", {}, "", _NO_FILES
        return

    # Session folder
    base = Path(output_dir)
    if resume_mode:
        session_dir   = str(base)
        conversations = load_existing(session_dir)
    else:
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir   = str(base / f"session_{ts}")
        conversations = []

    already_done   = len(conversations)
    log_lines: deque = deque(maxlen=_LOG_MAXLEN)
    last_conv_md   = "_Várakozás az első generálásra…_"

    log_lines.append(f"[INFO] Mappa: {session_dir}")
    if already_done:
        log_lines.append(f"[INFO] Folytatás: {already_done} meglévő adat betöltve.")

    needed     = max(0, count - already_done)
    success    = 0
    errors     = 0
    consec_err = 0
    start_ts   = time.time()

    # Initial update — show session path immediately
    yield (
        f"🚀 Generálás indul… | `{session_dir}`",
        _fmt_progress(already_done, count, 0, 0, 0.001),
        last_conv_md,
        "\n".join(log_lines),
        get_stats(conversations),
        session_dir,
        _NO_FILES,
    )

    if needed == 0:
        yield (
            "✅ Már megvan mind a kért párbeszéd ebben a session-ben!",
            _fmt_progress(count, count, already_done, 0, 0.001),
            last_conv_md,
            "\n".join(log_lines),
            get_stats(conversations),
            session_dir,
            _NO_FILES,
        )
        return

    for i in range(needed):
        if _stop_event.is_set():
            log_lines.append("[STOP] Felhasználó leállította a generálást.")
            break

        total_so_far = already_done + i
        result       = generate_one(provider, api_key, model)

        if result:
            conversations.append(result)
            success      += 1
            consec_err    = 0
            tema           = result.get("tema", "?")
            log_lines.append(f"[OK] #{total_so_far + 1} — {tema}")
            last_conv_md  = _fmt_last_conv(result)
        else:
            errors     += 1
            consec_err += 1
            log_lines.append(f"[HIBA] #{total_so_far + 1} — generálás sikertelen")
            # last_conv_md intentionally NOT reset — keep last successful one visible

        # Save every 10
        if (i + 1) % 10 == 0:
            save_results(conversations, session_dir)

        elapsed     = time.time() - start_ts
        status_text = f"Generálás folyamatban… | `{session_dir}`"

        yield (
            status_text,
            _fmt_progress(total_so_far + 1, count, success, errors, elapsed),
            last_conv_md,
            "\n".join(log_lines),
            get_stats(conversations),
            session_dir,
            _NO_FILES,
        )

        if consec_err >= 10:
            log_lines.append("[ABORT] 10 egymás utáni hiba — generálás leállítva.")
            break

    # Final save
    paths   = save_results(conversations, session_dir)
    elapsed = time.time() - start_ts

    final_status = (
        f"✅ Kész! **{len(conversations)}** párbeszéd mentve "
        f"({elapsed:.0f}s alatt) | ❌ {errors} hiba"
    )
    log_lines.append(f"[DONE] Fájlok mentve: {session_dir}")

    yield (
        final_status,
        _fmt_progress(count, count, success, errors, elapsed),
        last_conv_md,
        "\n".join(log_lines),
        get_stats(conversations),
        session_dir,
        [paths["json"], paths["jsonl"], paths["log"]],
    )


def stop_generation():
    _stop_event.set()
    return "⏹️ Leállítás kérve…"


# ── UI ─────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Vibráns, színes téma */
.gradio-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    min-height: 100vh;
}

/* Főcím stílus */
.gradio-container h1 {
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3, #54a0ff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2.5rem !important;
    text-align: center !important;
    text-shadow: 0 0 30px rgba(255, 107, 107, 0.5);
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 5px #ff6b6b); }
    to { filter: drop-shadow(0 0 20px #54a0ff); }
}

/* Panel és box stílusok */
.gr-box, .gr-panel, .gr-form {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}

/* Input mezők */
input, textarea, .gr-input {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 2px solid #48dbfb !important;
    color: #fff !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus {
    border-color: #ff6b6b !important;
    box-shadow: 0 0 20px rgba(255, 107, 107, 0.4) !important;
}

/* Dropdown */
.gr-dropdown {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 2px solid #feca57 !important;
    border-radius: 12px !important;
}

/* Slider */
input[type="range"] {
    accent-color: #ff9ff3 !important;
}

.gr-slider input {
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb) !important;
}

/* Gombok */
.gr-button {
    border-radius: 12px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 50%, #5f27cd 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(84, 160, 255, 0.4) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 30px rgba(84, 160, 255, 0.6) !important;
}

.gr-button-stop, button[variant="stop"] {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(255, 107, 107, 0.4) !important;
}

.gr-button-stop:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 30px rgba(255, 107, 107, 0.6) !important;
}

/* Másodlagos gomb */
.gr-button-secondary {
    background: linear-gradient(135deg, #feca57 0%, #ff9f43 100%) !important;
    border: none !important;
    color: #1a1a2e !important;
}

/* Radio és Checkbox */
.gr-radio label, .gr-checkbox label {
    color: #fff !important;
}

.gr-radio input:checked + span::before {
    background: linear-gradient(135deg, #ff6b6b, #ff9ff3) !important;
}

/* Címkék */
label {
    color: #48dbfb !important;
    font-weight: 600 !important;
    text-shadow: 0 0 10px rgba(72, 219, 251, 0.3) !important;
}

/* Markdown szövegek */
.markdown-text, .gr-markdown {
    color: #e8e8e8 !important;
}

.gr-markdown strong {
    color: #feca57 !important;
}

.gr-markdown code {
    background: rgba(255, 107, 107, 0.2) !important;
    color: #ff6b6b !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
}

/* JSON megjelenítő */
.gr-json {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 2px solid #54a0ff !important;
    border-radius: 12px !important;
}

/* Fájl lista */
.gr-file {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px dashed #ff9ff3 !important;
    border-radius: 12px !important;
}

/* Row és Column */
.gr-row {
    gap: 24px !important;
}

.gr-column {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Alcímek és szövegek */
p, span {
    color: #c8d6e5 !important;
}

/* Scrollbar stílus */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #ff6b6b, #54a0ff);
    border-radius: 4px;
}

/* Textbox stílusok */
.gr-textbox textarea {
    background: rgba(0, 0, 0, 0.3) !important;
    color: #1dd1a1 !important;
    font-family: 'Fira Code', monospace !important;
    border: 2px solid #1dd1a1 !important;
}

/* Progress/status text */
.status-text {
    color: #48dbfb !important;
}

/* Hover effektek az input elemekre */
.gr-box:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Tooltip stílus */
.gr-tooltip {
    background: #1a1a2e !important;
    border: 1px solid #54a0ff !important;
    color: #fff !important;
}
"""

def build_ui():
    with gr.Blocks(title="🎯 Sniper AI Training Data Generator", css=CUSTOM_CSS, theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
    )) as demo:

        gr.Markdown(
            "# 🎯 Sniper AI Training Data Generator\n\n"
            "### ✨ Generálj fine-tuning adatot a sniper kiképző AI modellhez ✨"
        )

        with gr.Row():
            # ── Left: configuration ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Konfiguráció")
                provider_radio = gr.Radio(
                    choices=["openai", "anthropic", "gemini"],
                    value="openai",
                    label="🤖 AI Provider",
                    info="Válaszd ki a szolgáltatót",
                )
                model_dropdown = gr.Dropdown(
                    choices=MODELS["openai"],
                    value=MODELS["openai"][0],
                    label="📦 Modell",
                    info="A generáláshoz használt modell",
                )
                api_key_box = gr.Textbox(
                    type="password",
                    label="🔑 API Kulcs",
                    placeholder="sk-... vagy hasonló",
                    info="A kiválasztott provider API kulcsa",
                )
                test_btn    = gr.Button("🔍 API Kulcs Tesztelése", variant="secondary")
                test_result = gr.Textbox(label="🧪 API teszt eredménye", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Beállítások")
                count_slider = gr.Slider(
                    minimum=50, maximum=2000, step=50, value=500,
                    label="📊 Generálandó párbeszédek száma",
                    info="Több = hosszabb idő, több adat",
                )
                output_dir_box = gr.Textbox(
                    value="./training_data",
                    label="💾 Kimeneti mappa",
                    info="Minden session külön almappába kerül",
                )
                resume_checkbox = gr.Checkbox(
                    value=False,
                    label="▶️ Folytatás — meglévő session folytatása",
                )
                gr.Markdown("---")
                estimate_md = gr.Markdown(_fmt_estimate(MODELS["openai"][0], 500))

            # ── Right: run & live output ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Vezérlés")
                with gr.Row():
                    start_btn = gr.Button("🚀 Generálás Indítása", variant="primary", size="lg")
                    stop_btn  = gr.Button("⏹️ Megállítás", variant="stop", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 📊 Állapot")
                status_md = gr.Markdown("_✨ Kész az indításra…_")
                progress_md = gr.Markdown("")

                session_path_box = gr.Textbox(
                    label="📁 Aktuális session mappa",
                    interactive=False,
                    placeholder="🔄 Indítás után jelenik meg…",
                )

                gr.Markdown("---")
                gr.Markdown("### 💬 Utolsó Generált Párbeszéd")
                last_conv_md = gr.Markdown(
                    "_🎯 Az utolsó generált párbeszéd itt jelenik meg…_",
                )

                gr.Markdown("---")
                gr.Markdown("### 📋 Napló")
                log_box = gr.Textbox(
                    label="📜 Események (utolsó 50 sor)",
                    lines=8,
                    interactive=False,
                    autoscroll=True,
                )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=1):
                        stats_json = gr.JSON(label="📈 Statisztikák")
                    with gr.Column(scale=1):
                        files_out  = gr.Files(label="📥 Letölthető fájlok")

        # ── Wiring ─────────────────────────────────────────────────────────────
        provider_radio.change(update_models, provider_radio, model_dropdown)

        for comp in [model_dropdown, count_slider]:
            comp.change(update_estimate, [model_dropdown, count_slider], estimate_md)

        test_btn.click(test_api_key, [provider_radio, api_key_box], test_result)

        start_btn.click(
            fn=start_generation,
            inputs=[
                provider_radio, model_dropdown, api_key_box,
                count_slider, output_dir_box, resume_checkbox,
            ],
            outputs=[
                status_md, progress_md, last_conv_md,
                log_box, stats_json, session_path_box, files_out,
            ],
        )

        stop_btn.click(stop_generation, [], status_md)

    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
