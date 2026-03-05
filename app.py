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
def build_ui():
    with gr.Blocks(title="🎯 Sniper AI Training Data Generator") as demo:

        gr.Markdown(
            "# 🎯 Sniper AI Training Data Generator\n"
            "Generálj fine-tuning adatot a sniper kiképző AI modellhez"
        )

        with gr.Row():
            # ── Left: configuration ───────────────────────────────────────────
            with gr.Column(scale=1):
                provider_radio = gr.Radio(
                    choices=["openai", "anthropic", "gemini"],
                    value="openai",
                    label="🤖 AI Provider",
                )
                model_dropdown = gr.Dropdown(
                    choices=MODELS["openai"],
                    value=MODELS["openai"][0],
                    label="📦 Modell",
                )
                api_key_box = gr.Textbox(
                    type="password",
                    label="🔑 API Kulcs",
                    placeholder="sk-... vagy hasonló",
                )
                test_btn    = gr.Button("🔍 API Kulcs Tesztelése")
                test_result = gr.Textbox(label="API teszt eredménye", interactive=False)

                count_slider = gr.Slider(
                    minimum=50, maximum=2000, step=50, value=500,
                    label="📊 Generálandó párbeszédek száma",
                )
                output_dir_box = gr.Textbox(
                    value="./training_data",
                    label="💾 Alap kimeneti mappa (minden session külön almappába kerül)",
                )
                resume_checkbox = gr.Checkbox(
                    value=False,
                    label="▶️ Folytatás — meglévő session folytatása a fenti mappából",
                )
                estimate_md = gr.Markdown(_fmt_estimate(MODELS["openai"][0], 500))

            # ── Right: run & live output ──────────────────────────────────────
            with gr.Column(scale=1):
                with gr.Row():
                    start_btn = gr.Button("🚀 Generálás Indítása", variant="primary")
                    stop_btn  = gr.Button("⏹️ Megállítás", variant="stop")

                status_md = gr.Markdown("_Kész az indításra…_")

                progress_md = gr.Markdown("")

                session_path_box = gr.Textbox(
                    label="📁 Aktuális session mappa",
                    interactive=False,
                    placeholder="Indítás után jelenik meg…",
                )

                last_conv_md = gr.Markdown(
                    "_Az utolsó generált párbeszéd itt jelenik meg…_",
                    label="💬 Utolsó generált párbeszéd",
                )

                log_box = gr.Textbox(
                    label="📋 Napló (utolsó 50 sor)",
                    lines=8,
                    interactive=False,
                    autoscroll=True,
                )

                stats_json = gr.JSON(label="📈 Statisztikák")
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
    build_ui().launch(server_name="0.0.0.0", inbrowser=True)
