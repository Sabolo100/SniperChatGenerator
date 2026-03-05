"""
Sniper AI Training Data Generator
===================================
Miért Gradio?
  Gradio = Python-ból azonnal kész webUI + WebSocket queue + live streaming (yield) +
  fájlletöltés. Ezeket Flask/FastAPI-val kézzel kellene felírni. Ha csak CLI kell,
  a generator.py önállóan is hívható — Gradio nem kötelező.

Indítás:  python app.py
"""

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gradio as gr

from converter import get_stats, load_existing, save_results
from generator import MODELS, estimate_cost, generate_one, validate_api_key

# ── Globális stop jelző ─────────────────────────────────────────────────────────
_stop_event = threading.Event()
_LOG_MAXLEN  = 50

_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL    = MODELS[_DEFAULT_PROVIDER][0]


# ── Segéd formázók ─────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
#  PLATFORM / MODELL VÁLASZTÓ — eseménykezelők
# ══════════════════════════════════════════════════════════════════════════════

def on_provider_change(provider: str):
    """
    Platform (openai / anthropic / gemini) váltásakor frissíti a modellek listáját.
    Visszaad egy gr.update()-et, ami lecseréli a Dropdown choices-át és értékét.
    """
    choices = MODELS.get(provider, [])
    if not choices:
        return gr.update(choices=[], value=None)
    return gr.update(choices=choices, value=choices[0])


def on_model_or_count_change(model: str, count: int) -> str:
    """Modell vagy darabszám változásakor frissíti a becslést."""
    if not model:
        return ""
    return _fmt_estimate(model, count)


def on_test_key(provider: str, api_key: str) -> str:
    """API kulcs ellenőrzése a kiválasztott platformon."""
    if not api_key or not api_key.strip():
        return "❌ Adj meg API kulcsot!"
    if not provider:
        return "❌ Válassz platformot!"
    ok, msg = validate_api_key(provider, api_key.strip())
    return f"✅ Kapcsolat OK — {msg}" if ok else f"❌ Hiba: {msg}"


# ══════════════════════════════════════════════════════════════════════════════
#  GENERÁLÁS
# ══════════════════════════════════════════════════════════════════════════════

def start_generation(provider, model, api_key, count, output_dir, resume_mode):
    global _stop_event
    _stop_event.clear()

    _NO_FILES = gr.update(value=None)

    if not provider:
        yield "❌ Válassz platformot!", "", "", "", "", "", _NO_FILES; return
    if not api_key or not api_key.strip():
        yield "❌ API kulcs hiányzik!", "", "", "", "", "", _NO_FILES; return
    if not model:
        yield "❌ Válassz modellt!", "", "", "", "", "", _NO_FILES; return

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

    yield (
        f"🚀 Generálás indul… | `{session_dir}`",
        _fmt_progress(already_done, count, 0, 0, 0.001),
        last_conv_md,
        "\n".join(log_lines),
        json.dumps(get_stats(conversations), indent=2, ensure_ascii=False),
        session_dir,
        _NO_FILES,
    )

    if needed == 0:
        yield (
            "✅ Már megvan mind a kért párbeszéd ebben a session-ben!",
            _fmt_progress(count, count, already_done, 0, 0.001),
            last_conv_md,
            "\n".join(log_lines),
            json.dumps(get_stats(conversations), indent=2, ensure_ascii=False),
            session_dir,
            _NO_FILES,
        )
        return

    for i in range(needed):
        if _stop_event.is_set():
            log_lines.append("[STOP] Felhasználó leállította.")
            break

        total_so_far = already_done + i
        result, err_msg = generate_one(provider, api_key, model)

        if result:
            conversations.append(result)
            success    += 1
            consec_err  = 0
            log_lines.append(f"[OK] #{total_so_far + 1} — {result.get('tema', '?')}")
            last_conv_md = _fmt_last_conv(result)
        else:
            errors     += 1
            consec_err += 1
            log_lines.append(f"[HIBA] #{total_so_far + 1} — {err_msg or ''}")

        if (i + 1) % 10 == 0:
            save_results(conversations, session_dir)

        elapsed = time.time() - start_ts
        yield (
            f"Generálás… | `{session_dir}`",
            _fmt_progress(total_so_far + 1, count, success, errors, elapsed),
            last_conv_md,
            "\n".join(log_lines),
            json.dumps(get_stats(conversations), indent=2, ensure_ascii=False),
            session_dir,
            _NO_FILES,
        )

        if consec_err >= 10:
            log_lines.append("[ABORT] 10 egymás utáni hiba — leállítva.")
            break

    paths   = save_results(conversations, session_dir)
    elapsed = time.time() - start_ts
    log_lines.append(f"[DONE] Mentve: {session_dir}")

    yield (
        f"✅ Kész! **{len(conversations)}** párbeszéd ({elapsed:.0f}s) | ❌ {errors} hiba",
        _fmt_progress(count, count, success, errors, elapsed),
        last_conv_md,
        "\n".join(log_lines),
        json.dumps(get_stats(conversations), indent=2, ensure_ascii=False),
        session_dir,
        [paths["json"], paths["jsonl"], paths["log"]],
    )


def stop_generation():
    _stop_event.set()
    return "⏹️ Leállítás kérve…"


# ══════════════════════════════════════════════════════════════════════════════
#  UI FELÉPÍTÉS
# ══════════════════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="Sniper AI Training Data Generator") as demo:

        gr.Markdown(
            "# 🎯 Sniper AI Training Data Generator\n"
            "Fine-tuning adat generátor — OpenAI · Anthropic · Gemini"
        )

        with gr.Row():

            # ── Bal oldal: konfiguráció ────────────────────────────────────────
            with gr.Column(scale=1):

                # ── 1. Platform ────────────────────────────────────────────────
                provider_dd = gr.Dropdown(
                    choices=list(MODELS.keys()),   # ["openai", "anthropic", "gemini"]
                    value=_DEFAULT_PROVIDER,
                    label="🌐 Platform",
                )

                # ── 2. Modell (a platform alapján frissül) ─────────────────────
                model_dd = gr.Dropdown(
                    choices=MODELS[_DEFAULT_PROVIDER],
                    value=_DEFAULT_MODEL,
                    label="🤖 Modell",
                )

                # ── 3. API kulcs ───────────────────────────────────────────────
                api_key_box = gr.Textbox(
                    type="password",
                    label="🔑 API Kulcs",
                    placeholder="OpenAI: sk-...   Anthropic: sk-ant-...   Gemini: AIza...",
                )

                # ── 4. API kulcs ellenőrzés ────────────────────────────────────
                test_btn = gr.Button("🔍 API Kulcs Ellenőrzése", variant="secondary")
                test_out = gr.Textbox(label="Ellenőrzés eredménye", interactive=False)

                gr.Markdown("---")

                # ── 5. Generálási beállítások ──────────────────────────────────
                count_slider = gr.Slider(
                    minimum=50, maximum=2000, step=50, value=500,
                    label="📊 Generálandó párbeszédek száma",
                )
                output_dir_box = gr.Textbox(
                    value="./training_data",
                    label="💾 Kimeneti mappa (session alkönyvtárak ide kerülnek)",
                )
                resume_cb = gr.Checkbox(
                    value=False,
                    label="▶️ Folytatás — meglévő session betöltése a fenti mappából",
                )
                estimate_md = gr.Markdown(_fmt_estimate(_DEFAULT_MODEL, 500))

            # ── Jobb oldal: futtatás + élő kimenet ───────────────────────────
            with gr.Column(scale=1):

                with gr.Row():
                    start_btn = gr.Button("🚀 Generálás Indítása", variant="primary")
                    stop_btn  = gr.Button("⏹️ Megállítás", variant="stop")

                status_md    = gr.Markdown("_Kész az indításra…_")
                progress_md  = gr.Markdown("")

                session_path = gr.Textbox(
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
                stats_box = gr.Textbox(
                    label="📈 Statisztikák",
                    interactive=False,
                    lines=6,
                )
                files_out = gr.Files(label="📥 Letölthető fájlok")

        # ── Esemény-bekötések ──────────────────────────────────────────────────

        # Platform váltás → modell lista frissítése
        provider_dd.change(
            fn=on_provider_change,
            inputs=provider_dd,
            outputs=model_dd,
        )

        # Modell vagy darabszám változás → becslés frissítése
        model_dd.change(
            fn=on_model_or_count_change,
            inputs=[model_dd, count_slider],
            outputs=estimate_md,
        )
        count_slider.change(
            fn=on_model_or_count_change,
            inputs=[model_dd, count_slider],
            outputs=estimate_md,
        )

        # API kulcs ellenőrzés gomb
        test_btn.click(
            fn=on_test_key,
            inputs=[provider_dd, api_key_box],
            outputs=test_out,
        )

        # Generálás indítása
        start_btn.click(
            fn=start_generation,
            inputs=[provider_dd, model_dd, api_key_box, count_slider,
                    output_dir_box, resume_cb],
            outputs=[status_md, progress_md, last_conv_md,
                     log_box, stats_box, session_path, files_out],
        )

        # Megállítás (bypass queue → azonnali)
        stop_btn.click(fn=stop_generation, inputs=[], outputs=status_md, queue=False)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="127.0.0.1", inbrowser=True)
