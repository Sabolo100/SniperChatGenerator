# Tanulságok — SniperChatGenerator debug session

## A hiba

**Tünet:** Platform váltáskor (OpenAI / Anthropic / Gemini) mindig Claude modellek
maradtak a dropdownban. Egyetlen Gradio event handler sem futott le.

**Hibaüzenet a böngészőben:** `No API found`

---

## A valódi ok (egyetlen gyökér)

```
gr.Files output komponens JSON sémája
  → "additionalProperties": false   ← ez egy boolean érték
    → gradio_client _json_schema_to_python_type() megpróbálja:
        if "const" in schema        ← TypeError: bool nem iterable
          → /api/info endpoint crash (500)
            → Gradio JS frontend: "No API found"
              → ÖSSZES event handler csendben meghal
                → provider_dd.change() soha nem hívódik meg
                  → a dropdown örökre az induláskori értéken marad
```

**A hiba tehát NEM a platform/modell logikában volt** — az teljesen helyes volt.
A hiba a `gr.Files` komponens sémájában volt, ami az egész eseményrendszert bénította.

---

## A fix

`gr.Files` → `gr.Textbox` (a mentett fájlok elérési útjait szövegként mutatja).

`gr.Textbox` JSON sémája triviális, nincs `additionalProperties`, nincs crash.

```python
# ELŐTTE — crasheli az /api/info endpointot:
files_out = gr.Files(label="Letölthető fájlok")

# UTÁNA — egyszerű, biztonságos:
files_out = gr.Textbox(label="Mentett fájlok elérési útjai", interactive=False, lines=3)
```

---

## Amit útközben próbáltunk (és miért nem segített)

| Próba | Miért nem volt elég |
|---|---|
| `gr.JSON` → `gr.Textbox` | Helyes irány, de `gr.Files` még maradt |
| `queue=False` eltávolítása az event handlerekről | Nem ez okozta a bajt |
| Gradio 4.44.1 → 4.28.3 downgrade | `gr.Files` schema bug mindkét verzióban él |
| Platform/modell választó nulláról újraírva | A logika mindig helyes volt |

---

## Gradio-val kapcsolatos tanulságok

### Mikor ne használj `gr.Files`-t outputként
Ha a Gradio verziód `gradio_client`-je tartalmazza ezt a bugot, akkor `gr.Files`
output crasheli az `/api/info` endpointot. Helyette: `gr.Textbox` az elérési utakkal.

### Az `/api/info` endpoint kritikus
Ez az első dolog amit a Gradio JS frontend betöltéskor lekér. Ha ez 500-at dob,
**az összes** `.change()`, `.click()`, `.submit()` esemény csendben meghal —
semmilyen Python callback nem fut le. A UI látszólag működik (gombok kattinthatók),
de a háttér nem reagál semmire.

### Diagnosztika
Ha Gradio eventok nem futnak le, először ezt ellenőrizd:
```bash
curl http://127.0.0.1:7860/api/info
# Ha 500-at kapsz vagy JSON parse hibát → schema bug valamelyik komponensben
```

### Miért Gradio és mikor nem kell
- **Kell:** Python-ból azonnal kész webUI, live streaming (`yield`), WebSocket queue
- **Nem kell:** Ha csak CLI — a `generator.py` önállóan is hívható
- **Alternatíva:** Flask + HTML/JS (több munka, de teljes kontroll a sémák felett)

---

## A működő megoldás komponensei

```python
# Egyszerű, schema-biztonságos komponensek:
provider_dd = gr.Dropdown(choices=list(MODELS.keys()), value="openai")
model_dd    = gr.Dropdown(choices=MODELS["openai"], value=MODELS["openai"][0])
api_key_box = gr.Textbox(type="password")
files_out   = gr.Textbox(interactive=False, lines=3)   # ← nem gr.Files!

# Esemény-bekötés — explicit, névvel ellátott függvények, nem lambda:
provider_dd.change(fn=on_provider_change, inputs=provider_dd, outputs=model_dd)
```
