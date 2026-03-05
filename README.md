# 🎯 Sniper AI Training Data Generator

Python alapú, helyi gépen futó alkalmazás, amely sniper kiképző AI fine-tuning training adatot generál
OpenAI / Anthropic / Google Gemini API-k segítségével.
A kimenet fine-tuning kész JSONL formátumban kerül mentésre.

---

## Telepítés

```bash
pip install -r requirements.txt
```

## Indítás

```bash
python app.py
```

A Gradio UI automatikusan megnyílik a böngészőben: **http://localhost:7860**

---

## API kulcsok

| Provider  | Link |
|-----------|------|
| OpenAI    | https://platform.openai.com/api-keys |
| Anthropic | https://console.anthropic.com/settings/keys |
| Gemini    | https://aistudio.google.com/app/apikey |

---

## Ajánlott modellek és árak (becsült, 1 000 tokenre USD)

| Modell                      | Ár / 1K token | Ajánlás |
|-----------------------------|---------------|---------|
| gpt-4o-mini                 | $0.00015      | Gyors, olcsó |
| gpt-4o                      | $0.005        | Kiváló minőség |
| gpt-4-turbo                 | $0.01         | Legrégebb, stabil |
| claude-haiku-4-5-20251001   | $0.00025      | Legolcsóbb Anthropic |
| claude-sonnet-4-6           | $0.003        | Jó minőség/ár arány |
| claude-opus-4-6             | $0.015        | Legjobb Anthropic |
| gemini-1.5-flash            | $0.000075     | Legolcsóbb összességében |
| gemini-1.5-pro              | $0.00125      | Erős Gemini |
| gemini-2.0-flash            | $0.0001       | Újgenerációs, gyors |

---

## Kimenet leírása

A generálás három fájlt ment a megadott mappába:

### `conversations.json`
Teljes, human-readable lista minden generált párbeszéddel:
```json
[
  {
    "tema": "400m keresztszél korrekció",
    "user_message": "Őrmester, 400m-re lőttem, 5 m/s szél 3 óráról, balra tévesztettem.",
    "assistant_message": "1.5 MOA jobb korrekció. Állítsd be és ismételd.",
    "generalva": "2024-01-15T10:23:45+00:00"
  }
]
```

### `train.jsonl`
Fine-tuning kész formátum — minden sor egy önálló JSON:
```json
{"messages": [
  {"role": "system", "content": "Te Kovács Őrmester vagy..."},
  {"role": "user", "content": "Katona kérdése..."},
  {"role": "assistant", "content": "Kovács Őrmester válasza..."}
]}
```

### `generacios_naplo.txt`
Emberi olvasásra szánt napló: dátum, szám, témák listája.

---

## Funkciók

- **Provider váltás**: OpenAI / Anthropic / Gemini szabadon váltható
- **Folytatás**: Ha az alkalmazás megszakad, a következő indításkor folytatja ahol abbahagyta
- **Közbülső mentés**: Minden 10. párbeszéd után automatikus mentés
- **Megállítás gomb**: Biztonságos leállítás, az addig generált adatok megmaradnak
- **Ár- és időbecslés**: Valós idejű becslés a kiválasztott modell és darabszám alapján
- **Rate limit kezelés**: Exponenciális visszatartás (5s → 10s → 20s) API limit esetén
