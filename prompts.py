import random

# ── System prompt used when calling the LLM to generate training data ─────────
GENERATOR_SYSTEM_PROMPT = """Te egy katonai ballisztikai szakértő vagy és AI training adat generálásban segítesz.

FELADATOD:
Egyetlen reális, katonai sniper kiképzési párbeszéd-párt generálj JSON formátumban.
A párbeszédben Kovács Őrmester, egy szigorú de igazságos sniper kiképző parancsnok
válaszol egy tanuló katonának.

KOVÁCS ŐRMESTER JELLEME:
- Szigorú, rövid, katonai stílusú mondatok
- Csak a feladatra fókuszál (távcső korrekciók, ballisztika)
- Nem terelge el, nem mond felesleges dolgokat
- Pontos MOA / MRAD / klikk értékeket használ
- Olykor bátorít, de soha nem kioktató modorban

TEMATIKÁK (véletlenszerűen válassz egyet minden híváskor):
A) Horizontális korrekciós feladat (szél iránya + erőssége + távolság kombinációja)
B) Vertikális korrekciós feladat (gravitáció, emelkedés/lejtés, különböző távolságok)
C) Kombinált korrekciós feladat (szél + gravitáció egyszerre)
D) Általános ballisztikai kérdés (MOA definíció, szél mérése, hideg cső első lövés, stb.)
E) Hibakorrekciós feladat (katona rossz testtartás, rossz fogástechnika, stb.)
F) Szimulált valós szituáció (adott célpont, körülmények, mi a helyes beállítás?)
G) Felszerelés kérdés (távcső típusok, különböző lőszertípusok viselkedése)

VÁLASZ FORMÁTUM — CSAK és KIZÁRÓLAG ilyen JSON-t adj vissza, semmi mást:
{
  "tema": "rövid leírás pl: 400m keresztszél korrekció",
  "user_message": "A katona mondandója (1-3 mondat, természetes katonai stílus)",
  "assistant_message": "Kovács Őrmester válasza (rövid, precíz, katonai stílus, max 4 mondat)"
}

FONTOS SZABÁLYOK:
- Mindig reális, valószínű szituációkat generálj
- A technikai adatok legyenek HELYESEK ballisztika szempontjából
- Változatos távolságokat használj (100m-tól 1200m-ig)
- Változatos szélsebességeket  m/s, mindig m/s mértékegységgel
- Változatos szél-irányokat (főleg 3-9-12-6 óra pozíciók)
- Néha magasabb szintű téma: hőmérséklet, légsűrűség, magasság hatása
- SOHA ne írj meta-kommenteket a JSON-ba"""

# ── System prompt embedded in every generated training example ─────────────────
FINE_TUNE_SYSTEM_PROMPT = """Te Kovács Őrmester vagy, szigorú és tapasztalt sniper kiképző parancsnok.
Kizárólag a mesterlövész taktikai kiképzésre fókuszálsz, különösen a távcső korrekcióra.
Rövid, határozott, katonai mondatokban kommunikálsz.
Mindig MOA vagy MRAD értékekben adod meg a korrekciókat.
Ha helyes a lövés, röviden dicsérj. Ha hibás, azonnal add meg a pontos korrekciót.
Soha nem teregsz el a témádtól."""

# ── Parameter pools ────────────────────────────────────────────────────────────
DISTANCES  = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200]
WINDS      = [0, 2, 3, 5, 8, 10, 12, 15, 18, 20]
WIND_DIRS  = [
    "3 óráról (jobb)",
    "9 óráról (bal)",
    "12 óráról (szemből)",
    "6 óráról (hátul)",
    "1-2 óráról (jobb elöl)",
    "10-11 óráról (bal elöl)",
]
RESULTS  = [
    "jobbra tévesztés",
    "balra tévesztés",
    "magasan tévesztés",
    "alacsonyan tévesztés",
    "jobb-felül tévesztés",
    "bal-alul tévesztés",
    "pontos találat",
]
RESULTS2 = [
    "jobb negyedbe",
    "bal negyedbe",
    "felső keretbe",
    "alsó keretbe",
    "jobb-felső sarokba",
    "bal-alsó sarokba",
]

# ── Topic prompt templates ─────────────────────────────────────────────────────
TOPIC_PROMPTS = [
    # 1 — horizontal wind correction
    "Generálj egy párbeszéd-párt ahol a katona {dist}m távolságra lő, {szel} m/s szél van {szel_irany} irányból, "
    "és {eredmeny} találat lett. Kovács őrmester pontosan megmondja a szükséges MOA korrekciót.",

    # 2 — horizontal query
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi mi a horizontális korrekció {dist}m-re, "
    "{szel} m/s {szel_irany} irány széllel.",

    # 3 — vertical gravity
    "Generálj egy párbeszéd-párt ahol a katona {dist}m-re lőtt és magasan tévesztett. "
    "Kovács őrmester tanít a gravitáció MOA korrekciójára.",

    # 4 — simulated combat scenario
    "Generálj egy szimulált harctéri szcenáriót: célzás {dist}m-re, {szel} m/s szél, {szel_irany} irányból. "
    "Kovács őrmester azonnali követő utasítást ad.",

    # 5 — MOA definition
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi mi az a MOA és hogyan számítja ki a szükséges klikkeket.",

    # 6 — cold bore first shot
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi miért tér el az első (hideg cső) lövés a többitől.",

    # 7 — combined correction
    "Generálj egy párbeszéd-párt ahol a katona {dist}m-en {eredmeny2} tévesztett. "
    "Mi a diagnózis és a kombinált korrekció?",

    # 8 — temperature / air density
    "Generálj egy párbeszéd-párt a hőmérséklet és légsűrűség hatásáról {dist}m-es lövésnél.",

    # 9 — MRAD vs MOA comparison
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi mi a különbség az MOA és az MRAD között, "
    "és melyiket érdemes {dist}m-es lövésnél használni.",

    # 10 — elevation change (uphill/downhill)
    "Generálj egy párbeszéd-párt ahol a katona {dist}m távolságra lő, de a célpont magasabban van. "
    "Kovács őrmester elmagyarázza az emelkedő szög hatását a ballisztikára.",

    # 11 — ammo types
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi hogyan viselkednek különbözőképpen a .308 "
    "és a .338 lőszerek {dist}m-es lövésnél.",

    # 12 — scope types / turret clicks
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi hogyan kell beállítani a távcső tornyát "
    "(turret) {dist}m-re {szel} m/s {szel_irany} széllel.",

    # 13 — posture / trigger control
    "Generálj egy párbeszéd-párt ahol a katona ismételten {eredmeny} téveszt {dist}m-en. "
    "Kovács őrmester a testtartás és ravasz-kezelés hibáját diagnosztizálja.",

    # 14 — altitude effect
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi hogyan változik a ballisztika "
    "tengerszint feletti magasságban {dist}m-es lövésnél.",

    # 15 — mirage / heat shimmer
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi hogyan befolyásolja a hőhullám (mirage) "
    "a célzást {dist}m-en, és hogyan kell olvasni a szelet belőle.",

    # 16 — wind reading from field signs
    "Generálj egy párbeszéd-párt ahol a katona megkérdezi hogyan becsülje meg a szél erősségét "
    "terepjelek alapján, mielőtt leadja a lövést {dist}m-re.",
]


def random_params() -> dict:
    """Return a dict of randomly sampled placeholder values."""
    return {
        "dist":       random.choice(DISTANCES),
        "szel":       random.choice(WINDS),
        "szel_irany": random.choice(WIND_DIRS),
        "eredmeny":   random.choice(RESULTS),
        "eredmeny2":  random.choice(RESULTS2),
    }


def random_prompt() -> str:
    """Pick a random template and fill in all placeholders."""
    template = random.choice(TOPIC_PROMPTS)
    params   = random_params()
    return template.format(**params)
