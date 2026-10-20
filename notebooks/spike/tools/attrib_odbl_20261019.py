"""그림에 출처를 표기하고 노트북 맨 앞에 출처 고지 셀을 넣는다(한 번만 실행)."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

BASE = r"C:\Users\mycoms\OneDrive\Desktop\Korea Feature Manifold\notebooks\spike"
OSM_MAP = "© OpenStreetMap contributors (ODbL 1.0) | boundaries: vuski/admdongkor (CC BY 4.0), SGIS"
OSM_TT = "Travel times derived from © OpenStreetMap contributors (ODbL 1.0)"
KOSIS = "Commuting OD: KOSIS DT_1PA2021 (2020 Population and Housing Census)"
FIGS = {
    r"s3_outputs\figures\map_kappa.png": OSM_MAP,
    r"s3_outputs\figures\map_kappa_bottom20.png": OSM_MAP,
    r"s3_outputs\figures\map_kappa_ex_top20.png": OSM_MAP,
    r"s4_outputs\d_time_247_poincare.png": OSM_TT,
    r"s4_outputs\d_time_247_euclid.png": OSM_TT,
    r"s4_outputs\d_flow_main_poincare.png": KOSIS,
    r"s4_outputs\d_flow_main_euclid.png": KOSIS,
}
FONT = r"C:\Windows\Fonts\arial.ttf"

for rel, text in FIGS.items():
    path = os.path.join(BASE, rel)
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    fs = max(12, h // 55)
    while True:
        font = ImageFont.truetype(FONT, fs)
        box = ImageDraw.Draw(im).textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        if tw <= w - 24 or fs <= 9:
            break
        fs -= 1
    pad = max(3, fs // 3)
    x, y = w - tw - 2 * pad - 4, h - th - 2 * pad - 4
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=(255, 255, 255, 210))
    im = Image.alpha_composite(im, overlay)
    ImageDraw.Draw(im).text((x + pad - box[0], y + pad - box[1]), text, fill=(40, 40, 40, 255), font=font)
    im.convert("RGB").save(path)
    print("stamped", rel, w, h, fs)

NOTE = (
    "> **출처 고지**: 도로망·통행시간과 그 파생값(표·그림)은 © OpenStreetMap contributors, "
    "[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)이다. "
    "그 밖의 자료 출처와 라이선스는 저장소의 `NOTICE.md`를 본다."
)
for name in ["01_network.ipynb", "02_curvature.ipynb", "03_embedding.ipynb"]:
    path = os.path.join(BASE, name)
    nb = json.load(open(path, encoding="utf-8"))
    if any("출처 고지" in "".join(c["source"]) for c in nb["cells"][:2]):
        print("skip", name)
        continue
    cell = {"cell_type": "markdown", "metadata": {}, "source": [NOTE]}
    if nb.get("nbformat", 4) >= 4 and nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = "odbl-notice"
    nb["cells"].insert(0, cell)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("notice", name)
