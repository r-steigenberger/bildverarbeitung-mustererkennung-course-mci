"""Filter-Detektiv (Streamlit): Foto wählen, durch die Filter blättern, Operation aus der Vorlesung erraten.
Gleiche Filter und Namen wie Foliensatz und Web-App. Start: streamlit run app.py"""
import inspect
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

L = 256
ORDNER = Path(__file__).parent / "beispiele"


def bild(x):
    return np.clip(x, 0, L-1).astype(np.uint8)       # auf 0 … 255 begrenzen


def grau(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------- Filter (wie in den Notebooks)
def normal(img, p):
    return img


def innsbruck(img, c):                                # Fade: s = 0,7 · r + c
    f = img.astype(float)
    return bild(0.7 * f + c)


def kufstein(img, faktor):                            # Skizze: Sobel-Betrag, dann Negativ
    g = cv2.GaussianBlur(grau(img), (3, 3), 0)
    Gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    G = np.sqrt(Gx**2 + Gy**2)
    skizze = (L-1) - faktor * G
    return bild(skizze)


def lienz(img, k):                                    # Pop-Art: k-Means mit k Farben
    z = img.reshape(-1, 3).astype(np.float32)          # ein Pixel = eine Zeile (R, G, B)
    kriterium = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, cluster, m = cv2.kmeans(z, k, None, kriterium, 5, cv2.KMEANS_RANDOM_CENTERS)
    return bild(np.reshape(m[cluster.ravel()], img.shape))


def schwaz(img, sigma):                               # Weichzeichner: Gauß-Filter
    return cv2.GaussianBlur(img, (0, 0), sigma)


def hall(img, p):                                     # Histogrammausgleich der Helligkeit (wie VL 3)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    r = ycc[:, :, 0]
    cdf = np.cumsum(np.bincount(r.ravel(), minlength=L)) / r.size
    T = np.round((L-1) * cdf).astype(np.uint8)          # s = round((L−1) · cdf(r))
    ycc[:, :, 0] = T[r]
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def imst(img, k):                                     # Schärfen: Unsharp Masking
    f = img.astype(float)
    unscharf = cv2.GaussianBlur(f, (0, 0), 3)
    detail = f - unscharf
    return bild(f + k * detail)


def reutte(img, t):                                   # Himmel ersetzen: Pixel mit r > t bleiben
    maske = grau(img) > t
    h, w = maske.shape
    y = np.linspace(0, 1, h)[:, None, None]
    himmel = (1 - y) * np.array([40, 20, 90]) + y * np.array([250, 120, 60])   # Abendhimmel
    neu = np.where(maske[:, :, None], img, himmel)
    return bild(neu)


def kitzbuehel(img, a):                               # Vignette: zum Rand dunkler
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    d = np.hypot(x - w/2, y - h/2) / np.hypot(w/2, h/2)   # Abstand zur Mitte, 0 … 1
    return bild(img * (1 - a * d**2)[:, :, None])


def landeck(img, t):                                  # Comic: Pop-Art + schwarze Kanten
    comic = lienz(img, 4)
    g = cv2.GaussianBlur(grau(img), (3, 3), 0)
    G = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    comic[G > t] = 0
    return comic


def seefeld(img, gamma):                              # Gamma: s = c · r^γ, c = (L−1)^(1−γ)
    r = np.arange(L)
    T = np.round((L-1)**(1 - gamma) * r**gamma).astype(np.uint8)   # Kennlinie als Lookup-Table
    return T[img]


def soelden(img, p):                                  # Mosaik: jedes n-te Pixel abtasten, dann vergrößern
    n, interpolation = p
    h, w = img.shape[:2]
    klein = img[n//2::n, n//2::n]                      # Abtastung (VL 1)
    art = cv2.INTER_NEAREST if interpolation == "Nearest Neighbour" else cv2.INTER_LINEAR
    return cv2.resize(klein, (w, h), interpolation=art)   # Interpolation (VL 2)


def mayrhofen(img, p):                                # Zauberstab: Region Growing ab Seed
    t, (sx, sy) = p
    g = cv2.GaussianBlur(grau(img), (0, 0), 1).astype(float)
    h, w = g.shape
    seed = (int(sy * (h-1)), int(sx * (w-1)))
    region = np.zeros((h, w), bool)
    region[seed] = True
    liste = [seed]
    while liste:                                        # Nachbarn prüfen, bis nichts mehr dazukommt
        y, x = liste.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):                       # 8er-Nachbarschaft
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and not region[yy, xx] and abs(g[yy, xx] - g[seed]) <= t:
                    region[yy, xx] = True
                    liste.append((yy, xx))
    grauwert = 0.6 * grau(img)
    neu = np.where(region[:, :, None], img, grauwert[:, :, None])
    return bild(neu)


# Name, Funktion, Operation, VL, Erklärung, Formel (LaTeX), In der App, Parameter (Name, min, max, Start, Schritt)
FILTER = [
    ("Normal", normal, "Original", "", "Das unbearbeitete Foto. Blättern Sie mit ▶ durch die Filter.", r"s = r", "", None),
    ("Innsbruck", innsbruck, "Fade (Tonwertkurve)", "VL 3 · Punktoperation",
     "Jedes Pixel wird für sich umgerechnet. Schwarz wird zu dunklem Grau, der Kontrast sinkt.",
     r"s = 0{,}7 \cdot r + c", "Instagram „Fade“ · Lightroom: Gradationskurve, linken Endpunkt anheben", ("c", 0, 120, 60, 1)),
    ("Kufstein", kufstein, "Skizze (Sobel + Negativ)", "VL 6 + VL 3",
     "Der Gradientenbetrag G ist an Kanten groß. Das Negativ macht Kanten dunkel und flache Flächen weiß.",
     r"G = \sqrt{G_x^2 + G_y^2}, \quad s = (L-1) - f \cdot G", "Skizzenfilter · Photoshop „Konturen finden“", ("f", 0.3, 3.0, 1.0, 0.1)),
    ("Lienz", lienz, "Pop-Art (k-Means)", "VL 8 · Clustering",
     "Jedes Pixel ist ein Punkt z = (R, G, B). k-Means teilt alle Pixel in k Cluster, jedes Pixel bekommt die Farbe seines Centroids.",
     r"z \rightarrow m_i \text{ (nächster Centroid)}", "Pop-Art- und Poster-Filter (Farbreduktion)", ("k", 2, 8, 4, 1)),
    ("Schwaz", schwaz, "Weichzeichner (Gauß)", "VL 4 · Faltung",
     "Jedes Pixel wird durch einen gewichteten Mittelwert seiner Nachbarn ersetzt. Rauschen verschwindet, feine Details aber auch.",
     r"g = h_\sigma * f", "Rauschreduzierung, Weichzeichner, Tilt-Shift", ("σ", 0.5, 8.0, 2.0, 0.5)),
    ("Hall", hall, "Histogrammausgleich", "VL 3 · Histogramm",
     "Das kumulierte Histogramm dient als Tonwertkurve. Die Helligkeiten werden über 0 … 255 verteilt.",
     r"s = \mathrm{round}\big((L-1) \cdot \mathrm{cdf}(r)\big)", "„Auto“-Kontrast, Instagram „Lux“ (ähnliches Prinzip)", None),
    ("Imst", imst, "Schärfen (Unsharp Masking)", "VL 4 · Faltung",
     "Detailbild = Original minus geglättetes Bild. Mit Faktor k verstärkt wieder addiert, werden Kanten betont.",
     r"g = f + k \cdot (f - \mathrm{Gauß}(f))", "Schärfen, Struktur, Textur, Klarheit", ("k", 0.0, 5.0, 1.5, 0.1)),
    ("Reutte", reutte, "Himmel ersetzen (Schwellwert)", "VL 5 · Segmentierung",
     "Alle Pixel mit r > t bleiben, alle anderen werden durch einen Farbverlauf ersetzt. Startwert: Otsu. "
     "Bei Fotos mit hellem Himmel sieht man die Grenzen eines globalen Schwellwerts.",
     r"s = \begin{cases} \text{Foto} & r > t \\ \text{Farbverlauf} & r \le t \end{cases}",
     "„Himmel auswählen“ in Lightroom (dort mit neuronalem Netz)", ("t", 0, 255, None, 1)),
    ("Kitzbühel", kitzbuehel, "Vignette", "VL 3 (ortsabhängig)",
     "Die Helligkeit wird zum Rand hin abgesenkt. Der Faktor hängt vom Ort ab, streng genommen also keine Punktoperation.",
     r"s = r \cdot (1 - a \cdot d^2)", "Vignette", ("a", 0.0, 1.0, 0.6, 0.05)),
    ("Landeck", landeck, "Comic (k-Means + Kanten)", "VL 8 + VL 6 + VL 5",
     "Pop-Art mit k = 4 Farben, dazu werden alle Pixel mit großem Gradientenbetrag G > t schwarz.",
     r"G > t \;\rightarrow\; 0", "Comic- und Cartoon-Filter", ("t", 20, 300, 100, 5)),
    ("Seefeld", seefeld, "Gamma-Korrektur", "VL 3 · Punktoperation",
     "Eine gekrümmte Kennlinie: γ < 1 hellt die dunklen Bereiche stark auf, γ > 1 dunkelt die Mitteltöne ab.",
     r"s = c \cdot r^\gamma, \quad c = (L-1)^{1-\gamma}", "Lightroom „Tiefen“ bzw. Gradationskurve in der Mitte ziehen",
     ("γ", 0.2, 3.0, 0.5, 0.05)),
    ("Sölden", soelden, "Mosaik (Abtastung + Interpolation)", "VL 1 + VL 2",
     "Nur jedes n-te Pixel wird abgetastet. Beim Vergrößern füllt Nearest Neighbour jeden Block mit dem nächsten "
     "Abtastwert (Mosaik), bilinear wird dazwischen weich interpoliert.",
     r"\text{Abtastabstand } n", "Mosaik-/Pixel-Filter, Pinch-Zoom in der Galerie", ("n", 2, 40, 14, 1)),
    ("Mayrhofen", mayrhofen, "Zauberstab (Region Growing)", "VL 7 · Regionen",
     "Ab dem Seed wächst die Region über die 8er-Nachbarschaft, solange sich der Grauwert um höchstens t vom Seed "
     "unterscheidet. Alles außerhalb wird grau. Den Seed stellen Sie unten ein.",
     r"|r_p - r_{Seed}| \le t", "Lightroom „Bereich auswählen“, Photoshop „Zauberstab“", ("t", 2, 80, 20, 1)),
]


@st.cache_data(show_spinner=False)
def anwenden(name, img, p):
    funktion = {f[0]: f[1] for f in FILTER}[name]
    return funktion(img, p)


def histogramm(vorher, nachher):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.hist(grau(vorher).ravel(), 64, range=(0, 255), color="#999999", alpha=0.7, label="vorher")
    ax.hist(grau(nachher).ravel(), 64, range=(0, 255), color="#F49B00", alpha=0.7, label="nachher")
    ax.set_yticks([]); ax.set_xlim(0, 255); ax.set_xlabel("Grauwert"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Oberfläche
st.set_page_config(page_title="Filter-Detektiv", page_icon="📷", layout="centered")
st.title("Filter-Detektiv")
st.caption("Stellen Sie sich vor, Sie posten dieses Foto gleich in Ihre Story. "
           "Blättern Sie durch die Filter: Welche Operation aus der Vorlesung steckt dahinter?")

quelle = st.radio("Foto", ["Beispiel", "Eigenes Foto", "Kamera"], horizontal=True, label_visibility="collapsed")
datei = None
if quelle == "Beispiel":
    namen = sorted(p.stem for p in ORDNER.glob("*.jpg"))
    datei = (ORDNER / (st.selectbox("Beispielbild", namen, index=namen.index("Espresso")) + ".jpg")).read_bytes()
elif quelle == "Eigenes Foto":
    up = st.file_uploader("Foto wählen", type=["jpg", "jpeg", "png"])
    datei = up.getvalue() if up else None
else:
    cam = st.camera_input("Foto aufnehmen")
    datei = cam.getvalue() if cam else None

if datei is None:
    st.info("Bitte ein Foto wählen. Bitte keine Fotos mit erkennbaren Personen.")
    st.stop()

img = cv2.imdecode(np.frombuffer(datei, np.uint8), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
s = 720 / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

if "i" not in st.session_state:
    st.session_state.i = 0
c1, c2, c3 = st.columns([1, 3, 1])
if c1.button("◀", width="stretch"):
    st.session_state.i = (st.session_state.i - 1) % len(FILTER)
if c3.button("▶", width="stretch"):
    st.session_state.i = (st.session_state.i + 1) % len(FILTER)
name, funktion, op, vl, text, formel, app, par = FILTER[st.session_state.i]
c2.markdown(f"<h2 style='text-align:center;margin:0'>{name}</h2>"
            f"<p style='text-align:center;margin:0;color:gray'>Filter {st.session_state.i} von {len(FILTER) - 1}</p>",
            unsafe_allow_html=True)

# Parameter: Startwert, Otsu für Reutte
if par:
    pname, pmin, pmax, pstart, pstep = par
    if pstart is None:
        pstart = int(cv2.threshold(grau(img), 0, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    key = f"p_{name}"
    p = st.session_state.get(key, pstart)
else:
    p = None

if name == "Sölden":                                   # Zusatzeinstellungen
    p = (p, st.session_state.get("interp", "Nearest Neighbour"))
if name == "Mayrhofen":
    p = (p, (st.session_state.get("sx", 0.5), st.session_state.get("sy", 0.5)))
ergebnis = anwenden(name, img, p)
vorher = st.toggle("Vorher zeigen")
st.image(img if vorher else ergebnis, width="stretch")
if st.toggle("Histogramm"):
    st.pyplot(histogramm(img, ergebnis))

with st.expander(f"Auflösen: Was steckt hinter „{name}“?"):
    st.markdown(f"### {name} = {op}")
    if vl:
        st.markdown(f"**{vl}**")
    st.write(text)
    st.latex(formel)
    if app:
        st.caption(f"In der App: {app}")
    if par:
        st.slider(f"Parameter {pname}", pmin, pmax, pstart, pstep, key=key)
    if name == "Sölden":
        st.radio("Interpolation", ["Nearest Neighbour", "bilinear"], horizontal=True, key="interp")
    if name == "Mayrhofen":
        st.slider("Seed x (links → rechts)", 0.0, 1.0, 0.5, 0.01, key="sx")
        st.slider("Seed y (oben → unten)", 0.0, 1.0, 0.5, 0.01, key="sy")
    st.code(inspect.getsource(funktion), language="python")

st.caption("Bilder: scikit-image (CC0/Public Domain), Earthrise: NASA. Filternamen frei erfunden.")
