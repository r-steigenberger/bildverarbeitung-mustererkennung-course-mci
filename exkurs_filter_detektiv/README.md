# Filter-Detektiv (Streamlit)

Gleiche Filter und Namen wie Foliensatz und Web-App (Innsbruck = Fade, Kufstein = Skizze, Lienz = Pop-Art, Schwaz = Gauß,
Hall = Histogrammausgleich, Imst = Schärfen, Reutte = Himmel ersetzen, Kitzbühel = Vignette, Landeck = Comic).
Beim Auflösen wird zusätzlich der Python-Code des Filters angezeigt.

## Lokal starten
    pip install -r requirements.txt
    streamlit run app.py

## Für die Studierenden online stellen (Streamlit Community Cloud, kostenlos)
1. Liegt im Repository unter `exkurs_filter_detektiv/`.
2. Auf share.streamlit.io mit GitHub anmelden → „Create app“ → Repository, Branch `main` und Main file path `exkurs_filter_detektiv/app.py` wählen → Deploy.
3. Den Link (…streamlit.app) als QR-Code auf die Folie „Live ausprobieren“.

Hinweis: In Streamlit werden hochgeladene Fotos zur Berechnung an den Server geschickt (nur im Arbeitsspeicher, nicht gespeichert).
Bei der Browser-App bleiben sie auf dem Handy.
