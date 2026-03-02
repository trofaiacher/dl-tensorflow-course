import matplotlib.pyplot as plt

# -----------------------------
# Colour-blind safe palette (Okabe–Ito)
# -----------------------------
COL_BLACK  = "#000000"
COL_ORANGE = "#E69F00"
COL_SKY    = "#56B4E9"
COL_GREEN  = "#009E73"
COL_BLUE   = "#0072B2"
COL_RED    = "#D55E00"
COL_PURPLE = "#CC79A7"
COL_GREY   = "#7F7F7F"

def savefig(name: str, dpi: int = 300):
    plt.savefig(name, dpi=dpi, bbox_inches="tight")
