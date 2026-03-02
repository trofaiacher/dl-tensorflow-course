import matplotlib.pyplot as plt

COLORS = {
    "black":  "#000000",
    "orange": "#E69F00",
    "blue":   "#0072B2",
    "red":    "#D55E00",
}

def savefig(name: str, dpi: int = 300):
    plt.savefig(name, dpi=dpi, bbox_inches="tight")
