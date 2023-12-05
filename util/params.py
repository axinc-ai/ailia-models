# =============================================================================
# Available input data modalities
MODALITIES = ['image', 'video', 'audio']

# recognized extension (for glob.glob)
EXTENSIONS = {
    'image': ['*.[pP][nN][gG]', '*.[jJ][pP][gG]', '*.[jJ][pP][eE][gG]', '*.[bB][mM][pP]'],
    'video': ['*.[mM][pP]4'],
    'audio': ['*.[mM][pP]3', '*.[wW][aA][vV]'],
    'text':  ["*.[tT][xX][tT]"],
}

# =============================================================================
