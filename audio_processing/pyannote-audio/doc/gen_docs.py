"""
This script will generate the rst docs for the api
"""

import os
from os import path

bp = breakpoint


def capitalise(s):
    news = ""
    for word in s.split("_"):
        news += word.capitalize()
    return news


def process_dir(level, p):
    md = ""
    basename = path.basename(p)

    title = capitalise(basename)
    md += f"{'#'*level} {title}\n\n"
    subdirs = os.listdir(p)

    for f in subdirs:
        m = path.join(subdir, f)
        if path.isdir(m):
            md += process_dir(level + 1, path.join(p, f))
        else:
            if "__" in f:
                continue
            module = m[3:].replace("/", ".")[:-3]
            md += f"""
```eval_rst
.. automodule:: {module}
    :members:

```

"""
    return md


DIR = "../pyannote/audio"

for module in os.listdir(DIR):
    # Each folder will become and rst file
    # Each file/folder will have a # prepended to it
    # Recursively we will add another # each level

    # Initialise Markdown
    md = ""

    subdir = path.join(DIR, module)

    # Skip if not directory
    if not path.isdir(subdir) or "__" in module:
        continue

    md += process_dir(1, subdir)
    with open(f"./source/api/{module}.md", "w") as f:
        f.write(md)
