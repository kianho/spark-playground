#!/usr/bin/env python
# encoding: utf-8
"""

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    Read a list of paths to text files from stdin and concatenate their contents
    into a minimal-quoted csv format.

Usage:
    make_train_data.py -l LABEL [-o CSV] [-e ENCODING]

Options:
    -l, --label LABEL           Label to assign each review instance.
    -o, --output CSV            Write output to .csv file.
    -e, --encoding ENCODING     Input file encoding [default: utf-8].

"""

import os
import sys
import re
import codecs
import pandas
import re

from docopt import docopt

NEWLINE_RE = re.compile(r"[\n\r]+")

if __name__ == '__main__':
    opts = docopt(__doc__)

    label = int(opts["--label"])
    records = []
    for fn in ( ln.strip() for ln in sys.stdin ):
        with open(fn) as f:
            blob = codecs.getreader(opts["--encoding"])(f).read()
            # Wrangle multiline strings into a single line.
            blob = NEWLINE_RE.sub(" ", blob)
        records.append((label, blob))

    df = pandas.DataFrame.from_records(records, columns=None)
    df.to_csv(opts["--output"] if opts["--output"] else sys.stdout,
            index=None, header=False, encoding="utf-8")
