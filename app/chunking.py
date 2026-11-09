import re
from bisect import bisect_right
from app.config import MIN_CHUNK_LENGTH

# Naive chunking, ON PURPOSE (V1 is the baseline):
# we cut the text into fixed windows of `chunk_size` characters, and each window starts
# `step` characters after the previous one, so two neighbours share `chunk_size - step` characters.
#
#   chunk_size = 800, overlap = 10%  ->  step = 720
#   chunk 0: [   0,  800)
#   chunk 1: [ 720, 1520)
#   chunk 2: [1440, 2240) ...
#
# Known weakness: windows ignore sentences and dialogues, so a passage can be cut in half.
#
# For every chunk we keep WHERE it comes from (character offsets and line numbers).
# The offsets are what makes the evaluation independent of the chunk size (see eval/run_eval.py),
# and the line numbers are what a user needs to check a reference in the original file.


def compute_step(chunk_size, overlap_percent):
    step = int(chunk_size * (1 - overlap_percent / 100))
    # Safety net: a step of 0 would loop forever.
    return max(step, 1)


def chunk_text(text, chunk_size, overlap_percent):
    step = compute_step(chunk_size, overlap_percent)
    chunks = []

    # Positions of every "\n" in the text. The line number of a character is
    # 1 + (number of "\n" before it), which bisect finds quickly in this sorted list.
    newline_positions = [i for i, char in enumerate(text) if char == "\n"]

    def line_of(offset):
        return bisect_right(newline_positions, offset - 1) + 1

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]

        if len(piece.strip()) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": piece,
                "start": start,                          # first character (included)
                "end": end,                              # last character (excluded)
                "start_line": line_of(start),
                "end_line": line_of(end - 1),
            })

        # The last window already reached the end of the text: stop here,
        # otherwise we would add chunks that are only a sub-part of this one.
        if end == len(text):
            break

    return chunks


# --- Quotes -> character spans ------------------------------------------------
# A quote (evidence in the question sets, provenance in the concept graph) is located in its file as a
# character span [start, end). Spans do not depend on the chunk setting, chunk ids do: that is why the
# evaluation AND the graph store quotes, and map them onto the chunks of the current setting with coverage().
def find_span(text, quote):
    # Files are hard-wrapped (a newline every ~70 characters), so a quote may cross line breaks.
    # We match it word by word, with any whitespace (spaces, newlines) between the words.
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Quote not found in the source: {quote!r}")
    return match.start(), match.end()


def coverage(chunk, span):
    # Share of the span [start, end) that lies inside the chunk (0 = none of it, 1 = all of it).
    overlap = min(chunk.end, span[1]) - max(chunk.start, span[0])
    return max(overlap, 0) / (span[1] - span[0])
