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
