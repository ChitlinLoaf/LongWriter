"""Chord progression research helper for Pythonista.

This script is designed for the Pythonista app on iOS, but runs in any
standard Python 3 environment. It models common music theory concepts
(scale construction, diatonic triads, and functional harmony) and then
uses those to "research" (explore) chord progressions.

Usage:
    python pythonista_chord_progression_research.py

The output includes:
- A summary of the chosen key and its diatonic chords.
- A short explanation of functional roles (Tonic, Predominant, Dominant).
- Several progression suggestions with reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC_FLATS = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}


MAJOR_SCALE_STEPS = [2, 2, 1, 2, 2, 2, 1]
MINOR_SCALE_STEPS = [2, 1, 2, 2, 1, 2, 2]  # natural minor

ROMAN_NUMERALS_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
ROMAN_NUMERALS_MINOR = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

FUNCTION_MAP = {
    "Tonic": {"I", "i", "vi", "VI", "iii", "III"},
    "Predominant": {"ii", "ii°", "IV", "iv"},
    "Dominant": {"V", "v", "vii°", "VII"},
}


@dataclass(frozen=True)
class Chord:
    degree: int
    roman: str
    name: str
    tones: Sequence[str]

    def function(self) -> str:
        for role, members in FUNCTION_MAP.items():
            if self.roman in members:
                return role
        return "Other"


def normalize_note(note: str) -> str:
    note = note.strip().capitalize()
    return ENHARMONIC_FLATS.get(note, note)


def build_scale(root: str, steps: Sequence[int]) -> List[str]:
    root = normalize_note(root)
    if root not in NOTE_NAMES:
        raise ValueError(f"Unknown root '{root}'. Use sharps or common flats.")

    idx = NOTE_NAMES.index(root)
    scale = [NOTE_NAMES[idx]]
    for step in steps[:-1]:
        idx = (idx + step) % len(NOTE_NAMES)
        scale.append(NOTE_NAMES[idx])
    return scale


def diatonic_triads(scale: Sequence[str], roman_numerals: Sequence[str]) -> List[Chord]:
    chords = []
    for degree in range(7):
        root = scale[degree]
        third = scale[(degree + 2) % 7]
        fifth = scale[(degree + 4) % 7]
        name = f"{root} {roman_numerals[degree]}"
        chords.append(Chord(degree + 1, roman_numerals[degree], name, (root, third, fifth)))
    return chords


def describe_functional_harmony(chords: Sequence[Chord]) -> List[str]:
    lines = []
    for chord in chords:
        lines.append(f"{chord.roman}: {chord.name} -> {chord.function()}")
    return lines


def suggest_progressions(chords: Sequence[Chord], length: int = 4) -> List[List[Chord]]:
    """Generate progression suggestions using functional flow rules.

    This is a lightweight "research" model:
    - Tonic can move to Predominant or Dominant.
    - Predominant should move to Dominant.
    - Dominant should resolve to Tonic.
    """
    tonic = {c for c in chords if c.function() == "Tonic"}
    predominant = {c for c in chords if c.function() == "Predominant"}
    dominant = {c for c in chords if c.function() == "Dominant"}

    def allowed_next(chord: Chord) -> Iterable[Chord]:
        if chord in tonic:
            return predominant | dominant | tonic
        if chord in predominant:
            return dominant
        if chord in dominant:
            return tonic
        return chords

    suggestions = []
    for start in tonic:
        for combo in product(chords, repeat=length - 1):
            progression = [start, *combo]
            if all(
                nxt in allowed_next(current)
                for current, nxt in zip(progression, progression[1:])
            ):
                suggestions.append(progression)
    return suggestions


def format_progression(prog: Sequence[Chord]) -> str:
    romans = " - ".join(chord.roman for chord in prog)
    names = ", ".join(chord.name for chord in prog)
    return f"{romans} | {names}"


def main() -> None:
    key = "C"
    mode = "major"

    steps = MAJOR_SCALE_STEPS if mode == "major" else MINOR_SCALE_STEPS
    roman = ROMAN_NUMERALS_MAJOR if mode == "major" else ROMAN_NUMERALS_MINOR

    scale = build_scale(key, steps)
    chords = diatonic_triads(scale, roman)

    print(f"Key: {key} {mode}")
    print("Scale:", " ".join(scale))
    print("\nDiatonic triads:")
    for chord in chords:
        print(f"- {chord.name}: {', '.join(chord.tones)}")

    print("\nFunctional roles:")
    for line in describe_functional_harmony(chords):
        print(f"- {line}")

    print("\nSuggested progressions (sample):")
    suggestions = suggest_progressions(chords, length=4)
    for prog in suggestions[:8]:
        print(f"- {format_progression(prog)}")

    print(
        "\nResearch tips: try changing key/mode or length to explore different "
        "progression behaviors and compare the functional flow."
    )


if __name__ == "__main__":
    main()
