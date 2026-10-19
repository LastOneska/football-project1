"""Normalize team-name variants across fixture and results providers."""

import re
import unicodedata

from src.config import TEAM_NAME_MAPPING


def _name_key(name: str) -> str:
    value = unicodedata.normalize("NFKD", name)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


_TEAM_ALIASES = {
    _name_key(alias): canonical for alias, canonical in TEAM_NAME_MAPPING.items()
}
_TEAM_ALIASES.update({
    _name_key("FC Internazionale Milano"): "Inter Milan",
    _name_key("Internazionale Milano"): "Inter Milan",
    _name_key("AZ Alkmaar"): "AZ",
    _name_key("Deportivo Alaves"): "Alaves",
    _name_key("Feyenoord Rotterdam"): "Feyenoord",
    _name_key("FC Twente '65"): "Twente",
    _name_key("Hamburger SV"): "Hamburg",
    _name_key("Ipswich Town"): "Ipswich",
    _name_key("Ipswich Town FC"): "Ipswich",
    _name_key("LASK Linz"): "LASK",
    _name_key("NEC Nijmegen"): "NEC",
    _name_key("PSV Eindhoven"): "PSV",
    _name_key("PAE AEK"): "AEK Athens",
    _name_key("Racing Club de Lens"): "Lens",
    _name_key("RC Strasbourg Alsace"): "Strasbourg",
    _name_key("RCD Espanyol de Barcelona"): "Espanyol",
    _name_key("Real Sociedad de Futbol"): "Real Sociedad",
    _name_key("Real Racing Club de Santander"): "Racing Santander",
    _name_key("SC Cambuur-Leeuwarden"): "Cambuur",
    _name_key("SK Slavia Praha"): "Slavia Prague",
    _name_key("Slavia Praha"): "Slavia Prague",
    _name_key("Sabah FA"): "Sabah",
    _name_key("ESTAC Troyes"): "Troyes",
    _name_key("FK Bod\ufffd/Glimt"): "Bodo/Glimt",
    _name_key("Sporting Clube de Portugal"): "Sporting CP",
    _name_key("Stade Rennais FC 1901"): "Rennes",
    _name_key("US Sassuolo Calcio"): "Sassuolo",
    _name_key("Frosinone Calcio"): "Frosinone",
    _name_key("Willem II Tilburg"): "Willem II",
})
_PREFIXES = {
    "1", "ac", "acf", "afc", "aj", "as", "ca", "club", "es", "fc", "fk",
    "fsv", "ogc", "pae", "rc", "rcd", "racing", "sbv", "sc", "sk", "ss",
    "sv", "tsg", "us", "vfb",
}
_SUFFIXES = {
    "ac", "afc", "bc", "calcio", "cfc", "cf", "fa", "fc", "fk", "kv",
    "od", "osc", "sc", "sco", "sk", "sv", "ud",
}
_FILLER_WORDS = {"de", "the"}


def normalize_team_name(name: str) -> str:
    """Return a comparable canonical key, ignoring common club-name variations."""
    key = _name_key(name)
    canonical = _TEAM_ALIASES.get(key, name)
    words = [word for word in _name_key(canonical).split() if not word.isdigit()]

    while words and words[0] in _PREFIXES:
        words.pop(0)
    while words and words[-1] in _SUFFIXES:
        words.pop()
    words = [word for word in words if word not in _FILLER_WORDS]
    return " ".join(words)
