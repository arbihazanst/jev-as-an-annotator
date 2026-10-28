"""Central registry for dataset specifications and question schemas.

Add a new dataset in two steps:
1. Add a DatasetSpec entry to DATASET_REGISTRY.
2. Add a matching question builder to QUESTION_REGISTRY.

Every other module reads from here — no hardcoded specs elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from pathlib import Path
import typing as t


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    text_column: str
    label_column: str
    labels: tuple[str, ...] = ()
    answer_name: str = "sentiment"
    auto_labels: bool = False

    def resolve_labels(self) -> tuple[str, ...]:
        """Return declared labels, or auto-detect from CSV if empty or auto_labels=True."""
        if self.labels and not self.auto_labels:
            return self.labels
        if not self.path.exists():
            return self.labels
        import pandas as pd
        df = pd.read_csv(self.path, usecols=[self.label_column])
        return tuple(sorted(df[self.label_column].dropna().astype(str).unique()))


# ---------------------------------------------------------------------------
# Question criteria definitions (ID, EN, TR)
# ---------------------------------------------------------------------------

EMOTION_CRITERIA_ID: dict[str, str] = {
    "anger": "Marah, kesal, jengkel, atau geram.",
    "fear": "Takut, cemas, khawatir, atau terancam.",
    "happy": "Senang, bahagia, gembira, atau puas.",
    "love": "Cinta, kasih sayang, afeksi, atau rasa suka yang kuat.",
    "sadness": "Sedih, kecewa, berduka, atau putus asa.",
    "surprise": "Terkejut, kaget, heran, atau takjub.",
    "disgust": "Jijik, muak, atau rasa tidak suka yang mendalam.",
    "neutral": "Netral atau tidak menunjukkan emosi yang jelas.",
}

EMOTION_CRITERIA_EN: dict[str, str] = {
    "anger": "Angry, annoyed, irritated, or furious.",
    "fear": "Afraid, anxious, worried, or threatened.",
    "happy": "Happy, joyful, cheerful, or satisfied.",
    "love": "Love, affection, fondness, or strong liking.",
    "sadness": "Sad, disappointed, grieving, or hopeless.",
    "surprise": "Surprised, shocked, astonished, or amazed.",
    "disgust": "Disgusted, repulsed, or revolted.",
    "neutral": "Neutral or lacking strong emotion.",
}


# ---------------------------------------------------------------------------
# Question builders — each returns the dict the decision API expects.
# ---------------------------------------------------------------------------

def _sentiment_question_id() -> dict[str, t.Any]:
    """Indonesian three-class sentiment."""
    return {
        "sentiment": {
            "type": "choice",
            "instructions": "Apa sentimen utama yang diungkapkan dalam teks ini?",
            "criteria": {
                "positive": "Nada keseluruhan positif, mendukung, puas, atau menyenangkan.",
                "negative": "Nada keseluruhan negatif, mengkritik, kecewa, marah, atau sedih.",
                "neutral": "Bersifat informatif atau tidak memiliki sentimen positif maupun negatif yang jelas.",
            },
        }
    }


def _emotion_question_id(labels: tuple[str, ...] | None = None) -> dict[str, t.Any]:
    """Indonesian emotion question — builds criteria dynamically for whatever labels are given."""
    active = labels or ("anger", "fear", "happy", "love", "sadness")
    return {
        "emotion": {
            "type": "choice",
            "instructions": "Apa satu emosi utama yang paling dominan dalam teks ini?",
            "criteria": {
                lbl: EMOTION_CRITERIA_ID.get(lbl, f"Mengekspresikan emosi {lbl}.")
                for lbl in active
            },
        }
    }


def _sentiment_question_en() -> dict[str, t.Any]:
    """English three-class sentiment."""
    return {
        "sentiment": {
            "type": "choice",
            "instructions": "What is the primary sentiment expressed in this text?",
            "criteria": {
                "positive": "Overall positive, supportive, satisfied, or pleasant tone.",
                "negative": "Overall negative, critical, disappointed, angry, or sad tone.",
                "neutral": "Informative or lacking clearly positive or negative sentiment.",
            },
        }
    }

def _sentiment_question_hau() -> dict[str, t.Any]:
    """Hausa three-class sentiment."""
    return {
        "sentiment": {
            "type": "choice",
            "instructions": "Mene ne babban yanayin ra'ayin da aka bayyana a wannan rubutu?",
            "criteria": {
                "positive": "Gaba ɗaya, rubutun yana nuna ra'ayi mai kyau, goyon baya, gamsuwa, ko farin ciki.",
                "negative": "Gaba ɗaya, rubutun yana nuna ra'ayi mara kyau, suka, rashin gamsuwa, fushi, ko baƙin ciki.",
                "neutral": "Rubutun yana bayar da bayani ko kuma ba ya nuna ra'ayi mai kyau ko mara kyau a fili.",
            },
        }
    }


def _emotion_question_en() -> dict[str, t.Any]:
    """English five-class emotion."""
    return {
        "emotion": {
            "type": "choice",
            "instructions": "What single primary emotion is most dominant in this text?",
            "criteria": {
                "anger": "Angry, annoyed, irritated, or furious.",
                "fear": "Afraid, anxious, worried, or threatened.",
                "happy": "Happy, joyful, cheerful, or satisfied.",
                "love": "Love, affection, fondness, or strong liking.",
                "sadness": "Sad, disappointed, grieving, or hopeless.",
            },
        }
    }

def _sentiment_question_tr() -> dict[str, t.Any]:
    """Turkish three-class sentiment."""
    return {
        "sentiment": {
            "type": "choice",
            "instructions": "Bu metinde ifade edilen temel duygu nedir?",
            "criteria": {
                "positive": "Genel olarak olumlu, destekleyici, memnun veya hoş bir ton.",
                "negative": "Genel olarak olumsuz, eleştirel, hayal kırıklığına uğramış, öfkeli veya üzgün bir ton.",
                "neutral": "Bilgilendirici olan veya belirgin şekilde olumlu ya da olumsuz bir duygu içermeyen ton.",
            },
        }
    }

def _emotion_question_tr() -> dict[str, t.Any]:
    """Turkish eight-class emotion."""
    return {
        "emotion": {
            "type": "choice",
            "instructions": "Bu metinde en baskın olan tek temel duygu nedir?",
            "criteria": {
                "iğrenme": "İğrenme, tiksinme, nefret etme veya rahatsızlık hissi.",
                "korku": "Korku, endişe, kaygı veya tehdit altında hissetme.",
                "minnet": "Minnettarlık, şükran, teşekkür veya takdir duygusu.",
                "pişmanlık": "Pişmanlık, suçluluk, keşke hissi veya yapılan bir şeyden üzüntü duyma.",
                "mutluluk": "Mutluluk, neşe, sevinç veya memnuniyet.",
                "öfke": "Öfke, kızgınlık, sinirlilik veya hiddet.",
                "şaşkınlık": "Şaşkınlık, hayret, sürpriz veya beklenmedik bir durum karşısında şaşırma.",
                "üzüntü": "Üzüntü, keder, hayal kırıklığı veya umutsuzluk.",
            },
        }
    }

def _sentiment_question_bn() -> dict[str, t.Any]:
    """Bengali three-class sentiment."""
    return {
        "sentiment": {
            "type": "choice",
            "instructions": "এই পাঠ্যের সামগ্রিক মনোভাব কী?",
            "criteria": {
                "positive": "সামগ্রিকভাবে ইতিবাচক, সমর্থনমূলক, সন্তুষ্ট বা আনন্দদায়ক মনোভাব।",
                "negative": "সামগ্রিকভাবে নেতিবাচক, সমালোচনামূলক, হতাশ, ক্ষুব্ধ বা দুঃখপূর্ণ মনোভাব।",
                "neutral": "তথ্যমূলক বা এমন মনোভাব যেখানে স্পষ্টভাবে ইতিবাচক বা নেতিবাচক প্রবণতা নেই।",
            },
        }
    }

def _emotion_question_bn() -> dict[str, t.Any]:
    """Bengali six-class emotion."""
    return {
        "emotion": {
            "type": "choice",
            "instructions": "এই লেখায় সবচেয়ে প্রভাবশালী প্রধান আবেগটি কী?",
            "criteria": {
                "anger": "রাগ, ক্ষোভ, বিরক্তি বা প্রচণ্ড ক্রোধ প্রকাশ করে।",
                "disgust": "ঘৃণা, বিতৃষ্ণা, বিরাগ বা তীব্র অপছন্দ প্রকাশ করে।",
                "fear": "ভয়, উদ্বেগ, আশঙ্কা বা হুমকির অনুভূতি প্রকাশ করে।",
                "joy": "আনন্দ, খুশি, উচ্ছ্বাস বা সন্তুষ্টি প্রকাশ করে।",
                "sadness": "দুঃখ, হতাশা, শোক বা নিরাশা প্রকাশ করে।",
                "surprise": "বিস্ময়, অবাক হওয়া বা অপ্রত্যাশিত ঘটনার প্রতিক্রিয়া প্রকাশ করে।",
            },
        }
    }


# ---------------------------------------------------------------------------
# Registries — add new datasets / languages here.
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "sentiment": DatasetSpec(
        name="sentiment",
        path=DATA_DIR / "sentiments_data.csv",
        text_column="text",
        label_column="sentiment",
        labels=("positive", "negative", "neutral"),
        answer_name="sentiment",
    ),
    "emotion": DatasetSpec(
        name="emotion",
        path=DATA_DIR / "emotionals_data.csv",
        text_column="tweet",
        label_column="label",
        labels=("anger", "fear", "happy", "love", "sadness"),
        answer_name="emotion",
    ),
    "sentiment_id": DatasetSpec(
        name="sentiment_id",
        path=DATA_DIR / "sentiments_data.csv",
        text_column="text",
        label_column="sentiment",
        labels=("positive", "negative", "neutral"),
        answer_name="sentiment",
    ),
    "emotion_id": DatasetSpec(
        name="emotion_id",
        path=DATA_DIR / "emotionals_data.csv",
        text_column="tweet",
        label_column="label",
        labels=("anger", "fear", "happy", "love", "sadness"),
        answer_name="emotion",
    ),
    "sentiment_tr": DatasetSpec(
        name="sentiment_tr",
        path=DATA_DIR / "turkiye_sentiments_data.csv",
        text_column="text",
        label_column="sentiment",
        labels=("positive", "negative", "neutral"),
        answer_name="sentiment",
    ),
    "emotion_tr": DatasetSpec(
        name="emotion_tr",
        path=DATA_DIR / "turkiye_emotionals_data.csv",
        text_column="text",
        label_column="label",
        labels=("iğrenme", "korku", "minnet", "pişmanlık", "mutluluk", "öfke", "şaşkınlık", "üzüntü"),
        answer_name="emotion",
    ),
    "sentiment_bn": DatasetSpec(
        name="sentiment_bn",
        path=DATA_DIR / "bengali_sentiments_data.csv",
        text_column="Data",
        label_column="Label",
        labels=("positive", "negative", "neutral"),
        answer_name="sentiment",
    ),
    "emotion_bn": DatasetSpec(
        name="emotion_bn",
        path=DATA_DIR / "bengali_emotionals_data.csv",
        text_column="text",
        label_column="label",
        labels=("anger", "disgust", "fear", "joy", "sadness", "surprise"),
        answer_name="emotion",
    ),
    "sentiment_hau": DatasetSpec(
        name="sentiment_hau",
        path=DATA_DIR / "hausa_sentiments_data.csv",
        text_column="text",
        label_column="sentiment",
        labels=("positive", "negative", "neutral"),
        answer_name="sentiment",
    ),
}

QUESTION_REGISTRY: dict[str, t.Callable[..., dict[str, t.Any]]] = {
    "sentiment": _sentiment_question_id,
    "emotion": _emotion_question_id,
    "sentiment_id": _sentiment_question_id,
    "emotion_id": _emotion_question_id,
    "sentiment_tr": _sentiment_question_tr,
    "emotion_tr": _emotion_question_tr,
    "sentiment_bn": _sentiment_question_bn,
    "emotion_bn": _emotion_question_bn,
    "sentiment_hau": _sentiment_question_hau,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_specs(names: list[str] | None = None) -> dict[str, DatasetSpec]:
    """Return dataset specs, auto-resolving labels from CSV if not specified or auto_labels=True."""
    pool = DATASET_REGISTRY if names is None else {
        n: DATASET_REGISTRY[n] for n in names if n in DATASET_REGISTRY
    }
    missing = [n for n in (names or []) if n not in DATASET_REGISTRY]
    if missing:
        available = sorted(DATASET_REGISTRY)
        raise ValueError(f"unknown datasets: {missing}; available: {available}")
    resolved: dict[str, DatasetSpec] = {}
    for name, spec in pool.items():
        if not spec.labels or spec.auto_labels:
            resolved[name] = replace(spec, labels=spec.resolve_labels())
        else:
            resolved[name] = spec
    return resolved


def question_for(dataset: str) -> dict[str, t.Any]:
    """Build the question payload for the given dataset name with active labels."""
    builder = QUESTION_REGISTRY.get(dataset)
    spec = DATASET_REGISTRY.get(dataset)
    if builder is None:
        if spec is not None:
            labels = spec.labels or spec.resolve_labels()
            lang = "en" if "_en" in dataset else "id"
            if spec.answer_name == "emotion":
                return _emotion_question_en(labels) if lang == "en" else _emotion_question_id(labels)
        available = sorted(QUESTION_REGISTRY)
        raise ValueError(f"unsupported dataset: {dataset!r}; available: {available}")

    if spec is not None:
        labels = spec.labels or spec.resolve_labels()
        if "labels" in inspect.signature(builder).parameters:
            return builder(labels=labels)
    return builder()


def available_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(DATASET_REGISTRY)
