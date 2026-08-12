from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_OPENAI_API_URL = "http://127.0.0.1:10000"
DEFAULT_OPENAI_MODEL = ""
EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
EMOTION_ALIASES = {
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "高兴": "happy",
    "开心": "happy",
    "快乐": "happy",
    "angry": "angry",
    "anger": "angry",
    "愤怒": "angry",
    "生气": "angry",
    "sad": "sad",
    "sadness": "sad",
    "悲伤": "sad",
    "伤心": "sad",
    "afraid": "afraid",
    "fear": "afraid",
    "fearful": "afraid",
    "害怕": "afraid",
    "恐惧": "afraid",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "反感": "disgusted",
    "厌恶": "disgusted",
    "melancholic": "melancholic",
    "melancholy": "melancholic",
    "低落": "melancholic",
    "忧郁": "melancholic",
    "surprised": "surprised",
    "surprise": "surprised",
    "惊讶": "surprised",
    "吃惊": "surprised",
    "calm": "calm",
    "neutral": "calm",
    "自然": "calm",
    "平静": "calm",
}
ANNOTATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:json|text|plaintext)?\s*|\s*```\s*$", re.IGNORECASE)


def _api_endpoints(api_url: str) -> tuple[str, str]:
    raw = (api_url or DEFAULT_OPENAI_API_URL).strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OpenAI API URL must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OpenAI API URL must not contain embedded credentials")
    path = parsed.path.rstrip("/")
    lower = path.lower()
    for suffix in ("/v1/chat/completions", "/v1/models", "/v1"):
        if lower.endswith(suffix):
            path = path[: -len(suffix)]
            break
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))
    return f"{base}/v1/models", f"{base}/v1/chat/completions"


def _discover_model(models_endpoint: str, api_key: str, timeout_seconds: int) -> str:
    headers = {"Accept": "application/json"}
    if (api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = urllib.request.Request(models_endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
        raise RuntimeError(f"Cannot discover a model from {models_endpoint}: {type(error).__name__}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"Model endpoint at {models_endpoint} returned invalid JSON") from error
    try:
        model = str(payload["data"][0]["id"]).strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Model endpoint at {models_endpoint} returned no model IDs") from error
    if not model:
        raise RuntimeError(f"Model endpoint at {models_endpoint} returned an empty model ID")
    return model


def openai_chat_completion(
    messages: list[dict[str, str]],
    api_url: str = DEFAULT_OPENAI_API_URL,
    api_key: str = "",
    model: str = DEFAULT_OPENAI_MODEL,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_seconds: int = 120,
) -> str:
    models_endpoint, endpoint = _api_endpoints(api_url)
    selected_model = (model or "").strip() or _discover_model(
        models_endpoint, api_key, timeout_seconds
    )
    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if (api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"LLM API HTTP {error.code} from {endpoint}: {detail}") from error
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", type(error).__name__)
        raise RuntimeError(f"Cannot reach llama.cpp/OpenAI-compatible API at {endpoint}: {reason}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"LLM API at {endpoint} returned invalid JSON") from error

    try:
        content: Any = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"LLM API at {endpoint} returned no chat-completion content") from error
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    content = str(content or "").strip()
    if not content:
        raise RuntimeError(f"LLM API at {endpoint} returned empty content")
    return content


def _json_object_from_text(content: str) -> dict[str, Any]:
    cleaned = CODE_FENCE_PATTERN.sub("", (content or "").strip()).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response does not contain a JSON object")


def parse_emotion_response(content: str) -> tuple[float, ...]:
    payload = _json_object_from_text(content)
    for container_key in ("emotions", "emotion_vector", "scores", "result"):
        nested = payload.get(container_key)
        if isinstance(nested, dict):
            payload = nested
            break

    scores = {name: 0.0 for name in EMOTION_ORDER}
    label = payload.get("emotion") or payload.get("label") or payload.get("情绪")
    if isinstance(label, str):
        resolved_label = EMOTION_ALIASES.get(label.strip().casefold())
        if resolved_label:
            scores[resolved_label] = 1.0

    for raw_key, raw_value in payload.items():
        key = EMOTION_ALIASES.get(str(raw_key).strip().casefold())
        if key is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        scores[key] = max(0.0, min(1.2, value))
    if not any(value > 0.0 for value in scores.values()):
        scores["calm"] = 1.0
    return tuple(scores[name] for name in EMOTION_ORDER)


def analyze_emotion_text(
    text: str,
    api_url: str = DEFAULT_OPENAI_API_URL,
    api_key: str = "",
    model: str = DEFAULT_OPENAI_MODEL,
    timeout_seconds: int = 120,
) -> tuple[float, ...]:
    source = (text or "").strip()
    if not source:
        raise ValueError("Emotion text is empty")
    system_prompt = (
        "You classify performance emotion for IndexTTS-2.5. Return one JSON object only, "
        "with exactly these numeric keys in this order: happy, angry, sad, afraid, disgusted, "
        "melancholic, surprised, calm. Each score must be between 0 and 1.2. Distinguish "
        "melancholic/low/depressed from ordinary sadness. Use calm for neutral speech. "
        "Do not include markdown, explanations, or any other keys."
    )
    content = openai_chat_completion(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": source}],
        api_url=api_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=timeout_seconds,
    )
    try:
        return parse_emotion_response(content)
    except ValueError as error:
        raise RuntimeError(f"Invalid emotion JSON from llama.cpp/OpenAI-compatible API: {error}") from error


def _strip_annotations(text: str) -> str:
    return ANNOTATION_PATTERN.sub(lambda match: match.group(1), text)


def _clean_text_response(content: str) -> str:
    cleaned = CODE_FENCE_PATTERN.sub("", (content or "").strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("text", "enhanced_text", "result", "output"):
            if isinstance(payload.get(key), str):
                cleaned = payload[key].strip()
                break
    cleaned = re.sub(r"^(?:输出|结果|增强文本|enhanced text|output)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def normalize_pronunciation_response(content: str, original_text: str, language: str) -> str:
    enhanced = _clean_text_response(content)
    if not enhanced:
        raise ValueError("LLM returned empty enhanced text")
    if language.upper() in {"ZH", "EN"}:
        enhanced = ANNOTATION_PATTERN.sub(
            lambda match: f"<{match.group(1)}|{match.group(2).upper()}>", enhanced
        )
    if _strip_annotations(enhanced) != original_text:
        raise ValueError(
            "LLM changed the source text instead of only adding <surface|pronunciation> annotations"
        )
    for match in ANNOTATION_PATTERN.finditer(enhanced):
        if not match.group(1).strip() or not match.group(2).strip() or "|" in match.group(2):
            raise ValueError("LLM returned an invalid pronunciation annotation")
        if language.upper() == "ZH":
            if len(match.group(1)) != 1 or re.fullmatch(r"[A-ZÜV]+[1-5]", match.group(2)) is None:
                raise ValueError(
                    "Chinese annotations must wrap one character with one numbered Pinyin syllable"
                )
    return enhanced


def enhance_pronunciation_text(
    text: str,
    language: str,
    api_url: str = DEFAULT_OPENAI_API_URL,
    api_key: str = "",
    model: str = DEFAULT_OPENAI_MODEL,
    timeout_seconds: int = 120,
    instruction: str = "",
) -> str:
    original = text or ""
    if not original.strip():
        raise ValueError("Pronunciation source text is empty")
    language = language.upper()
    format_help = {
        "ZH": (
            "Annotate only the single ambiguous Chinese character, never the whole word. "
            "Use exactly one uppercase Pinyin syllable with tone number 1-5. "
            "Example: 银<行|HANG2>里<行|XING2>走, never <银行|YIN2HANG2>."
        ),
        "EN": "For ambiguous English words use <original word|UPPERCASE CMU PHONEMES>, e.g. <minute|M IH1 . N AH0 T>.",
        "JA": "For ambiguous Japanese words use <原文|かな>, e.g. <上手|じょうず>.",
    }.get(language)
    if format_help is None:
        raise ValueError("Pronunciation enhancement supports only ZH, EN, and JA")
    system_prompt = (
        "You add pronunciation annotations for IndexTTS-2.5. "
        f"{format_help} Preserve every original character, whitespace, punctuation, and line break exactly. "
        "Only wrap text spans that genuinely need pronunciation disambiguation. Never translate, rewrite, "
        "correct, explain, or add content. Return only the final annotated text without markdown."
    )
    if (instruction or "").strip():
        system_prompt += f" Additional user constraint: {instruction.strip()}"
    content = openai_chat_completion(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": original}],
        api_url=api_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=max(512, min(4096, len(original) * 4)),
        timeout_seconds=timeout_seconds,
    )
    try:
        return normalize_pronunciation_response(content, original, language)
    except ValueError as error:
        raise RuntimeError(f"Unsafe pronunciation response rejected: {error}") from error
