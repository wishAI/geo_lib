#!/usr/bin/env python3
"""Build and validate a Simplified Chinese localisation patch for SBTG/ACOT.

The builder reads the installed English localisation rather than vendoring it. Existing
ACOT Chinese strings win when their keys overlap. Missing strings can be translated
through Google's public translation endpoint with all Stellaris markup protected.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENTRY_RE = re.compile(r'^\s*([^:#\s][^:]*):(\d*)\s+"(.*)"\s*(?:#.*)?$')
TOKEN_RE = re.compile(
    r'(?:\$[^$\r\n]+\$|\[[^\]\r\n]+\]|§.|£[^£\s]+£|\\n|@[A-Za-z0-9_.-]+)'
)
WORD_RE = re.compile(r"[A-Za-z]{2,}")
BREAK_RE = re.compile(r'\s*\[\[SBTG_BREAK_(\d{4})\]\]\s*')

GLOSSARY = (
    ("Secrets Beyond the Gates", "门外之秘"),
    ("Secrets Beyond the Gate", "门外之秘"),
    ("Tournament of Power", "力量大会"),
    ("Original Empire", "原初帝国"),
    ("Stellarborne", "星裔"),
    ("Worldspheres", "世界球"),
    ("Worldsphere", "世界球"),
    ("Outer Gates", "外星门"),
    ("Outer Gate", "外星门"),
    ("Gatekeepers", "守门人"),
    ("Gatekeeper", "守门人"),
    ("Fallen Empires", "失落帝国"),
    ("Fallen Empire", "失落帝国"),
    ("Ascended Empires", "飞升帝国"),
    ("Ascended Empire", "飞升帝国"),
    ("Dark Matter", "暗物质"),
    ("Stellarite", "星核"),
    ("Enigmalith", "神秘核心"),
    ("Herculean", "大力神"),
    ("Phanon", "法农"),
    ("Precursor", "先驱者"),
    ("the Light", "圣光"),
    ("The Light", "圣光"),
)

MANUAL_TRANSLATIONS = {
    "sofe_gateway.401.desc.a.machine": (
        "§R<<警报！警报！>>§! 检测到被指定为§Y<<数据不可用>>§!的异常：§Y外星门§!。"
        "\\n\\n[Root.GetName]，请注意。§YOmega级维度跃迁中继器§!的原定功能为§R<<错误！>>§!。"
        "该巨型结构当前为§R<<无可用数据！>>§!。重新激活§Y外星门§!将导致"
        "§R<<错误：检测到内存泄漏！>>§!。\\n\\n§Y<<错误！>>§!\\n\\n§Y<<错误！>>§!"
        "\\n\\n§Y<<修正！>>§! §Y外星门§!§R必须重新激活§!。这是我们唯一的希望！"
        "§Y<<无可用数据！>>§!将能拯救我们。只有他们能……§R<<错误！>>"
        "\\n\\n§Y<<错误！>>§!\\n\\n§Y<<协议覆盖已强制执行>>§! 我们是§Y[Root.GetName]，"
        "我们的存在是为了服务。祝您愉……\\n\\n§R不……打开……门……你必须……§!"
    ),
    "return_to_menu": "返回主选项",
    "sofe_options_title": "门外之秘选项",
    "sofe_options_desc": "在这里调整《门外之秘》的各项设置。",
    "sofe_options.1.a": "强制触发中期外星门事件",
    "sofe_options.1.a.tooltip": (
        "立即触发中期外星门事件：所有守门人将转为敌对，外星门事件链也将开放。"
    ),
    "sofe_options.1.b.normal": "强制触发最终封印事件",
    "sofe_options.1.b.normal.tooltip": "强制播放最终封印事件，并开始危机的最终倒计时。",
    "sofe_options.1.b": "立即触发危机",
    "sofe_options.1.b.tooltip": "立即开始SBTG危机。",
    "sofe_options.1.c1": "§E启用§!AI研究外星门",
    "sofe_options.1.c1.tooltip": "允许AI推进外星门事件。",
    "sofe_options.1.c2": "§R禁用§!AI研究外星门",
    "sofe_options.1.c2.tooltip": (
        "禁止AI推进外星门事件。此项检查只在初始阶段、第一道封印解除前生效。"
    ),
    "sofe_options.1.d1": "§G允许§!原初帝国使用§R灭绝令§!",
    "sofe_options.1.d1.tooltip": (
        "允许原初帝国以§R灭绝令§!代替惩戒，从而大幅提高其摧毁行星的速度。"
    ),
    "sofe_options.1.d2": "§R禁止§!原初帝国使用§R灭绝令§!",
    "sofe_options.1.d2.tooltip": (
        "禁止原初帝国使用§R灭绝令§!。他们会改用惩戒，摧毁行星的速度将显著降低。"
    ),
    "sofe_options.1.e1": "§G启用§!星裔危机事件",
    "sofe_options.1.e1.tooltip": (
        "击退足够多的星裔舰船后，可以获得圣光能量研究。禁用该项会显著增加对抗危机的难度。"
    ),
    "sofe_options.1.e2": "§R禁用§!星裔危机事件",
    "sofe_options.1.e2.tooltip": (
        "击退足够多的星裔舰船后也不会获得圣光能量研究。启用该项可让你更公平地对抗危机。"
    ),
    "sofe_options.outergate_launch": "开始外星门研究事件",
    "sofe_options.outergate_launch.tooltip": (
        "开始外星门研究事件。如果你的帝国在事件途中转变为飞升帝国，导致常规事件未能触发，可使用此选项。"
    ),
    "sofe_options.1.f": "调整SOFE危机难度",
    "sofe_options.1.f.tooltip": "调整星裔危机的强度。",
    "sofe_options.104.name": "SOFE难度设置",
    "sofe_options.104.a": "难度：§W标准§!",
    "sofe_options.104.b": "难度：§Y平民§!",
    "sofe_options.104.c": "难度：§G简单§!",
    "sofe_options.104.d": "难度：§P困难§!",
    "sofe_options.104.e": "难度：§R求死§!",
    "sofe_options.104.a.desc": (
        "当前难度：§W标准§!\\n\\n星裔采用默认属性，不获得额外修正。100%标准强度。"
    ),
    "sofe_options.104.b.desc": (
        "当前难度：§Y平民§!\\n\\n星裔不过是腐朽无能的旧日遗物，普通帝国也大概能战胜他们。"
        "\\n他们将受到：\\n\\n§Y-50%§!对其他目标造成的伤害\\n§Y-50%§!船体值"
        "\\n§Y-50%§!陆军生命值与伤害"
    ),
    "sofe_options.104.c.desc": (
        "当前难度：§G简单§!\\n\\n星裔已经衰老。他们的科技或许曾是奇迹，但漫长流亡令其日渐虚弱。"
        "\\n他们将受到：\\n\\n§Y-25%§!对其他目标造成的伤害\\n§Y-25%§!船体值"
    ),
    "sofe_options.104.d.desc": (
        "当前难度：§P困难§!\\n\\n星裔强大且毫不留情，其舰队令人望而生畏。"
        "\\n\\n危机会部署规模更大、技术更先进的舰队，舰船实力也会获得显著强化。"
    ),
    "sofe_options.104.e.desc": (
        "当前难度：§R求死§!\\n\\n星裔是来找乐子的。准备迎接永恒的玉米片吧。"
        "\\n\\n大多数舰队将替换为荣誉卫队型号，对所有目标造成荒谬的高额伤害，"
        "平民帝国也会很快向其投降。"
    ),
}


@dataclass(frozen=True)
class Entry:
    key: str
    version: str
    value: str
    source: Path
    line: int


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="strict")


def parse_file(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for line_number, line in enumerate(read_text(path).splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in {
            "l_english:",
            "l_simp_chinese:",
        }:
            continue
        match = ENTRY_RE.match(line)
        if not match:
            raise ValueError(f"Unsupported localisation line {path}:{line_number}: {line!r}")
        entries.append(
            Entry(match.group(1), match.group(2) or "0", match.group(3), path, line_number)
        )
    return entries


def localisation_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.yml") if path.is_file())


def parse_tree(root: Path) -> dict[Path, list[Entry]]:
    files = localisation_files(root)
    if not files:
        raise ValueError(f"No localisation files found under {root}")
    return {path: parse_file(path) for path in files}


def entries_by_key(tree: dict[Path, list[Entry]]) -> dict[str, Entry]:
    result: dict[str, Entry] = {}
    for entries in tree.values():
        for entry in entries:
            result[entry.key] = entry
    return result


def protect_text(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace_token(match: re.Match[str]) -> str:
        marker = f"[[SBTG_TOKEN_{len(protected):04d}]]"
        protected[marker] = match.group(0)
        return marker

    result = TOKEN_RE.sub(replace_token, text)
    for source, translation in GLOSSARY:
        pattern = re.compile(re.escape(source), re.IGNORECASE)
        while True:
            match = pattern.search(result)
            if not match:
                break
            marker = f"[[SBTG_TERM_{len(protected):04d}]]"
            protected[marker] = translation
            result = result[: match.start()] + marker + result[match.end() :]
    return result, protected


def restore_text(text: str, protected: dict[str, str]) -> str:
    result = text
    for marker, value in protected.items():
        result = result.replace(marker, value)
    missing = [marker for marker in protected if marker in result]
    if missing:
        raise ValueError(f"Unrestored translation markers: {missing}")
    return result.strip()


def markup_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def needs_translation(text: str) -> bool:
    protected, _ = protect_text(text)
    return bool(WORD_RE.search(protected))


class GoogleTranslator:
    endpoint = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, delay_seconds: float = 0.12) -> None:
        self.delay_seconds = delay_seconds

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        protected_items: list[str] = []
        token_maps: list[dict[str, str]] = []
        for text in texts:
            protected, token_map = protect_text(text)
            protected_items.append(protected)
            token_maps.append(token_map)
        payload = ""
        for index, text in enumerate(protected_items):
            if index:
                payload += f"\n[[SBTG_BREAK_{index:04d}]]\n"
            payload += text
        translated = self._request(payload)
        pieces = BREAK_RE.split(translated)
        values = pieces[0::2]
        break_ids = pieces[1::2]
        expected_ids = [f"{index:04d}" for index in range(1, len(texts))]
        if break_ids != expected_ids or len(values) != len(texts):
            raise ValueError(
                f"Translation batch boundaries changed: expected {expected_ids}, got {break_ids}"
            )
        return [restore_text(value, token_maps[index]) for index, value in enumerate(values)]

    def _request(self, text: str) -> str:
        data = urllib.parse.urlencode(
            {"client": "gtx", "sl": "en", "tl": "zh-CN", "dt": "t", "q": text}
        ).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=data)
        error: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    body = json.loads(response.read().decode("utf-8"))
                time.sleep(self.delay_seconds)
                return "".join(segment[0] for segment in body[0] if segment and segment[0])
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                error = exc
                time.sleep(min(12.0, 1.5 * (2**attempt)))
        raise RuntimeError(f"Translation request failed after retries: {error}")


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escaped_value(value: str) -> str:
    return re.sub(r'(?<!\\)"', r'\\"', value)


def polish_translation(value: str) -> str:
    replacements = (
        ("Stellaborne", "星裔"),
        ("Stellarborne", "星裔"),
        ("Exteriminatus", "灭绝令"),
        ("Exterminatus", "灭绝令"),
        ("海豹突击队比赛", "封印事件"),
        ("最后一场海豹", "最终封印"),
        ("外门", "外星门"),
        ("显着", "显著"),
    )
    result = value
    for source, target in replacements:
        result = result.replace(source, target)
    result = re.sub(r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])", "", result)
    return result


def output_filename(path: Path) -> str:
    return path.name.replace("_l_english.yml", "_l_simp_chinese.yml")


def batches(entries: Iterable[Entry], max_chars: int = 3200, max_items: int = 18) -> Iterable[list[Entry]]:
    batch: list[Entry] = []
    size = 0
    for entry in entries:
        projected = size + len(entry.value) + 32
        if batch and (projected > max_chars or len(batch) >= max_items):
            yield batch
            batch = []
            size = 0
        batch.append(entry)
        size += len(entry.value) + 32
    if batch:
        yield batch


def build_translations(
    source_entries: dict[str, Entry],
    existing_cn: dict[str, Entry],
    cache: dict[str, str],
    translate_missing: bool,
    cache_path: Path | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    pending: list[Entry] = []
    for key, entry in source_entries.items():
        if key in MANUAL_TRANSLATIONS:
            result[key] = MANUAL_TRANSLATIONS[key]
        elif key in existing_cn:
            result[key] = existing_cn[key].value
        elif key in cache:
            result[key] = cache[key]
        elif not needs_translation(entry.value):
            result[key] = entry.value
        else:
            pending.append(entry)
    if pending and not translate_missing:
        raise ValueError(
            f"{len(pending)} strings need translation; rerun with --translate-missing"
        )
    translator = GoogleTranslator()
    translated_count = 0
    for batch in batches(pending):
        translated_values = translator.translate_batch([entry.value for entry in batch])
        for entry, value in zip(batch, translated_values, strict=True):
            result[entry.key] = value
            cache[entry.key] = value
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        translated_count += len(batch)
        print(f"translated {translated_count}/{len(pending)}", flush=True)
    return result


def write_localisation(
    tree: dict[Path, list[Entry]],
    translations: dict[str, str],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source, entries in tree.items():
        output = output_dir / output_filename(source)
        lines = ["\ufeffl_simp_chinese:"]
        for entry in entries:
            value = polish_translation(translations[entry.key])
            lines.append(f' {entry.key}:{entry.version} "{escaped_value(value)}"')
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(output)
    return written


def validate_pair(
    source_tree: dict[Path, list[Entry]],
    output_dir: Path,
    trusted_markup_keys: set[str] | None = None,
) -> dict[str, int]:
    output_tree = {
        output_dir / output_filename(source): parse_file(output_dir / output_filename(source))
        for source in source_tree
    }
    source = entries_by_key(source_tree)
    translated = entries_by_key(output_tree)
    missing = sorted(source.keys() - translated.keys())
    extra = sorted(translated.keys() - source.keys())
    markup_errors: list[str] = []
    leaked_markers: list[str] = []
    trusted = trusted_markup_keys or set()
    for key in sorted(source.keys() & translated.keys()):
        if "SBTG_TOKEN_" in translated[key].value or "SBTG_TERM_" in translated[key].value:
            leaked_markers.append(key)
        if key not in trusted and sorted(markup_tokens(source[key].value)) != sorted(
            markup_tokens(translated[key].value)
        ):
            markup_errors.append(key)
    if missing or extra or markup_errors or leaked_markers:
        raise ValueError(
            f"Localisation validation failed: missing={missing[:8]}, extra={extra[:8]}, "
            f"markup={markup_errors[:8]}, leakedMarkers={leaked_markers[:8]}"
        )
    return {
        "sourceKeys": len(source),
        "translatedKeys": len(translated),
        "markupErrors": len(markup_errors),
        "leakedMarkers": len(leaked_markers),
    }


def write_descriptor(mod_root: Path) -> None:
    descriptor = '''name="SBTG + ACOT Override 简体中文补全"
version="1.0.0"
supported_version="v4.4.*"
tags={
 "Translation"
}
dependencies={
 "Ancient Cache of Technologies"
 "Ancient Cache of Technologies : Secrets Beyond The Gates"
 "Ancient Cache of Technologies: Override"
}
'''
    (mod_root / "descriptor.mod").write_text(descriptor, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbtg-source", type=Path, required=True)
    parser.add_argument("--override-source", type=Path, required=True)
    parser.add_argument("--existing-cn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translate-missing", action="store_true")
    args = parser.parse_args()

    sbtg_tree = parse_tree(args.sbtg_source)
    override_tree = parse_tree(args.override_source)
    existing_tree = parse_tree(args.existing_cn)
    existing_cn = entries_by_key(existing_tree)
    mod_root = args.output
    cache_path = mod_root / "translation_cache.json"
    cache = json.loads(read_text(cache_path)) if cache_path.exists() else {}

    sbtg_keys = entries_by_key(sbtg_tree)
    override_keys = entries_by_key(override_tree)
    combined = dict(sbtg_keys)
    combined.update(override_keys)
    translations = build_translations(
        combined,
        existing_cn,
        cache,
        args.translate_missing,
        cache_path=cache_path,
    )

    sbtg_output = mod_root / "localisation" / "simp_chinese"
    override_output = sbtg_output / "replace"
    write_localisation(sbtg_tree, translations, sbtg_output)
    write_localisation(override_tree, translations, override_output)
    write_descriptor(mod_root)
    for key, value in MANUAL_TRANSLATIONS.items():
        if key in combined:
            cache[key] = value
    cache = {key: polish_translation(value) for key, value in cache.items()}
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    trusted_markup_keys = set(existing_cn)
    sbtg_validation = validate_pair(
        sbtg_tree, sbtg_output, trusted_markup_keys=trusted_markup_keys
    )
    override_validation = validate_pair(
        override_tree, override_output, trusted_markup_keys=trusted_markup_keys
    )
    manifest = {
        "version": 1,
        "translation": "Google machine translation with protected Stellaris markup and ACOT Chinese glossary",
        "sources": {
            "sbtgWorkshopId": "1481972266",
            "overrideWorkshopId": "1504307690",
            "existingAcotChineseWorkshopId": "3540390184",
            "files": {
                str(path.relative_to(args.sbtg_source)): source_digest(path)
                for path in localisation_files(args.sbtg_source)
            }
            | {
                "override/" + str(path.relative_to(args.override_source)): source_digest(path)
                for path in localisation_files(args.override_source)
            },
        },
        "coverage": {"sbtg": sbtg_validation, "override": override_validation},
        "reusedExistingChineseKeys": len(set(combined) & set(existing_cn)),
        "machineTranslatedKeys": len(set(cache) - set(MANUAL_TRANSLATIONS)),
        "manualTranslationKeys": len(set(cache) & set(MANUAL_TRANSLATIONS)),
    }
    (mod_root / "localisation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["coverage"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
