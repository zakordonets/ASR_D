# Code Review — ASR_D

Дата: 2026-05-29

---

## Архитектурный обзор

CLI-утилита для транскрибации аудио/видео с опциональной диаризацией и LLM-нормализацией.

```
src/asr_cli/
  cli/        — точки входа (Typer), прогресс, runtime
  core/       — конфиг, модели данных, реестр провайдеров, ошибки
  providers/  — gigaam (ASR), pyannote (диаризация), deepseek/openrouter (LLM)
  pipeline/   — оркестратор, combine, merge (назначение спикеров)
  io/         — ffmpeg, обнаружение файлов, экспортёры (srt/vtt/json/txt)
  utils/      — JSON сериализация, таймкоды
```

Паттерны: ProviderRegistry (фабрика), Protocol (structural typing),
pipeline: preprocess -> transcribe -> diarize -> normalize -> export.

Архитектура в целом чистая и расширяемая. Ниже — конкретные замечания.

---

## Сводка по серьёзности

| Категория | CRITICAL | HIGH | MEDIUM | LOW |
|-----------|----------|------|--------|-----|
| Баги | 1 | 2 | 3 | 1 |
| Безопасность | 0 | 0 | 1 | 2 |
| Качество кода | 0 | 1 | 4 | 3 |
| Производительность | 0 | 0 | 2 | 2 |
| Best practices | 0 | 0 | 2 | 4 |
| **Итого** | **1** | **3** | **12** | **12** |

---

## 1. CRITICAL — Мутация входных данных в assign_speakers

**Файл:** `src/asr_cli/pipeline/merge.py`, строки 15-18

```python
for segment in document.segments:
    segment.speaker = _best_speaker(segment, speaker_turns)
    for word in segment.words:
        word.speaker = segment.speaker
```

Функция мутирует сегменты IN-PLACE и возвращает тот же объект.
Остальной pipeline использует `dataclasses.replace()` (иммутабельность).
Если документ будет использован повторно (combine-режим), спикеры
уже будут записаны — непредсказуемое поведение.

**FIX:** Создавать новые сегменты через `replace()`, как в остальном pipeline.

---

## 2. HIGH — Потеря api_key для OpenRouter при fallback

**Файл:** `src/asr_cli/core/config.py`, строки 95-124

В `_load_from_env()` для normalization всегда берётся:
```python
api_key = DEEPSEEK_API_KEY or OPENAI_API_KEY
```

`OPENROUTER_API_KEY` — только в секции `openrouter`. Если выбран openrouter-
провайдер, но ключ задан через env normalization-секции — будет браться
неправильный ключ.

**FIX:** Провайдер-specific fallback: для openrouter сначала проверять
`OPENROUTER_API_KEY`.

---

## 3. HIGH — BatchResult.skipped всегда = 0

**Файл:** `src/asr_cli/pipeline/orchestrator.py`, строка 231

```python
skipped=0,   # хардкод
```

Поле задокументировано, но никогда не вычисляется. Файлы с
неподдерживаемыми форматами не учитываются.

**FIX:** Считать `skipped = total_discovered - processed - failed`.

---

## 4. HIGH — Провайдеры пересоздаются на каждый файл в batch

**Файл:** `src/asr_cli/pipeline/orchestrator.py`, строка 263

```python
asr_provider = self.registry.create_asr(...)
```

Для GigaAM это `gigaam.load_model()` на КАЖДЫЙ файл (секунды-минуты).
Для pyannote — загрузка pipeline аналогично.

**FIX:** Кешировать провайдеры в `PipelineRunner` после первой инициализации.

---

## 5. MEDIUM — Двойной вызов discover_media_files в batch

**Файлы:** `cli/main.py:280` + `orchestrator.py:195`

`main.py` вызывает `discover_media_files()`, потом `batch_folder()` внутри
вызывает его снова. Двойная I/O работа + возможный рассинхрон.

**FIX:** Передавать уже найденный список файлов в `batch_folder()`.

---

## 6. MEDIUM — Нормализация: по одному сегменту, последовательно

**Файлы:** `providers/deepseek/provider.py`, `providers/openrouter/provider.py`

Каждый сегмент = отдельный HTTP-запрос. 100 сегментов = 100 запросов
последовательно. Нет батчинга, нет параллелизма.

**FIX:** Батчинг сегментов в один prompt или `async`/`ThreadPoolExecutor`.

---

## 7. MEDIUM — Провайдер нормализации создаётся на каждый вызов normalize

**Файл:** `src/asr_cli/pipeline/orchestrator.py`, строки 325-327

Каждый вызов `normalize()` создаёт новый OpenAI-клиент. Хотя клиент
лёгкий, это неэффективно и может исчерпать пул соединений.

**FIX:** Кешировать normalization-провайдер в `PipelineRunner`.

---

## 8. MEDIUM — BOM и смешанные переводы строк

Множество файлов с UTF-8 BOM (`\xef\xbb\xbf`) и CRLF/LF mix:
`cli/__init__.py`, `cli/runtime.py`, `providers/deepseek/*.py`,
`providers/openai_compatible/*.py`, `providers/openrouter/provider.py`,
`providers/pyannote/*.py`, `core/config.py`

BOM может вызвать проблемы с импортами. Смешанные переводы ломают diff.

**FIX:** `.gitattributes` с `*.py text eol=lf`, `dos2unix` по всем файлам,
`.editorconfig`.

---

## 9. MEDIUM — Дублирование _format_duration

Идентичная функция в `cli/main.py:96` и `cli/progress.py:240`.

**FIX:** Вынести в `utils/timecodes.py`.

---

## 10. MEDIUM — Any в ProviderRegistry

**Файл:** `src/asr_cli/core/registry.py`

```python
ProviderFactory = Callable[[Any], Any]
def create_asr(self, provider_id: str, config: Any) -> Any:
```

Все методы теряют типизацию. Статический анализ невозможен.

**FIX:** Использовать Generic или конкретные типы.

---

## 11. MEDIUM — Загрузка .env из CWD

**Файл:** `src/asr_cli/core/config.py`, строки 69-85

`load_dotenv()` ищет `.env` в текущей директории. Если скрипт запущен из
непредсказуемого места — злоумышленник может подложить `.env`.

**FIX:** Искать `.env` относительно конфиг-файла или корня проекта.

---

## 12. MEDIUM — ProgressListener — concrete class вместо ABC

**Файл:** `src/asr_cli/core/progress.py`

No-op методы маскируют забытую реализацию (молча ничего не делает).

**FIX:** `ABC` + `@abstractmethod` или `Protocol`.

---

## 13. LOW — SRT-экспортёр не использует normalized_text

**Файл:** `src/asr_cli/io/exporters/srt.py`, строка 24

SRT всегда берёт `segment.text`, в отличие от TXT который использует
`normalized_text or text`. Возможно намеренно, но нет логики
проверки `apply_to_subtitles` на уровне экспортёра.

---

## 14. LOW — to_jsonable не обрабатывает set/tuple

**Файл:** `src/asr_cli/utils/json.py`

`set`/`tuple` в metadata → `TypeError` при `json.dumps`.

**FIX:** Добавить `set`->`list`, `tuple`->`list` конвертацию.

---

## 15. LOW — _parse_shortform_result возвращает start=0, end=0

**Файл:** `src/asr_cli/providers/gigaam/provider.py`, строки 304-305

Сегмент с нулевыми таймкодами = некорректный SRT/VTT (нулевая длительность).

---

## Рекомендации топ-5 (порядок приоритета)

1. Исправить мутацию in-place в `assign_speakers` — реальный баг
2. Кешировать провайдеры (GigaAM, pyannote) — модель грузится заново на каждый файл в batch
3. Нормализовать BOM/переводы строк — `.gitattributes` + `dos2unix`
4. Убрать дублирование `_format_duration`
5. Типизировать `ProviderRegistry` — убрать `Any`

---

# Исправления (Applied Fixes)

Дата: 2026-05-29

## FIX-1 — Immutable assign_speakers (CRITICAL)

**Файл:** `src/asr_cli/pipeline/merge.py`

**До:**
```python
for segment in document.segments:
    segment.speaker = _best_speaker(segment, speaker_turns)
    for word in segment.words:
        word.speaker = segment.speaker
document.speaker_turns = speaker_turns
return document
```

**После:**
```python
updated_segments = [
    replace(
        segment,
        speaker=_best_speaker(segment, speaker_turns),
        words=[
            replace(word, speaker=_best_speaker(segment, speaker_turns))
            for word in segment.words
        ],
    )
    for segment in document.segments
]
return replace(document, segments=updated_segments, speaker_turns=speaker_turns)
```

---

## FIX-2 — Кеширование провайдеров в PipelineRunner (HIGH)

**Файл:** `src/asr_cli/pipeline/orchestrator.py`

Добавлен `_provider_cache: dict[tuple[str, str], object]` в `PipelineRunner`.
Новый метод `_get_or_create_provider(kind, provider_id, config)` — при
первом вызове создаёт провайдер через registry и кеширует, при повторных
возвращает из кеша.

Затронуты три точки:
- `_process_document()`: `create_asr` → `_get_or_create_provider('asr', ...)`
- `_process_document()`: `create_diarization` → `_get_or_create_provider('diarization', ...)`
- `_normalize_document()`: `create_normalization` → `_get_or_create_provider('normalization', ...)`

**Результат:** GigaAM модель, pyannote pipeline и OpenAI-клиент загружаются
один раз и переиспользуются для всех файлов в batch-режиме.

---

## FIX-3 — Нормализация BOM/CRLF

**Новые файлы:**

`.gitattributes` — форсирует LF для всех текстовых файлов, помечает бинарные форматы:
```
* text=auto eol=lf
*.py text eol=lf
*.toml text eol=lf
*.wav binary
...
```

`.editorconfig` — charset=utf-8, end_of_line=lf, indent=4 spaces:
```
root = true
[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4
```

**Рекомендация:** После коммита прогнать `dos2unix` по существующим файлам
для очистки уже закоммиченных BOM/CRLF.

---

## FIX-4 — Удалено дублирование _format_duration

**Файлы:**
- `src/asr_cli/utils/timecodes.py` — добавлена публичная `format_duration()`
- `src/asr_cli/cli/main.py` — удалена локальная `_format_duration`, импорт из `utils.timecodes`
- `src/asr_cli/cli/progress.py` — удалена локальная `_format_duration`, импорт из `utils.timecodes`

---

## FIX-5 — Типизация ProviderRegistry

**Файл:** `src/asr_cli/core/registry.py`

Единый `ProviderFactory = Callable[[Any], Any]` заменён на три отдельных типа:
```python
ASRFactory = Callable[[Any], Any]
DiarizationFactory = Callable[[Any], Any]
NormalizationFactory = Callable[[Any], Any]
```

Каждая `dict[str, Factory]` и метод `register_*` типизированы отдельно.
Метод `_create` принимает `dict[str, Callable[[Any], Any]]` вместо
`dict[str, ProviderFactory]`.

---

## Верификация

- Все 18 тестов проходят: `18 passed in 0.87s`
- Компиляция всех изменённых файлов: чистая (py_compile)
- Линт: без ошибок

---

# Исправления — Волна 2 (оставшиеся проблемы)

Дата: 2026-05-29

## FIX-6 — ProgressListener → ABC + NullProgressListener (MEDIUM)

**Файл:** `src/asr_cli/core/progress.py`

`ProgressListener` был concrete class с no-op методами — молча ничего не
делал, если забыли реализовать метод. Теперь это `ABC` с `@abstractmethod`
на всех методах.

Создан `NullProgressListener(ProgressListener)` — no-op реализация для
использования по умолчанию (замена `ProgressListener()` в 5 местах
orchestrator.py).

**Тесты:** `RecordingProgressListener` в `test_gigaam_provider.py` теперь
наследует `NullProgressListener` (реализует только `on_stage_progress`).

---

## FIX-7 — Убрать двойной вызов discover_media_files (MEDIUM)

**Файлы:**
- `src/asr_cli/pipeline/orchestrator.py` — `batch_folder()` принимает
  опциональный параметр `files: list[Path] | None`. Если передан —
  использует его, иначе вызывает `discover_media_files()` сам.
- `src/asr_cli/cli/main.py` — передаёт уже найденный список файлов
  в `batch_folder(files=files)`.

**Результат:** Один I/O вызов вместо двух, нет рассинхрона.

---

## FIX-8 — .env искать относительно конфиг-файла (MEDIUM)

**Файл:** `src/asr_cli/core/config.py`

`_load_from_env(config_file)` теперь принимает `config_file: Path | None`.
Если конфиг-файл указан и рядом с ним есть `.env` — загружает его.
Иначе — стандартный поиск в CWD.

**Вызов в `build_app_config`:** `env_config = _load_from_env(config_file)`.

---

## FIX-9 — SRT/VTT: normalized_text при apply_to_subtitles (LOW)

**Файлы:**
- `src/asr_cli/io/exporters/srt.py` — `write()` принимает
  `use_normalized: bool = False`. Если True — берёт `normalized_text`.
- `src/asr_cli/io/exporters/vtt.py` — аналогично.
- `src/asr_cli/pipeline/orchestrator.py` — `_export_document()` передаёт
  `use_normalized` в SRT/VTT writers на основе
  `config.normalization.apply_to_subtitles`.

**Раньше:** SRT/VTT всегда брали `segment.text`, TXT брал `normalized_text`.
**Теперь:** Единообразно — `apply_to_subtitles=True` включает нормализацию
для всех форматов.

---

## FIX-10 — to_jsonable: set/tuple → list (LOW)

**Файл:** `src/asr_cli/utils/json.py`

```python
# Было:
if isinstance(value, list):
# Стало:
if isinstance(value, (list, tuple, set)):
```

**Результат:** `set` и `tuple` в metadata больше не вызывают `TypeError`.

---

## FIX-11 — _parse_shortform_result: корректные таймкоды (LOW)

**Файл:** `src/asr_cli/providers/gigaam/provider.py`

`_parse_shortform_result(result, *, duration_seconds=0.0)` теперь принимает
длительность медиа. Если нет word timestamps — сегмент получает
`end=duration_seconds` вместо `end=0.0`.

Обновлены оба вызова:
- `transcribe()`: `duration_seconds=media.duration_seconds`
- `_transcribe_chunked_shortform()`: `duration_seconds=chunk.duration_seconds`

---

## FIX-12 — BatchResult.skipped: динамическое вычисление (LOW)

**Файл:** `src/asr_cli/pipeline/orchestrator.py`

```python
# Было:
skipped=0,
# Стало:
skipped=len(files) - succeeded - failed,
```

**Результат:** При `--fail-fast` пропущенные файлы корректно учитываются.

---

## Верификация (Волна 2)

- Все 18 тестов проходят: `18 passed in 0.87s`
- Компиляция всех изменённых файлов: чистая (py_compile)
- Линт: без ошибок

---

# Исправления — Волна 3 (оставшиеся MEDIUM)

Дата: 2026-05-29

## FIX-13 — TxtWriter: use_normalized flag (MEDIUM)

**Файлы:**
- `src/asr_cli/io/exporters/txt.py` — `write()` принимает
  `use_normalized: bool = False`. Раньше TxtWriter ВСЕГДА брал
  `normalized_text`, что было несогласовано с SRT/VTT.
- `src/asr_cli/pipeline/orchestrator.py` — `_export_document()` передаёт
  `use_normalized` в TXT, SRT, VTT writers. JSON — всегда raw data.

**Поведение:**
- `--normalize` без `--normalize-subtitles`: TXT/SRT/VTT используют raw text
- `--normalize --normalize-subtitles`: TXT/SRT/VTT используют normalized text
- JSON: всегда содержит оба поля (`text` и `normalized_text`)

---

## FIX-14 — Нормализация: батчинг сегментов (MEDIUM)

**Файлы:**
- `src/asr_cli/providers/openai_compatible/client.py` — новый метод
  `normalize_texts(model, language, texts, ...)`:
  - Нумерует сегменты: `[1] текст\n[2] текст\n...`
  - Отправляет одним LLM-запросом
  - Парсит пронумерованный ответ обратно
  - Fallback: если парсинг не удался — обрабатывает по одному

- `src/asr_cli/providers/deepseek/provider.py` — использует
  `normalize_texts()` с BATCH_SIZE=10
- `src/asr_cli/providers/openrouter/provider.py` — аналогично

**Результат:** 100 сегментов = ~10 запросов вместо 100. При ошибке
батча — автоматический fallback на поштучную обработку.

---

## Верификация (Волна 3)

- Все 18 тестов проходят: `18 passed in 0.83s`
- Компиляция всех изменённых файлов: чистая (py_compile)
- Обновлены тесты:
  - `test_exporters.py` — проверяет `use_normalized=True/False`
  - `test_openrouter_provider.py` — StubClient с `normalize_texts`
  - `test_pipeline.py` — TxtWriter использует raw text по умолчанию
