# Semantic memory benchmarks → один CSV

Сборщик скачивает закреплённые версии диагностических benchmark-датасетов и
приводит их к единой схеме задач для black-box тестирования памяти.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_csv.py
```

Готовый результат:

```text
data/tasks.csv
```

Это единственный выходной артефакт. Сборка сразу создаёт его целиком, а
отсутствующие или неполные raw-источники автоматически докачивает в
`data/raw`.

Сборка атомарна: до полного завершения новый результат находится во временном
файле. Ошибка скачивания, адаптера, схемы, ожидаемого числа задач или
дублирующегося ID оставляет предыдущий `data/tasks.csv` неизменным.

Plain `make` выполняет ту же сборку. Из другой директории можно вызвать:

```bash
python semantic_bench_csv/build_csv.py
```

## Состав по умолчанию

| Dataset | Источники | Задачи |
|---|---:|---:|
| MRCR | 6 parquet shards | 2 400 |
| bAbI | 160 configs, 360 split-файлов | 2 080 000 |
| BABILong | 70 JSON-файлов до 16k | 7 000 |
| CLUTRR | 6 configs × 3 splits | 141 262 |
| ProofWriter | 4 parquet shards | 845 496 |
| ReCOGS | 68 TSV официальных вариантов | 1 102 402 |
| SLOG | 2 LF-представления × 4 splits | 115 694 |
| **Итого** | **534 источника** | **4 294 254** |

Для bAbI используются все задачи `qa1..qa20` в семействах `en`, `en-10k`,
`en-valid`, `en-valid-10k`, `hn`, `hn-10k`, `shuffled` и `shuffled-10k`.
Варианты `*-nosf` не входят.

ReCOGS включает COGS, concat-варианты, token-removal, preposing,
preposing+sprinkles, participle/easy, positional-index, ReCOGS v1/v2 и
variable-free. SLOG включает `cogs_LF` и `varfree_LF`; generalization sets
читаются непосредственно из официального password ZIP. Файлы из
`generation_scripts` не считаются benchmark splits.

## Параметры

```bash
python build_csv.py \
  --datasets all \
  --variants full,oracle \
  --mrcr-needles 2needle,4needle,8needle \
  --babilong-lengths 0k,1k,2k,4k,8k,16k \
  --raw-dir data/raw \
  --output data/tasks.csv
```

- `--datasets` принимает список из `mrcr,babi,babilong,clutrr,proofwriter,recogs,slog`;
- `--force-download` заново получает выбранные закреплённые источники;
- одинаковые inputs и параметры дают байт-в-байт одинаковый CSV;
- `validate_csv.py data/tasks.csv` остаётся необязательной независимой
  диагностикой и поддерживает очень большие MRCR-поля.

## CSV-схема

| Колонка | Смысл |
|---|---|
| `id` | стабильный глобально уникальный ID |
| `dataset` | семейство benchmark |
| `config` | конфигурация источника |
| `split` | `train`, `validation`, `test`, `generalization` |
| `task` | подзадача или диагностическая категория |
| `variant` | `full` либо `oracle` |
| `input` | полный вход black-box алгоритма |
| `expected_output` | правильный ответ |
| `expected_probability` | целевая вероятность ответа |
| `answer_candidates_json` | конечное множество ответов, если оно задано |
| `metadata_json` | source key, supporting facts, proof/depth и диагностика |

`full` содержит исходный контекст. `oracle` для bAbI оставляет supporting
facts, а для CLUTRR использует `clean_story`. Остальные датасеты создают только
`full`.

Скрипты не создают nonce/counterfactual варианты: это отдельные
воспроизводимые metamorphic transforms поверх базового CSV.

## Тесты

```bash
python -m pytest -q
```

Тесты проверяют точные source allowlists, полный bAbI, split mapping,
уникальность ID между shards, атомарный rollback и CSV-поля длиннее 128 KiB.
