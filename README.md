# Antigravity Codex Orchestrator

**Розширена мульти-бекендна версія гібридного Cloud–Edge ШІ-оркестратора із Zero-Knowledge AST-приватністю, адаптивним R&D-рушієм, контролем контексту локальної моделі та системою запису/відтворення/відновлення сесій (`Record / Replay / Resume`).**

Поєднує хмарного стратегічного планувальника (**Cloud Architect** з вибором бекенду: **Claude Haiku 5.5 (`claude`)**, **OpenAI Codex (`codex exec`)** або **Antigravity (`agy`)**) із локальною робочою силою на GPU (**On-Device Workforce** через `Ollama` / `Gemma 4`). Тіла функцій, алгоритми та секрети ніколи не залишають пристрій.

---

## 1. Головні відмінності та можливості цієї версії

| Шар системи | Базовий `antigravity-hybrid-orchestrator` | `antigravity-codex-orchestrator` (Цей репозиторій) |
| :--- | :--- | :--- |
| **Бекенди Cloud Architect** | Лише `agy` (`--output-format json`) | **3 бекенди (`--architect`)**: `haiku` (`claude`), `luna` (`codex exec`), `agy` |
| **Валідація плану (DAG)** | Базова перевірка ациклічності | `PlanValidator` (`planner/validator.py`): перевіряє, що `verification_command` реально тестує артефакт, а не просто перелічує файли (`ls`/`rg --files`), та автоматично просить хмару виправити слабкий план |
| **Адаптивне виконання** | Фіксований ліміт кроків та ретраїв | `ExecutionControl` (`engine/control.py`): необмежена кількість ходів воркера, детектор зациклення за відбитками прогресу (fingerprints), бюджети часу (`--max-runtime`) і токенів (`--max-remote-tokens`), watchdog-таймаути |
| **Запис, Replay і Resume** | Немає | Автоматичний запис у `.antigravity_runs/run-*.jsonl` (`0600`), повне відтворення в HUD (`--replay`) та відновлення перерваних сесій з `.state.json` (`--resume`) |
| **Керування контекстом GPU** | Повна історія повідомлень | `workforce/context.py`: явний `OLLAMA_NUM_CTX=16384`, розумне обрізання shell-виводу (перші 20 + останні 50 рядків зі збереженням traceback), стиснення старих tool-результатів в однорядкові резюме |
| **Інтерактивний режим** | Одноразовий запуск | `--interactive`: живий REPL-сеанс для послідовної постановки R&D-задач у спільному воркспейсі |
| **Тестове покриття** | 574 тести | **693 тести** + окремий `tools/eval_suite.py` для незалежної перевірки якості |

---

## 2. Повна архітектурна схема

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     ЛОКАЛЬНИЙ РОБОЧИЙ ПРОСТІР (Workspace)                    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                1. ZERO-KNOWLEDGE AST PRIVACY ENGINE (privacy/)               │
│  • ast_stripper.py      : Вирізає всі тіла функцій (`...`) і константи       │
│  • workspace_privacy.py : Гарантує відсутність секретів і тіл в огляді папок │
│  • verifier.py          : Блокує відправку при найменшому витоку літералів   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ 0 байт коду функцій / секретів
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│             2. МУЛЬТИ-БЕКЕНДНИЙ CLOUD ARCHITECT (architect/client.py)        │
│    [--architect haiku (Claude CLI) | luna (Codex CLI) | agy (Antigravity)]   │
│                                      │                                       │
│                                      ▼                                       │
│               [PlanValidator (planner/validator.py)]                         │
│     Перевіряє коректність залежностей, шляхів та verification_command.       │
│     Якщо команда перевірки фейкова (напр. `ls`), вимагає ремонт плану.       │
└──────────────────┬────────────────────────────────────────┬──────────────────┘
                   │ Режим Gauntlet (--target)              │ Режим R&D (--goal / --interactive)
                   ▼                                        ▼
┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
│   3А. MULTI-AGENT DEBATE ARENA      │  │   3Б. АДАПТИВНИЙ R&D РУШІЙ          │
│       (Ollama / Local GPU)          │  │       (engine/ + workforce/)        │
│                                     │  │                                     │
│  • Builder A (хірургічний патч)     │  │  • ExecutionControl: бюджети часу,  │
│  • Builder B (глибокий захист)      │  │    токенів та watchdog неактивності │
│  • Blind Critic (сліпий вибір       │  │  • ContextCompactor: утримання      │
│    між Candidate Alpha та Beta)     │  │    16K контексту для локальної LLM  │
│  • GauntletTestRunner (Pytest 6/6)  │  │  • AutonomousWorker (інструменти:   │
└──────────────────┬──────────────────┘  │    read_file, write_file, list_dir, │
                   │                     │    patch_file, run_shell)           │
                   │                     │  • SubtaskVerifier + Fingerprint    │
                   │                     │    Stall Detector -> Re-plan у хмарі│
                   │                     └──────────────────┬──────────────────┘
                   └───────────────────┬────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│         4. ЗАПИС СЕСІЙ, СНАПШОТИ СТАНУ ТА RICH TERMINAL HUD (hud/)           │
│  • .antigravity_runs/run-<ts>.jsonl      : Повний запис подій (права 0600)   │
│  • .antigravity_runs/run-<ts>.state.json : Снапшот DAG та виконаних задач    │
│  • --replay (відтворення в HUD) / --resume (продовження перерваного запуску) │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Життєвий цикл адаптивного виконання та детекції зациклення (`engine/`)

Замість жорсткого ліміту в 10 ходів воркера система використовує **відбитки прогресу (Progress Fingerprints)**:

```
Підзадача з DAG ──► AutonomousWorker виконує кроки (без ліміту кількості ходів)
                           │
                           ├── Інструмент `read_file` (>400 рядків) -> віддає рядки 1..400 + підказку діапазонів
                           ├── Інструмент `run_shell` -> зберігає перші 20 та останні 50 рядків (traceback не губиться)
                           └── Старі результати (>3 кроків тому) -> згортаються в 1-рядкове резюме
                           │
                           ▼
                    [SubtaskVerifier] запускає verification_command
                           │
              ┌────────────┴────────────┐
            PASS                      FAIL
              │                         │
              ▼                         ▼
    1. Git Commit чекпоінт    1. Обчислення Fingerprint = Hash(subtask + останні дії + помилка + git diff)
    2. Запис у .state.json    2. Чи змінився воркспейс або з'явилися нові дані?
    3. Перехід до наступної      • ТАК -> Передача повного traceback воркеру для наступної спроби
       вершини DAG               • НІ (fingerprint повторився) -> Стан REPLANNING:
                                   запит до Cloud Architect на перебудову DAG з урахуванням помилки
```

### Захисні запобіжники підзадач та процесу:
* `RD_MAX_REPAIR_ATTEMPTS` (за замовчуванням `6`) — ліміт спроб ремонту однієї підзадачі.
* `RD_SUBTASK_TIMEOUT_S` (за замовчуванням `1200` с) — максимальний час на одну підзадачу.
* `--model-idle-timeout` (`900` с) та `--tool-idle-timeout` (`300` с) — watchdog неактивності моделі або shell-команди (`0` вимикає).
* **Безпечне переривання (`Ctrl+C`)**: миттєво зупиняє дочірні процеси, закриває `.jsonl`-лог зі статусом `INTERRUPTED`, зберігає `.state.json` та залишає незавершені правки на диску для подальшого `--resume`.

---

## 4. Запис, Відтворення (`--replay`) та Відновлення (`--resume`)

Кожен R&D-запуск автоматично створює у `<workspace>/.antigravity_runs/` два файли з правами доступу `0600` (тільки для власника):
1. `run-<timestamp>.jsonl` — потік усіх подій (план, старт підзадач, виклики інструментів, відповіді моделі, верифікація, фінальний підсумок).
2. `run-<timestamp>.state.json` — поточний знімок стану (ціль, повний DAG, список уже верифікованих `verified` підзадач та остання подія).

```
Під час роботи:
  HybridCoordinator ──► RunRecorder ──► run-20261015.jsonl
                    ──► RunState    ──► run-20261015.state.json

Відтворення у терміналі:
  ./run.sh --replay /path/to/.antigravity_runs/run-20261015.jsonl --replay-speed 8

Миттєве відтворення без анімації (для CI / перевірки):
  ./run.sh --replay /path/to/.antigravity_runs/run-20261015.jsonl --replay-speed 0 --no-hud

Відновлення перерваної або зупиненої за бюджетом сесії:
  ./run.sh --resume /path/to/.antigravity_runs/run-20261015.jsonl --mode live
  (Пропускає всі вже верифіковані підзадачі DAG і продовжує з місця зупинки!)
```

---

## 5. Структура проєкту

```
antigravity-codex-orchestrator/
├── main.py                                 # Точка входу
├── run.sh                                  # Скрипт запуску
├── pyproject.toml                          # Конфігурація пакета та CLI-скриптів
├── docs/                                   # Архітектурні специфікації та плани
├── tools/
│   └── eval_suite.py                       # Незалежний бенчмарк-стенд (перевірка реальної поведінки коду)
├── antigravity_hybrid_orchestrator/
│   ├── cli.py                              # Розширений парсер CLI + інтерактивний REPL-режим
│   ├── coordinator.py                      # Головний координатор (Gauntlet + R&D + Resume + Replan)
│   ├── telemetry.py                        # Облік токенів і вартості (Codex / Claude / Gemini / Local GPU)
│   ├── privacy/
│   │   ├── ast_stripper.py                 # Видалення тіл функцій та констант з AST
│   │   ├── verifier.py                     # Сувора перевірка Zero-Knowledge інваріанта
│   │   └── workspace_privacy.py            # Фільтрація секретів і тіл при скануванні проєкту
│   ├── workspace/
│   │   └── scanner.py                      # Побудова скелета директорій та сигнатур
│   ├── architect/
│   │   ├── client.py                       # Мульти-бекендний клієнт: `luna` (codex), `haiku` (claude), `agy`
│   │   └── mock_client.py                  # Детермінований мок-клієнт для офлайн-режиму
│   ├── planner/
│   │   ├── dag.py                          # Граф виконання ExecutionPlanDAG та модель SubTask
│   │   └── validator.py                    # Перевірка якості плану та команд верифікації
│   ├── engine/
│   │   ├── control.py                      # Контролер бюджетів часу/токенів, таймаутів та сигналів
│   │   ├── loop.py                         # Адаптивний цикл самокорекції з детекцією зациклення
│   │   ├── run_state.py                    # Збереження та завантаження снапшотів `.state.json`
│   │   └── verifier.py                     # Запуск команд перевірки та збір доказів
│   ├── workforce/
│   │   ├── ollama_client.py                # Клієнт локальної Ollama з підтримкою `OLLAMA_NUM_CTX`
│   │   ├── context.py                      # Управління контекстним вікном, обрізання логів і стиснення історії
│   │   ├── autonomous_worker.py            # Багатоходовий агент з локальними інструментами
│   │   ├── worker.py                       # Виконавець підзадач DAG
│   │   ├── tools.py                        # Реєстр інструментів (read/write/patch/list/shell)
│   │   ├── builder_a.py                    # Builder A для Security Gauntlet
│   │   ├── builder_b.py                    # Builder B для Security Gauntlet
│   │   ├── blind_critic.py                 # Blind Critic для Security Gauntlet
│   │   └── mock_*.py                       # Повний набір моків для тестування без GPU та хмари
│   ├── tools/
│   │   └── executor.py                     # Безпечне виконання файлових операцій та shell-команд
│   ├── hud/
│   │   ├── dashboard.py                    # HUD для Security Gauntlet
│   │   ├── rd_dashboard.py                 # HUD для R&D DAG (таймери, бюджети, статуси, токен-стрім)
│   │   ├── recorder.py                     # Запис подій у `.jsonl` з правами `0600`
│   │   ├── replay.py                       # Відтворення записаних сесій у терміналі
│   │   ├── theme.py                        # Кольорова палітра та стилі HUD
│   │   ├── layout.py                       # Компонування екрана
│   │   ├── components.py                   # Візуальні панелі та таблиці
│   │   └── summary.py                      # Підсумковий звіт після завершення
│   ├── gauntlet/
│   │   ├── baselines.py                    # Вразливі базлайни для скидання (`--reset`)
│   │   ├── auth.py                         # Модуль автентифікації
│   │   ├── billing.py                      # Модуль білінгу
│   │   └── database.py                     # Модуль бази даних
│   └── runner/
│       └── test_runner.py                  # Запуск тестового полігону Gauntlet
└── tests/                                  # 693 тести
```

---

## 6. Швидкий старт та приклади використання

### Встановлення
```bash
uv venv
uv pip install -e .
```

### Основні сценарії запуску
```bash
# 1. Офлайн-прогін R&D-задачі в режимі mock (не потребує ні хмарних CLI, ні Ollama)
./run.sh --goal "implement an add function with unit tests" \
         --workspace /tmp/ws --mode mock --no-hud

# 2. Живий R&D-запуск (за замовчуванням архітектор: Claude Haiku 5.5, воркер: локальна Ollama)
./run.sh --goal "add CSV export and tests" \
         --workspace /path/to/project --mode live

# 3. Живий запуск через OpenAI Codex CLI (`luna`) або Antigravity (`agy`)
./run.sh --goal "refactor parser module" \
         --workspace /path/to/project --mode live --architect luna

# 4. Запуск із бюджетами часу, хмарних токенів та watchdog-таймаутами
./run.sh --goal "optimize matrix routines" \
         --workspace /path/to/project --mode live \
         --max-runtime 7200 --max-remote-tokens 50000 \
         --model-idle-timeout 300 --tool-idle-timeout 300

# 5. Інтерактивна сесія (постановка кількох задач по черзі; команди /help, /quit)
./run.sh --interactive --workspace /path/to/project --mode live

# 6. Відтворення записаного прогону в живому HUD або миттєво без HUD
./run.sh --replay /tmp/ws/.antigravity_runs/run-<timestamp>.jsonl
./run.sh --replay /tmp/ws/.antigravity_runs/run-<timestamp>.jsonl --replay-speed 0 --no-hud

# 7. Відновлення перерваного запуску зі збереженого стану (.state.json)
./run.sh --resume /tmp/ws/.antigravity_runs/run-<timestamp>.jsonl --mode live

# 8. Запуск полігону безпеки (Security Gauntlet)
./run.sh --mode mock --target all --reset --no-hud
```

---

## 7. Довідник прапорців CLI та змінних середовища

### Прапорці CLI
| Прапорець | За замовчуванням | Опис |
| :--- | :--- | :--- |
| `--mode {live,mock}` | `live` | Режим роботи: `live` (реальні моделі) або `mock` (тестові фікстури) |
| `--architect {haiku,luna,agy}` | `haiku` | Хмарний планувальник: `haiku` (`claude`), `luna` (`codex exec`), `agy` (`Antigravity`) |
| `--worker-model MODEL` | `gemma4:12b` | Локальна модель Ollama для воркера (напр. `gemma4:12b`, `qwen3.5:27b-q4_K_M`) |
| `--goal TEXT` | `None` | Опис R&D-задачі для побудови та виконання DAG |
| `--workspace PATH` | поточна тека | Робоча директорія проєкту |
| `--interactive` | `False` | Інтерактивний REPL-режим для виконання серії задач підряд |
| `--target {all,auth,billing,database}` | `all` | Ціль для режиму Security Gauntlet (коли `--goal` не вказано) |
| `--reset` | `False` | Відновити вразливі вихідні файли Gauntlet перед запуском |
| `--max-runtime SEC` | без ліміту | Загальний ліміт часу на весь запуск у секундах |
| `--max-remote-tokens INT` | без ліміту | Ліміт хмарних токенів на планування та репланування |
| `--model-idle-timeout SEC` | `900` | Таймаут очікування відповіді від моделі (`0` — вимкнути) |
| `--tool-idle-timeout SEC` | `300` | Таймаут неактивності локального інструменту / shell (`0` — вимкнути) |
| `--no-record` | `False` | Не записувати `.jsonl` та `.state.json` під час R&D-прогону |
| `--replay RUN_FILE` | `None` | Відтворити записаний `.jsonl`-файл у терміналі |
| `--replay-speed FLOAT` | `8.0` | Швидкість відтворення подій на секунду (`0` — максимально швидко) |
| `--resume RUN_FILE` | `None` | Продовжити перерваний прогін, пропускаючи вже верифіковані задачі |
| `--no-hud` | `False` | Вимкнути Rich Live HUD (текстовий режим для CI/скриптів) |
| `--refresh-rate FLOAT` | `8.0` | Частота оновлення HUD (FPS) |

### Змінні середовища
| Змінна | За замовчуванням | Призначення |
| :--- | :--- | :--- |
| `OLLAMA_MODEL` | `gemma4:12b` | Ідентифікатор локальної моделі Ollama (перевизначається `--worker-model`) |
| `OLLAMA_NUM_CTX` | `16384` (мін. `2048`) | Розмір контекстного вікна, що явно передається в кожному запиті до Ollama |
| `OLLAMA_TIMEOUT_S` | `900` | Таймаут HTTP-звернення до локального сервера Ollama |
| `RD_MAX_REPAIR_ATTEMPTS` | `6` | Максимальна кількість спроб виправлення однієї підзадачі |
| `RD_SUBTASK_TIMEOUT_S` | `1200` | Максимальний час виконання однієї підзадачі (у секундах) |

---

## 8. Тестування та незалежна оцінка (`eval_suite.py`)

Повний набір модульних та інтеграційних тестів (**712 тестів**):
```bash
.venv/bin/pytest -q
```

Запуск незалежного стенду оцінки (`tools/eval_suite.py`), який перевіряє згенерований оркестратором код у чистих тимчасових воркспейсах зовнішніми перевірочними скриптами (невидимими для самої моделі):
```bash
.venv/bin/python tools/eval_suite.py --mode mock
```


