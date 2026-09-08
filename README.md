# Автоматическая обработка текста

Дущенко Даниил Александрович, К3341, Университет ИТМО. Преподаватель: Гусарова Наталия Федоровна. Санкт-Петербург, 2026.

Шесть лабораторных работ по заданию `assignment/requirements.txt`.

Все шесть работ выполнены: код, данные, фактические результаты и отчёты включены в репозиторий. Проверка свежего клона с отключённым преобразованием окончаний строк прошла: 64 теста кода, 22 проверки комплектности, структура всех шести отчётов и контрольные суммы замороженных данных. Использовано установленное окружение Python; повторные запросы к модели не выполнялись. Подробности — [output/clone_verification.json](output/clone_verification.json).

| Работа и инструкция запуска | Отчёт DOCX | Отчёт PDF |
|---|---|---|
| [1. Один текст — разные представления](lab01_representations) | [DOCX](lab01_representations/report.docx) | [PDF](lab01_representations/report.pdf) |
| [2. Разметка неоднозначной категории](lab02_annotation) | [DOCX](lab02_annotation/report.docx) | [PDF](lab02_annotation/report.pdf) |
| [3. Классика против LLM](lab03_classification) | [DOCX](lab03_classification/report.docx) | [PDF](lab03_classification/report.pdf) |
| [4. Промпт как программа](lab04_prompt_program) | [DOCX](lab04_prompt_program/report.docx) | [PDF](lab04_prompt_program/report.pdf) |
| [5. Мини-RAG с доказательствами](lab05_mini_rag) | [DOCX](lab05_mini_rag/report.docx) | [PDF](lab05_mini_rag/report.pdf) |
| [6. AI-детектив](lab06_ai_detective) | [DOCX](lab06_ai_detective/report.docx) | [PDF](lab06_ai_detective/report.pdf) |

## Среда

Python 3.12, локальный LM Studio (`http://localhost:1234/v1`), `mistralai/ministral-3-14b-reasoning`. Компьютер: RTX 5060 Ti 16 ГБ, ОЗУ 32 ГБ. Модели и виртуальное окружение в Git не включаются.

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

Отдельные инструкции запуска и фактические результаты находятся в папках работ. Длительные эксперименты сохраняют ответы по мере выполнения и допускают продолжение.

Точные версии установленных пакетов сохранены в `requirements-lock.txt`. Для LLM использован контекст 24 576 токенов, квантование Q4_K_M; embeddings `intfloat/multilingual-e5-small` вычисляются на CPU. Локальной RTX 5060 Ti хватило для этой конфигурации; Google Colab не требуется. Плата за внешний API отсутствует, стоимость электроэнергии не измерялась.

Проверка комплектности сохранённых результатов без повторных запросов к LLM:

```powershell
.venv/Scripts/python.exe -I scripts/verify_submission.py
.venv/Scripts/python.exe -I scripts/verify_report_layout.py
```

Проверки создают `output/submission_verification.json` и `output/report_layout_verification.json`, возвращая ненулевой код при недостающих артефактах или нарушениях проверяемой структуры отчётов. Структурная проверка дополняет визуальный просмотр страниц. Для запуска интерфейса шестой работы:

```powershell
.venv/Scripts/python.exe -I -m streamlit run lab06_ai_detective/app.py --server.headless true
```

Исходники отчётов — `report.md`, генератор DOCX — `scripts/build_reports.py`, экспорт через установленный Microsoft Word — `scripts/export_reports.ps1`. Обезличенный шаблон находится в `assignment/report_template.docx`. Рабочая копия перенесена в `E:/AOT-work` после обнаружения ошибок диска D:; подробности в [WORKSPACE.md](WORKSPACE.md).

## Происхождение результатов

Код и аналитические материалы подготовлены с помощью ИИ. Независимые участники командных заданий моделируются изолированными агентами/запросами, а не реальными людьми. Первичные разметки сохраняются без исправлений. Это ограничение учитывается при интерпретации согласованности. Метрики вычисляются из сохранённых результатов реальных запусков; отсутствующие ошибки или ручные действия не выдумываются.

Оформление отчётов: предоставленный образец ИТМО и применимые требования ГОСТ 7.32–2017; DOCX и PDF в каждой папке.
