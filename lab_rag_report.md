# МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ

ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ  
ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ  
«НОВОСИБИРСКИЙ НАЦИОНАЛЬНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ»

ФАКУЛЬТЕТ ИНФОРМАЦИОННЫХ ТЕХНОЛОГИЙ

Кафедра Систем информатики  
Направление подготовки 09.06.01 - Информатика и вычислительная техника

## ОТЧЕТ

Обучающегося Лебедева Николая Ивановича группы № 23214 курса 3  
Тема задания: Разработка алгоритма для реализации метода Rag

Новосибирск 2026

## Оглавление

1. Введение  
2. Описание реализации  
3. Тестирование и пример использования  
4. Заключение  
5. Список литературы

## Введение

Цель данной лабораторной работы - разработать алгоритм и программную реализацию метода RAG (Retrieval-Augmented Generation) для генерации ответов на вопросы по онтологии предметной области «Микроэлектроника цифровых интегральных схем».

RAG представляет собой подход, при котором ответ языковой модели формируется не только на основе внутренних знаний модели, но и на основе найденного внешнего контекста. В данной работе внешним источником знаний выступает онтология, сохраненная в файле `graph.json`. Она содержит классы, экземпляры, атрибуты и связи предметной области.

Для реализации цели были поставлены следующие задачи:

- загрузить онтологию из JSON-представления графа;
- преобразовать узлы, атрибуты и связи онтологии в короткие текстовые фрагменты;
- построить эмбеддинги текстовых фрагментов;
- выполнить семантический поиск релевантных узлов по вопросу пользователя;
- сформировать промпт для языковой модели;
- выполнить второй поиск по черновому ответу модели;
- сформировать финальный ответ на основе объединенного набора найденных фрагментов.

Работа продолжает предыдущие лабораторные работы: в ЛР 2 был разработан драйвер для Neo4j, в ЛР 3 - репозиторий для редактирования онтологии, в ЛР 4 - репозиторий для построения эмбеддингов, в ЛР 5 - описана предметная область микроэлектроники, а в ЛР 6 - выполнена семантическая разметка корпуса.

## Описание реализации

Для выполнения лабораторной работы был разработан модуль `rag_graph.py`. Он реализует загрузку онтологии, преобразование графовых данных в текст, построение индекса эмбеддингов, поиск релевантных фрагментов и двухэтапный RAG-пайплайн.

В качестве входных данных используется файл `graph.json`. В нем основная структура графа находится в двух полях:

- `nodes` - основные узлы онтологии;
- `arcs` - связи между узлами.

В файле содержится 71 основной узел и 192 связи. При этом некоторые служебные узлы свойств, например `DatatypeProperty` и `ObjectProperty`, встречаются внутри связей как `start_node` и `end_node`. Поэтому при загрузке графа модуль собирает не только основные узлы, но и вложенные property-узлы. Они используются для корректного отображения названий атрибутов и отношений, но по умолчанию не включаются как самостоятельные результаты поиска, чтобы в выдачу не попадали служебные элементы вроде «Название» или «режим_работы».

Также в экспортированном графе сохранены результаты семантической разметки, выполненной в лабораторной работе №6. У части узлов заполнено поле `text_mentions`, где указаны позиции и идентификаторы текстовых ресурсов, в которых был размечен соответствующий объект онтологии. В текущем файле `graph.json` такие упоминания есть у 39 узлов, всего найдено 80 упоминаний. Сами полные тексты корпуса в JSON не сохранены, но связь между узлами онтологии и размеченными местами текста используется как дополнительный признак при формировании текстового фрагмента.

### Загрузка графа

Загрузка графа выполняется функцией `load_ontology_graph()`. Она читает JSON-файл, извлекает узлы и связи, а затем приводит связи к внутренней структуре `GraphArc`.

Листинг 1 - Фрагмент функции загрузки графа

```python
def load_ontology_graph(path: str | Path) -> OntologyGraph:
    with open(path, encoding="utf-8") as graph_file:
        raw_graph = json.load(graph_file)

    raw_arcs = raw_graph.get("arcs", [])
    arcs = [
        GraphArc(
            id=arc.get("id"),
            uri=arc.get("data", {}).get("uri") or arc.get("type") or "",
            source=arc.get("source") or arc.get("data", {}).get("start_node", {}).get("id") or "",
            target=arc.get("target") or arc.get("data", {}).get("end_node", {}).get("id") or "",
            raw=arc,
        )
        for arc in raw_arcs
        if arc.get("source") and arc.get("target")
    ]

    return OntologyGraph(raw_graph.get("nodes", []), arcs)
```

Класс `OntologyGraph` строит словари для быстрого доступа:

- `nodes_by_uri` - поиск узла по URI;
- `outgoing` - исходящие связи узла;
- `incoming` - входящие связи узла.

Эти структуры используются при построении текстового описания каждого элемента онтологии.

### Преобразование узла в текстовый фрагмент

Основная задача первой фазы заключается в том, чтобы представить каждый узел онтологии в виде короткого понятного текста. Такой текст должен содержать название, описание, тип узла, значения атрибутов и ближайшие связи.

Листинг 2 - Фрагмент функции преобразования узла в текст

```python
def node_to_text(node: Dict[str, Any], graph: OntologyGraph, max_relations: int = 24) -> str:
    data = node.get("data", {})
    params_values = data.get("params_values", {})
    node_id = node.get("id") or data.get("uri", "")
    label = extract_label(node) or graph.label_for_uri(node_id)
    comment = extract_comment(node)
    kind = graph.node_kind(node)

    lines: List[str] = []
    if label:
        lines.append(f"Название: {label}")
    if comment:
        lines.append(f"Описание: {comment}")
    if kind:
        lines.append(f"Тип узла: {kind}")
```

Если узел встречался в размеченных текстах корпуса, в его фрагмент добавляется информация о количестве упоминаний. Это связывает RAG-поиск не только с онтологической структурой, но и с результатами семантической разметки.

Листинг 3 - Обработка сведений о разметке текста

```python
def _text_mentions_summary(node: Dict[str, Any]) -> List[str]:
    mentions = node.get("data", {}).get("text_mentions") or []
    if not mentions:
        return []

    lines = [f"Упоминаний в размеченных текстах: {len(mentions)}"]
    return lines
```

Для атрибутов используется расшифровка URI в человекочитаемое название. Например, если у экземпляра есть параметр, ключом которого является URI свойства, то функция находит соответствующий property-узел и подставляет его русское название.

Листинг 4 - Обработка атрибутов узла

```python
for param_uri, value in params_values.items():
    if _is_service_param(param_uri):
        continue
    param_label = graph.label_for_uri(param_uri)
    formatted_value = _format_property_value(value)
    if param_label and formatted_value:
        lines.append(f"{param_label}: {formatted_value}")
```

Связи также преобразуются в читаемый вид. Для стандартных RDF/RDFS-связей используются специальные названия:

- `rdf:type` - «Имеет тип»;
- `rdfs:subClassOf` - «Является подклассом»;
- `rdfs:domain` - «Область определения»;
- `rdfs:range` - «Область значений».

Остальные связи интерпретируются как объектные свойства предметной области.

### Построение эмбеддингов

Для построения эмбеддингов используется класс `EmbeddingRepository`, разработанный в предыдущей лабораторной работе. Основная модель - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. Она поддерживает русский и английский языки, что важно для онтологии, где названия часто представлены в двух языках.

Для демонстрации без установленных зависимостей в модуле также реализован резервный backend `FallbackEmbeddingBackend`. Он использует простую TF-IDF-векторизацию и позволяет проверить работу алгоритма даже без `sentence-transformers`.

Листинг 5 - Выбор backend для эмбеддингов

```python
def create_embedding_backend(preferred_model: str = SentenceTransformerBackend.name):
    try:
        return SentenceTransformerBackend(preferred_model)
    except Exception as exc:
        backend = FallbackEmbeddingBackend()
        backend.name = f"{backend.name} ({exc.__class__.__name__}: {exc})"
        return backend
```

После преобразования узлов в тексты функция `build_index()` строит векторный индекс. По умолчанию в индекс попадают основные узлы онтологии, а property-узлы используются как справочник названий.

Листинг 6 - Построение индекса

```python
def build_index(graph_path: str | Path = "graph.json", include_properties: bool = False) -> OntologyRagIndex:
    graph = load_ontology_graph(graph_path)
    source_nodes = list(graph.nodes_by_uri.values()) if include_properties else graph.main_nodes

    fragments = []
    for node in source_nodes:
        text = node_to_text(node, graph)
        if not text.strip():
            continue
        fragments.append(TextFragment(...))

    backend = create_embedding_backend()
    backend.fit([fragment.text for fragment in fragments])
    vectors = backend.embed([fragment.text for fragment in fragments])
    return OntologyRagIndex(graph, fragments, vectors, backend)
```

### Семантический поиск

Поиск выполняется методом `search()` класса `OntologyRagIndex`. Сначала вычисляется эмбеддинг запроса пользователя, затем он сравнивается с эмбеддингами всех фрагментов. В качестве метрики используется косинусное сходство. Дополнительно применяется небольшой приоритет точного совпадения по названию или числовому токену: например, если в вопросе есть «7404», узел «Микросхема 7404» поднимается выше похожих, но других микросхем.

Листинг 7 - Фрагмент функции поиска

```python
def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
    query_vec = normalize_vectors(self.backend.embed([query]))
    scores = np.dot(self.vectors, query_vec[0])
    ranked_indexes = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(score=float(scores[idx]), fragment=self.fragments[int(idx)])
        for idx in ranked_indexes
    ]
```

### Генерация ответа

Для генерации ответа реализованы несколько клиентов: общий `LLMClient`, отдельный `YandexGPTClient` и `OpenAILLMClient`. Для лабораторной работы основными учебными вариантами являются YandexGPT API и локальная Qwen-модель, так как они соответствуют предложенным в задании способам: использование доступного API или запуск модели через Python. Генерация выполняется программно из Python-кода, без использования веб-интерфейса.

В демонстрационном скрипте используется следующий порядок:

1. Если заданы `YANDEX_FOLDER_ID` и `YANDEX_API_KEY` или `YANDEX_IAM_TOKEN`, используется `YandexGPTClient`.
2. Если задана переменная окружения `OPENAI_API_KEY`, используется `OpenAILLMClient`.
3. Если API-ключи не заданы, используется общий клиент `LLMClient`.

Общий клиент поддерживает три режима:

1. API-режим. Если заданы переменные окружения `RAG_API_URL`, `RAG_API_KEY` и `RAG_API_MODEL`, то запрос отправляется во внешнюю языковую модель.
2. Локальная модель. Если API-ключ не задан, система пробует использовать локальную text-generation модель через библиотеку `transformers`. По умолчанию используется instruction-модель `Qwen/Qwen2.5-1.5B-Instruct`, но ее можно заменить через `RAG_LOCAL_MODEL`.
3. Резервный режим. Если локальная модель недоступна или возвращает некачественный текст, система не падает, а формирует извлекающий ответ по найденному контексту и сохраняет готовый промпт.

Перед передачей промпта в локальную модель выполняется ограничение длины входа по максимальному числу токенов модели. Это необходимо, потому что небольшие модели имеют ограниченное контекстное окно, например около 1024 токенов. Также выполняется простая проверка качества сгенерированного текста: если модель возвращает слишком короткий, повторяющийся или нечитаемый ответ, используется резервный извлекающий ответ на основе найденных фрагментов.

Листинг 8 - Формирование промпта

```python
def build_prompt(question: str, contexts: Sequence[str]) -> str:
    context_text = "\n\n".join(f"[{idx + 1}]\n{text}" for idx, text in enumerate(contexts))
    return (
        "Дай ответ на данный вопрос, используя информацию из текста. "
        "Если в тексте недостаточно данных, укажи это явно.\n\n"
        f"Вопрос: {question}\n\n"
        f"Текст:\n{context_text}\n\n"
        "Ответ:"
    )
```

Такой подход соответствует постановке задачи: ChatGPT, YandexGPT или другая языковая модель не используются через веб-интерфейс. Генерация выполняется либо локальной моделью через Python, либо программным API.

Листинг 9 - Фрагмент клиента YandexGPT API

```python
class YandexGPTClient:
    def __init__(self):
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.model_name = os.getenv("YANDEX_MODEL", "yandexgpt-lite")

    def generate(self, question: str, contexts: Sequence[str]) -> LLMResult:
        prompt = build_prompt(question, contexts)
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers={"Authorization": f"Api-Key {self.api_key}"},
            json={
                "modelUri": f"gpt://{self.folder_id}/{self.model_name}",
                "messages": [{"role": "user", "text": prompt}],
            },
        )
        answer = response.json()["result"]["alternatives"][0]["message"]["text"]
        return LLMResult(answer=answer, prompt=prompt, used_api=True, mode="yandex-api")
```

Листинг 10 - Фрагмент клиента OpenAI API

```python
class OpenAILLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, contexts: Sequence[str]) -> LLMResult:
        prompt = build_prompt(question, contexts)
        response = self.client.responses.create(
            model=self.model,
            instructions="Отвечай на русском языке только на основе контекста.",
            input=prompt,
            max_output_tokens=600,
        )
        return LLMResult(
            answer=response.output_text.strip(),
            prompt=prompt,
            used_api=True,
            mode="openai-api",
        )
```

### Двухэтапный RAG-пайплайн

Основной алгоритм реализован в классе `RagPipeline`. Он выполняет три фазы:

1. По вопросу пользователя находятся N релевантных фрагментов.
2. На основе этих фрагментов формируется черновой ответ.
3. По эмбеддингу чернового ответа выполняется второй поиск M фрагментов, после чего уникальное объединение N + M фрагментов используется для финального ответа.

Листинг 11 - Фрагмент RAG-пайплайна

```python
def answer(self, question: str, n: int = 5, m: int = 3) -> RagResult:
    first_results = self.index.search(question, top_k=n)
    first_contexts = [result.fragment.text for result in first_results]
    draft = self.llm.generate(question, first_contexts)

    second_results = self.index.search(draft.answer, top_k=m)
    all_results = self._merge_results(first_results, second_results)
    final_contexts = [result.fragment.text for result in all_results]
    final = self.llm.generate(question, final_contexts)

    return RagResult(...)
```

Дедупликация выполняется по идентификатору узла, поэтому один и тот же узел не попадает в финальный контекст дважды.

## Тестирование и пример использования

Для демонстрации был добавлен файл `example/rag_graph_example.py`. Он строит индекс по файлу `graph.json`, задает вопрос и запускает полный RAG-пайплайн.

Команда запуска:

```bash
python example/rag_graph_example.py
```

Для запуска через OpenAI API необходимо установить SDK и задать ключ:

```powershell
pip install openai
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-5.4-mini"
python example\rag_graph_example.py
```

Для запуска через YandexGPT API необходимо задать идентификатор каталога и ключ:

```powershell
$env:YANDEX_FOLDER_ID="..."
$env:YANDEX_API_KEY="..."
$env:YANDEX_MODEL="yandexgpt-lite"
python example\rag_graph_example.py
```

Для локального запуска Qwen без API-ключей достаточно оставить переменные API пустыми. При необходимости можно выбрать другую Qwen-модель:

```powershell
$env:RAG_LOCAL_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
python example\rag_graph_example.py
```

В демонстрационном примере используется вопрос:

```text
Какие параметры и связи есть у микросхемы 7404?
```

Ожидаемый вывод включает:

- количество основных узлов;
- количество связей;
- количество текстовых фрагментов в индексе;
- название используемой модели эмбеддингов;
- результаты первой фазы поиска;
- черновой ответ;
- результаты второй фазы поиска;
- финальный ответ или финальный промпт.

Если зависимости `sentence-transformers` не установлены, программа использует резервный TF-IDF-backend для поиска. Если ключ OpenAI не задан, программа пробует альтернативные режимы генерации. Если локальная модель недоступна или генерирует некачественный текст, выводится извлекающий ответ и готовый промпт для языковой модели. Благодаря этому можно проверить всю логику загрузки онтологии, построения контекста и поиска без внешних сервисов.

Пример фрагмента, который формируется для узла онтологии:

```text
Название: Микросхема 7404
Описание: Описание
Тип узла: Экземпляр
Имеет тип: Интегральная схема
Упоминаний в размеченных текстах: 2
частота_работы: 25
Название: 7404 hex inverter
технологический_узел_нм: 5000
напряжение_питания_В: 5.0
тип_ИС: logic
содержит_прибор: Пятинанометровая нмос
реализует: Инвертор
схема_характеризуется: Напряжение питания
```

Таким образом, языковая модель получает не исходный JSON, а компактный текстовый контекст, пригодный для генерации ответа.

## Заключение

В результате проведенной работы были выполнены все поставленные задачи:

- реализована загрузка онтологии из файла `graph.json`;
- выполнена нормализация узлов и связей графа;
- реализовано преобразование узлов, атрибутов и отношений в текстовые фрагменты;
- построен индекс эмбеддингов для семантического поиска;
- реализован поиск релевантных фрагментов по вопросу пользователя;
- реализован двухэтапный RAG-пайплайн с повторным поиском по черновому ответу;
- реализована генерация через YandexGPT API, OpenAI API, локальную Qwen-модель `transformers` и безопасный резервный режим;
- подготовлен демонстрационный пример запуска.

Разработанный алгоритм позволяет использовать онтологию предметной области как базу знаний для генерации ответов. Такой подход связывает результаты предыдущих лабораторных работ: графовую модель Neo4j, репозиторий онтологии, эмбеддинги текстов и семантическую разметку корпуса.

## Список литературы

1. Reimers N., Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. 2019.
2. Документация библиотеки Sentence Transformers. URL: https://www.sbert.net/
3. Lewis P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2020.
4. Документация Neo4j Python Driver. URL: https://neo4j.com/docs/python-manual/current/
5. Harris S., Harris D. Digital Design and Computer Architecture: RISC-V Edition. Morgan Kaufmann, 2021.
6. Документация OpenAI Responses API. URL: https://platform.openai.com/docs/api-reference/responses
7. Документация Yandex Foundation Models API. URL: https://yandex.cloud/ru/docs/foundation-models/text-generation/api-ref/TextGeneration/completion
