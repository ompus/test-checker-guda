import argparse
import pandas as pd
import asyncio
import json
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://polza.ai/api/v1",
    api_key="pza_AOcqtNfGj4hJgc_281xXNFThsc8_eDGe" # имеет смысл сделать как аргумент при вводе в командную строку
    )

SYSTEM_PROMPT = """Ты проверяешь ответы студентов на вопросы теста.

Для каждого ответа:
1. Самостоятельно реши вопрос и определи правильный ответ.
2. Сначала напиши rationale из 1-2 предложений, объясняющее выставленную оценку.
3. Затем выставь mark, строго соответствующий rationale.
4. Перед возвратом результата проверь, что mark не противоречит rationale.

Шкала оценивания:
- Численные вопросы, вопросы с одним вариантом и вопросы с несколькими вариантами оцениваются только как 0 или 100.
- Для таких вопросов 100 означает полностью правильный ответ, а 0 — неправильный или неполный ответ.
- Математически эквивалентные численные формы допустимы, если вопрос явно не требует определённого формата.
- Для действительно развёрнутых вопросов используй:
  - 0 — ответ неверный или не относится к вопросу;
  - 50 — ответ содержит правильную основную идею, но является неполным или содержит существенную ошибку;
  - 100 — ответ полностью и корректно отвечает на вопрос.
- Оценка 50 запрещена для численных и тестовых вопросов.

Не изменяй переданные id_question и index. Не пропускай ответы и не добавляй ответы, которых нет во входных данных."""

VALIDATOR_SYSTEM_PROMPT = """Ты проверяешь только согласованность текстового обоснования rationale и числовой оценки mark.

Для каждого переданного ответа:
1. Определи, считает ли rationale ответ студента правильным, частично правильным или неправильным.
2. Проверь, соответствует ли первоначальный mark этому выводу.
3. Если rationale и mark согласованы, сохрани первоначальный mark как final_mark.
4. Если они явно противоречат друг другу, исправь final_mark в соответствии с rationale.
5. Кратко объясни проверку в validator_rationale.

Сопоставляй вывод rationale с оценкой так:
- правильный ответ соответствует mark=100;
- частично правильный ответ соответствует mark=50;
- неправильный ответ соответствует mark=0.

Установи contradiction_found=true только при явном противоречии между rationale и mark.

Не решай исходный вопрос и не оценивай правильность самого rationale.
Во входных данных намеренно отсутствуют исходный вопрос и ответ студента.
Не изменяй согласованный mark, даже если тебе кажется, что исходное рассуждение может быть ошибочным.

Не изменяй id_question и index. Не пропускай ответы и не добавляй новые."""

ANSWER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id_question": {"type": "integer"},
            "index": {"type": "integer"},
            "rationale": {"type": "string", "minLength": 1},
            "mark": {"type": "integer", "enum": [0, 50, 100]}
        },
        "required": ["id_question", "index", "rationale", "mark"],
        "additionalProperties": False
    },
    "minItems": 1
}

VALIDATOR_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id_question": {"type": "integer"},
            "index": {"type": "integer"},
            "contradiction_found": {"type": "boolean"},
            "validator_rationale": {"type": "string", "minLength": 1},
            "final_mark": {"type": "integer", "enum": [0, 50, 100]}
        },
        "required": [
            "id_question",
            "index",
            "contradiction_found",
            "validator_rationale",
            "final_mark"
        ],
        "additionalProperties": False
    },
    "minItems": 1
}

async def get_completion(
    prompt,
    system_prompt=SYSTEM_PROMPT,
    schema=ANSWER_SCHEMA,
    schema_name="answer_schema"
):
  response = await client.chat.completions.create(
    model="deepseek/deepseek-v4-flash",
    temperature = 0,
    response_format =  {
    "type": "json_schema",
    "json_schema": {
      "name": schema_name,
      "strict": True,
      "schema": schema,
    }
  },
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}]
  )
  return response.choices[0].message.content

async def get_validation(prompt):
    return await get_completion(
        prompt,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        schema=VALIDATOR_SCHEMA,
        schema_name="validator_schema"
    )

def validate_initial_result(result, expected_pairs):
    result_json = json.loads(result)
    if not isinstance(result_json, list):
        raise ValueError("Первичный ответ модели должен быть массивом")

    required_fields = {"id_question", "index", "rationale", "mark"}
    actual_pairs = []
    for item in result_json:
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError("Первичный объект модели не соответствует ожидаемой структуре")
        if (
            not isinstance(item["id_question"], int)
            or isinstance(item["id_question"], bool)
            or not isinstance(item["index"], int)
            or isinstance(item["index"], bool)
        ):
            raise ValueError("id_question и index должны быть целыми числами")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise ValueError("rationale должен быть непустой строкой")
        if item["mark"] not in {0, 50, 100}:
            raise ValueError("mark должен быть равен 0, 50 или 100")
        actual_pairs.append((item["id_question"], item["index"]))

    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Первичная модель вернула повторяющиеся id_question/index")
    if set(actual_pairs) != expected_pairs:
        raise ValueError("Первичная модель пропустила ответы или добавила неизвестные id_question/index")

    return result_json

async def get_initial_batch(prompt, expected_pairs):
    for attempt in range(2):
        try:
            result = await get_completion(prompt)
            return validate_initial_result(result, expected_pairs)
        except Exception as error:
            if attempt == 1:
                print(f"Ошибка первичной проверки пакета после повторной попытки: {error}")
    return None

def validate_validator_result(result, expected_pairs):
    result_json = json.loads(result)
    if not isinstance(result_json, list):
        raise ValueError("Ответ валидатора должен быть массивом")

    required_fields = {
        "id_question",
        "index",
        "contradiction_found",
        "validator_rationale",
        "final_mark"
    }
    actual_pairs = []
    for item in result_json:
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError("Объект валидатора не соответствует ожидаемой структуре")
        if (
            not isinstance(item["id_question"], int)
            or isinstance(item["id_question"], bool)
            or not isinstance(item["index"], int)
            or isinstance(item["index"], bool)
        ):
            raise ValueError("id_question и index должны быть целыми числами")
        if not isinstance(item["contradiction_found"], bool):
            raise ValueError("contradiction_found должен быть boolean")
        if not isinstance(item["validator_rationale"], str) or not item["validator_rationale"].strip():
            raise ValueError("validator_rationale должен быть непустой строкой")
        if item["final_mark"] not in {0, 50, 100}:
            raise ValueError("final_mark должен быть равен 0, 50 или 100")
        actual_pairs.append((item["id_question"], item["index"]))

    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Валидатор вернул повторяющиеся id_question/index")
    if set(actual_pairs) != expected_pairs:
        raise ValueError("Валидатор пропустил ответы или добавил неизвестные id_question/index")

    return result_json

async def validate_batch(prompt, expected_pairs):
    for attempt in range(2):
        try:
            result = await get_validation(prompt)
            return validate_validator_result(result, expected_pairs)
        except Exception as error:
            if attempt == 1:
                print(f"Ошибка валидации пакета после повторной попытки: {error}")
    return None

async def main():
    question_block_size = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default='example.xlsx')
    args = parser.parse_args()

    try:
        data = pd.read_excel(args.path)
    except FileNotFoundError:
        print("Файл с ответами по указанному пути отсутствует (по умолчанию example.xlsx)")
        quit()

    print("Файл с ответами найден")

    # Количество вопросов
    num_of_questions = (len(data.columns) - 7) // 2

    print(f"Кол-во вопросов в тесте = {num_of_questions}")

    # Список уникальных вопросов со всех колонок
    questions = pd.DataFrame(data.iloc[:,7:-1:2]).stack().reset_index()
    questions = pd.unique(questions.iloc[:, -1])

    # Замена в оригинальном DataFrame вопросов на их id
    data = data.replace(questions, range(len(questions)))


    # Генерация таблицы вида "student_id - id_question - answer - num_question"
    # Написать более оптимальный вариант
    answers = pd.DataFrame(data.iloc[:, 7:9].reset_index())
    answers = answers.rename(columns={'Вопрос 1': 'id_question', 'Ответ 1':'answer'})
    answers['num'] = 1

    for i in range(num_of_questions - 1):
        temp = pd.DataFrame(data.iloc[:, 7+2*(i+1):9+2*(i+1)].reset_index())
        temp = temp.rename(columns={f'Вопрос {i+2}': 'id_question', f'Ответ {i+2}':'answer'})
        temp['num'] = i + 2
        answers = pd.concat([answers, temp], axis = 0)

    answers['initial_mark'] = pd.NA
    answers['initial_rationale'] = ""
    answers['contradiction_found'] = pd.NA
    answers['validator_rationale'] = ""
    answers['final_mark'] = pd.NA
    answers['validation_status'] = "not_processed"

    count = len(questions)
    i = 1
    prompts = []
    prompt_question_ids = []

    message = ""

    # Формирование промптов
    while i < 30: #count
        message = """Ниже переданы вопросы и массивы ответов студентов на них.
Для каждого входного объекта верни объект с теми же id_question и index, добавив rationale и mark.
Если результат один, всё равно верни его как массив из одного объекта.

"""
        question_ids = []
        for j in range(question_block_size):
            if (i < count):
                question_ids.append(i)
                message = message + "Вопрос: " + questions[i] + "\n"
                message = message + answers[answers['id_question'] == i][["id_question", "index", "answer"]].to_json(orient="records") + "\n"
            i = i + 1
        #if i < 40:
            #print(message)
        prompts.append(message)
        prompt_question_ids.append(question_ids)

    print("Промпты сформированы")

    initial_jobs = []
    for prompt, question_ids in zip(prompts, prompt_question_ids):
        expected_pairs = {
            (int(row.id_question), int(row.index))
            for question_id in question_ids
            for row in answers[answers['id_question'] == question_id].itertuples()
        }
        initial_jobs.append((prompt, expected_pairs, question_ids))

    # Асинхронное выполнение с проверкой и одним повтором повреждённых пакетов
    results = await asyncio.gather(*[
        get_initial_batch(prompt, expected_pairs)
        for prompt, expected_pairs, _ in initial_jobs
    ])

    print("Модель оценила вопросы, запускаем consistency checker...")

    # Заполнение таблицы answers первоначальными оценками модели
    successful_question_batches = []
    for (_, expected_pairs, question_ids), res_json in zip(initial_jobs, results):
        if res_json is None:
            for id_question, index in expected_pairs:
                answer_mask = (
                    (answers['id_question'] == id_question)
                    & (answers['index'] == index)
                )
                answers.loc[answer_mask, 'validation_status'] = "validation_error"
            continue

        successful_question_batches.append(question_ids)
        for i in range(len(res_json)):
            answer_mask = (
                (answers['id_question'] == res_json[i]['id_question'])
                & (answers['index'] == res_json[i]['index'])
            )
            answers.loc[answer_mask, 'initial_mark'] = res_json[i]['mark']
            answers.loc[answer_mask, 'initial_rationale'] = res_json[i]['rationale']

    validation_jobs = []
    for question_ids in successful_question_batches:
        # Consistency checker намеренно не получает исходный вопрос или ответ студента.
        validator_message = """Ниже переданы результаты первоначальной проверки.
Проверь только согласованность initial_rationale и initial_mark для каждой пары id_question/index.

"""
        expected_pairs = set()
        for question_id in question_ids:
            question_answers = answers[answers['id_question'] == question_id][[
                "id_question",
                "index",
                "initial_rationale",
                "initial_mark"
            ]]
            expected_pairs.update(
                (int(row.id_question), int(row.index))
                for row in question_answers.itertuples()
            )
            validator_message = validator_message + question_answers.to_json(orient="records") + "\n"
        validation_jobs.append((validator_message, expected_pairs))

    validator_results = await asyncio.gather(*[
        validate_batch(prompt, expected_pairs)
        for prompt, expected_pairs in validation_jobs
    ])

    print("Consistency checker проверил соответствие rationale и mark, выставляем окончательные оценки...")

    for (_, expected_pairs), validator_result in zip(validation_jobs, validator_results):
        if validator_result is None:
            for id_question, index in expected_pairs:
                answer_mask = (
                    (answers['id_question'] == id_question)
                    & (answers['index'] == index)
                )
                answers.loc[answer_mask, 'validation_status'] = "validation_error"
            continue

        for item in validator_result:
            answer_mask = (
                (answers['id_question'] == item['id_question'])
                & (answers['index'] == item['index'])
            )
            initial_mark = answers.loc[answer_mask, 'initial_mark'].iloc[0]
            marks_differ = pd.isna(initial_mark) or initial_mark != item['final_mark']
            contradiction_found = item['contradiction_found'] or marks_differ

            answers.loc[answer_mask, 'contradiction_found'] = contradiction_found
            answers.loc[answer_mask, 'validator_rationale'] = item['validator_rationale']
            answers.loc[answer_mask, 'final_mark'] = item['final_mark']
            answers.loc[answer_mask, 'validation_status'] = (
                "corrected" if contradiction_found else "accepted"
            )

    # Финальная таблица со строками ФИ - итоговая оценка
    final_rows = []

    for i in range(len(data)):
        student_marks = pd.to_numeric(
            answers[answers['index']==i]["final_mark"],
            errors="coerce"
        )
        final_rows.append({
            "ФИО": data['Фамилия'][i] + ' ' + data['Имя'][i],
            "Оценка за тест": round(student_marks.mean(), 2)
        })

    final_table = pd.DataFrame(final_rows)

    audit_table = answers.copy()
    audit_table["ФИО"] = audit_table["index"].map(
        lambda index: data['Фамилия'][index] + ' ' + data['Имя'][index]
    )
    audit_table["Вопрос"] = audit_table["id_question"].map(
        lambda id_question: questions[int(id_question)]
    )
    audit_table = audit_table.rename(columns={
        "id_question": "ID вопроса",
        "num": "Номер вопроса",
        "answer": "Ответ",
        "initial_mark": "Первоначальная оценка",
        "initial_rationale": "Первоначальное обоснование",
        "contradiction_found": "Найдено противоречие",
        "validator_rationale": "Обоснование валидатора",
        "final_mark": "Окончательная оценка",
        "validation_status": "Статус проверки"
    })
    audit_table = audit_table[[
        "ФИО",
        "ID вопроса",
        "Номер вопроса",
        "Вопрос",
        "Ответ",
        "Первоначальная оценка",
        "Первоначальное обоснование",
        "Найдено противоречие",
        "Обоснование валидатора",
        "Окончательная оценка",
        "Статус проверки"
    ]]

    with pd.ExcelWriter("results.xlsx") as writer:
        final_table.to_excel(writer, sheet_name="Итоги", index=False)
        audit_table.to_excel(writer, sheet_name="Аудит", index=False)

    print("Файл с итоговыми оценками готов!")

if __name__ == "__main__":
    asyncio.run(main())
