import argparse
import pandas as pd
import asyncio
import json
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://polza.ai/api/v1",
    api_key="pza_AOcqtNfGj4hJgc_281xXNFThsc8_eDGe" # имеет смысл сделать как аргумент при вводе в командную строку
    )

async def get_completion(prompt):
  response = await client.chat.completions.create(
    model="deepseek/deepseek-v4-flash",
    temperature = 0,
    response_format =  {
    "type": "json_schema",
    "json_schema": {
      "name": "answer_schema",
      "strict": True,
      "schema": {
          "items": {
              "type": "object",
              "properties": {
                "id_question": {"type": "number"},
                "index": {"type": "number"},
                "mark": {"type": "number"}
              }
          },
      "minItems": 1
      },
    }
  },
    messages=[
        {"role": "user", "content": prompt}]
  )
  return response.choices[0].message.content

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
    answers['mark'] = 0.0

    for i in range(num_of_questions - 1):
        temp = pd.DataFrame(data.iloc[:, 7+2*(i+1):9+2*(i+1)].reset_index())
        temp = temp.rename(columns={f'Вопрос {i+2}': 'id_question', f'Ответ {i+2}':'answer'})
        temp['num'] = i + 2
        temp['mark'] = 0.0
        answers = pd.concat([answers, temp], axis = 0)

    count = len(questions)
    i = 1
    prompts = []

    message = ""

    # Формирование промптов
    while i < 30: #count
        message = """На вход поданы вопросы, после каждого вопроса записан массив json-объектов.
                    В каждом из этих объектов содержится свойство "answer", значение которого содержит ответ на этот вопрос.

                    Для каждого вопроса выполняем следующие действия:
                        - модель отвечает на этот вопрос
                        - модель проходит по массиву json-объектов, относящемуся к этому вопросу
                        - оцениваем соответствие значения по ключу "answer" окончательному ответу модели от 0 до 100 (оценка кратна 10).
                            Если ответ на вопрос - число, то он либо полностью совпадает с ответом модели (100), либо не совпадает (0)
                        - заменяем в этом json-объекте свойство "answer" на свойство "mark" и присваиваем ему оценку соответствия
                        - записываем этот json-объект в итоговый безымянный массив JSON-объектов

                    Формат ответа: безымянный массив json-объектов из запроса со свойствами id_question, index, mark.
                    Если ответом является один объект - всё равно возвращать его как массив из одного объекта"""
        for j in range(question_block_size):
            if (i < count):
                message = message + "Вопрос:" + questions[i] + "\n"
                message = message + answers[answers['id_question'] == i][["id_question", "index", "answer"]].to_json(orient="records") + "\n"
            i = i + 1
        #if i < 40:
            #print(message)
        prompts.append(message)
    
    print("Промпты сформированы")

    # Асинхронное выполнение
    tasks = [get_completion(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    print("Модель оценила вопросы, выставляем оценки...")

    # Заполнение таблицы answers оценками модели
    for res in results:
        res_json = json.loads(res)
        #print(res_json)
        for i in range(len(res_json)):
            answers.loc[(answers['id_question'] == res_json[i]['id_question']) & (answers['index'] == res_json[i]['index']), 'mark'] = res_json[i]['mark']
    
    # Финальная таблица со строками ФИ - итоговая оценка
    final_rows = []

    for i in range(len(data)):
        final_rows.append({
            "ФИО": data['Фамилия'][i] + ' ' + data['Имя'][i],
            "Оценка за тест": round(answers[answers['index']==i]["mark"].mean(),2)
        })

    final_table = pd.DataFrame(final_rows)
    final_table.to_excel(excel_writer = "results.xlsx")

    print("Файл с итоговыми оценками готов!")

if __name__ == "__main__":
    asyncio.run(main())