import pandas as pd

question_block_size = 10

data = pd.read_excel('example.xlsx')

# Количество вопросов
num_of_questions = (len(data.columns) - 7) // 2

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

answers