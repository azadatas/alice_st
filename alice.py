import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

# Set background color and text color
st.markdown(
    """
    <style>
        .highlight {
            background-color: #DAA520;  /* Yellow color as an example */
            padding: 10px;
            border-radius: 5px;
        }
        .highlight h1 {
            color: #000000;  /* Black color for text */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add title with the highlighted background
st.markdown("<div class='highlight'><h1> Метрики для Яндекс Алисы </h1></div>", unsafe_allow_html=True)


st.subheader("Вступление")
st.write("В этом проекте я выбрал ключевые показатели, такие как Accuracy, Precision, Recall, F1 score, Confusion matrix, Контекстуальная релевантность и Человекоподобное взаимодействие, чтобы оценить производительность Алисы. Эти выбранные показатели дают ценную информацию о качестве и надежности взаимодействия Алисы, обеспечивая надежную оценку ее производительности в реальных диалоговых сценариях. Мы также признаем важность человеческой оценки, понимая, что такие факторы, как читаемость, слаженность и общая удовлетворенность пользователей, в значительной степени способствуют оценке успеха виртуальных ассистентов.")

st.write("Данные ниже были созданы мной и Алисой. Я старался быть максимально объективным в оценивании.")

# Read csv and display df
csv_file_path = 'query-response.csv'
df = pd.read_csv(csv_file_path)
st.subheader("Начальный DataFrame")
st.write(df)

st.write("Представлены 31 вопросов на русском и 31 идентичных вопросов на казахском языке, чтобы увидеть, как Алиса отвечает на один и тот же вопрос на разных языках. Наша цель — не только оценить возможности Алисы, но и изучить, чем казахская версия отличается от оригинальной русской версии. Выбранные вопросы охватывают 10 категорий, представляющих запросы, которые пользователи могут задавать в своей повседневной жизни.")

# Divide
index_of_query = df[df['User Query'] == 'Kazakh language'].index

# Creating df with only Kazakh queries
qaz_queries = df[df.index >= index_of_query.min()]

qaz_queries.reset_index(drop=True, inplace=True)

# Creating df with only Russian queries
rus_queries = df[df.index < index_of_query.min()]

# Display the filtered DataFrame
st.subheader("DataFrame состоящий из запросов только на русском языке")
st.write(rus_queries)

# Display the filtered DataFrame
st.subheader("DataFrame состоящий из запросов только на казахском языке")
st.write(qaz_queries)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# ACCURACY 
st.title("Accuracy")
st.write("Accuracy (Точность) - это показатель, используемый для измерения эффективности ответов Алисы. Формула точности рассчитывается как количество правильных ответов, разделенное на общее количество ответов.")
st.latex(r"\text{Accuracy} = \frac{\text{Correct Responses}}{\text{Total Responses}}")
st.write("Correct Responses - Правильные ответы")
st.write("Total Responses - Общие ответы")
st.write("Is it perfect response? - Это идеальный ответ?")
st.write("Yes - Да")
st.write("Satisfactory - Удовлетворительно")
st.write("Not satisfactory - Неудовлетворительно")
st.write("Для этого проекта ответы <Yes> и <Satisfactory> в столбце <Is it perfect response?> считаются правильными, а ответы <Not satisfactory> считаются неправильными.")
st.write("Показатель точности представляет собой общий процент, отражающий, насколько часто ответы Алисы соответствуют желаемым критериям, предлагая количественную оценку производительности системы.")

# Accuracy calculation for Total 
correct_total = df["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_total = df["Is it perfect response?"].isin(["Not satisfactory"]).sum()

accuracy_total = correct_total / (correct_total + incorrect_total)

# Russian DataFrame
correct_russian = rus_queries["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_russian = rus_queries["Is it perfect response?"].isin(["Not satisfactory"]).sum()

accuracy_russian = correct_russian / (correct_russian + incorrect_russian)

# Kazakh DataFrame
correct_kazakh = qaz_queries["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_kazakh = qaz_queries["Is it perfect response?"].isin(["Not satisfactory"]).sum()

accuracy_kazakh = correct_kazakh / (correct_kazakh + incorrect_kazakh)

# Display accuracy
st.subheader("Accuracy Progress Bars")

# Display the accuracy for total using a progress bar
st.write(f"Accuracy Total: {accuracy_total:.2%}")
st.progress(accuracy_total)

# Display the accuracy for Russian using a progress bar
st.write(f"Accuracy на Русском языке: {accuracy_russian:.2%}")
st.progress(accuracy_russian)

# Display the accuracy for Kazakh using a progress bar
st.write(f"Accuracy на Казахском языке: {accuracy_kazakh:.2%}")
st.progress(accuracy_kazakh)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Selectbox
accuracy_choice = st.selectbox("Выберите Accuracy Data Frame:", ["Total", "Kazakh", "Russian"])

if accuracy_choice == "Kazakh":
    accuracy_df = qaz_queries
elif accuracy_choice == "Russian":
    accuracy_df = rus_queries
    pass
else:
    accuracy_df = df  # Total DataFrame

# Calculate accuracy
correct_rows = accuracy_df["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_rows = accuracy_df["Is it perfect response?"].isin(["Not satisfactory"]).sum()

# Calculate accuracy
total_rows = correct_rows + incorrect_rows
accuracy = correct_rows / total_rows if total_rows > 0 else 0


# Display accuracy
st.subheader(f"Accuracy for {accuracy_choice} Data Frame:")
st.write(f"Правильные ответы: {correct_rows}")
st.write(f"Общие ответы: {total_rows}")
st.write(f"Accuracy: {accuracy:.2%}")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# PRECISION
st.title("Precision")
st.write("Precision это показатель, оценивающий точность положительных прогнозов, сделанных моделью. В контексте этого проекта Precision измеряет точность ответов Алисы при прогнозировании положительного результата, например, удовлетворительного или правильного ответа. Более высокая точность означает, что положительные предсказания Алисы более надежны и с меньшей вероятностью содержат ложноположительные результаты.")
st.latex(r"\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Calculate Precision for each data frame
TP_total = (df['True Positive'] == True ).sum()
FP_total = (df['False Positive'] == True ).sum()

precision_total = TP_total / (TP_total + FP_total)

TP_rus = (rus_queries['True Positive'] == True ).sum()
FP_rus = (rus_queries['False Positive'] == True ).sum()

precision_russian = TP_rus / (TP_rus + FP_rus)

TP_qaz = (qaz_queries['True Positive'] == True ).sum()
FP_qaz = (qaz_queries['False Positive'] == True ).sum()

precision_kazakh = TP_qaz / (TP_qaz + FP_qaz)

# Display Precision
st.subheader("Precision для разных DataFrame:")

# Specify the y-axis range to focus on the relevant percentage range
y_axis_range = [min(precision_kazakh, precision_russian, precision_total) * 100 - 5, 100]

# Set the bar width
bar_width = 0.5

# Create a figure
precision_fig = go.Figure()

# Add bar traces
for i, label in enumerate(["Total", "Kazakh", "Russian"]):
    precision_fig.add_trace(
        go.Bar(
            x=[label],
            y=[[precision_total * 100, precision_kazakh * 100, precision_russian * 100][i]],
            name=label,
            marker_color=['#FFD700', '#00A1DE', '#FF6666'][i],
            width=bar_width,
        )
    )

# Update layout
precision_fig.update_layout(
    title="Precision Distribution",
    yaxis=dict(title="Precision (%)", range=y_axis_range),
)

# Show the plot
st.plotly_chart(precision_fig)


# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# RECALL
st.title("Recall")
st.write("Recall, также известный как чувствительность или истинно положительный результат, оценивает способность модели правильно идентифицировать все соответствующие случаи, особенно истинные положительные результаты. В контексте этого проекта отзыв оценивает, насколько хорошо Алиса может фиксировать и получать все правильные ответы на запросы пользователей. Более высокий уровень recall указывает на то, что Алиса эффективно получает релевантные ответы, даже если есть некоторые ложноотрицательные ответы.")
st.latex(r"\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Calculate recall for each data frame
TP_total = (df['True Positive'] == True ).sum()
FN_total = (df['False Negative'] == True ).sum()

recall_total = TP_total / (TP_total + FN_total)

TP_rus = (rus_queries['True Positive'] == True ).sum()
FN_rus = (rus_queries['False Negative'] == True ).sum()

recall_russian = TP_rus / (TP_rus + FN_rus)

TP_qaz = (qaz_queries['True Positive'] == True ).sum()
FN_qaz = (qaz_queries['False Negative'] == True ).sum()

recall_kazakh = TP_qaz / (TP_qaz + FN_qaz)

# Display recall
st.subheader("Recall для разных DataFrame:")

# Specify the y-axis range to focus on the relevant percentage range
y_axis_range = [min(recall_kazakh, recall_russian, recall_total) * 100 - 15, 100]

# Set the bar width
bar_width = 0.5

precision_fig = go.Figure()

# Add bar traces
for i, label in enumerate(["Total", "Kazakh", "Russian"]):
    precision_fig.add_trace(
        go.Bar(
            x=[label],
            y=[[recall_total * 100, recall_kazakh * 100, recall_russian * 100][i]],
            name=label,
            marker_color=['#FFD700', '#00A1DE', '#FF6666'][i],
            width=bar_width,
        )
    )

# Update layout
precision_fig.update_layout(
    title="Precision Distribution",
    yaxis=dict(title="Precision (%)", range=y_axis_range),
)

# Show the plot
st.plotly_chart(precision_fig)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# F1 SCORE
st.title("F1 score")
st.write("F1 score, также известный как показатель F1 или значение F1, представляет собой показатель, который объединяет precision и recall в одно значение. Он обеспечивает баланс между ними, что делает его полезным в сценариях с неравномерным распределением классов. Формула для оценки F1 определяется следующим образом:")

# Formula
f1_formula = r"F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}"
# Display the formula using st.latex()
st.latex(f1_formula)

st.write("The F1 score варьируется от 0 до 1, причем более высокие значения указывают на лучший баланс между precision и recall. Это особенно ценно в задачах двоичной классификации, где необходимо учитывать как ложноположительные, так и ложноотрицательные результаты.")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("F1 Score Progress Bars")

# Total
f1_total = 2 * precision_total * recall_total / (precision_total + recall_total)

st.write(f"F1 Score Total: {f1_total:.2%}")
# Display the F1 score using a progress bar
st.progress(f1_total)

# Russian 
f1_russian = 2 * precision_russian * recall_russian / (precision_russian + recall_russian)

st.write(f"F1 Score для Русского языка: {f1_russian:.2%}")
# Display the F1 score using a progress bar
st.progress(f1_russian)

# Kazakh 
f1_kazakh = 2 * precision_kazakh * recall_kazakh / (precision_kazakh + recall_kazakh)

st.write(f"F1 Score для Казахского языка: {f1_kazakh:.2%}")
# Display the F1 score using a progress bar
st.progress(f1_kazakh)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

st.title("Confusion Matrix")
st.write("Confusion matrix — это таблица, которая часто используется для оценки эффективности модели классификации. В нем представлена ​​подробная разбивка прогнозов модели и их фактических результатов. В контексте этого проекта, где оцениваются ответы Алисы, confusion matrix помогает анализировать производительность системы с точки зрения истинно положительных результатов (правильно идентифицированных положительных ответов), истинно отрицательных результатов (правильно определенных отрицательных ответов), ложных срабатываний (неправильно определенных положительных ответов) и ложноотрицательные (неправильно идентифицированные отрицательные ответы).")
st.write("Confusion matrix дает комплексное представление о том, насколько ответы Алисы соответствуют фактической правильности, помогая более глубокому пониманию сильных и слабых сторон системы. Отсутствие True Negative результатов в этом контексте обусловлено характером оценки, целью которой является оценка правильности ответов, а не сценарий бинарной классификации.")
st.write("Все ответы")

# Assuming you have the following values
TP = (df['True Positive'] == True ).sum()
FP = (df['False Positive'] == True ).sum()
FN = (df['False Negative'] == True ).sum()

# Create a DataFrame for the confusion matrix
confusion_matrix_data = pd.DataFrame({
    'Actual Negative': [0, FP],
    'Actual Positive': [FN, TP]
}, index=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix для всех ответов')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Display the plot in Streamlit
st.pyplot(plt)

# Russian language
st.write("Ответы на Русском языке")

# Assuming you have the following values
TP = (rus_queries['True Positive'] == True ).sum()
FP = (rus_queries['False Positive'] == True ).sum()
FN = (rus_queries['False Negative'] == True ).sum()

# Create a DataFrame for the confusion matrix
confusion_matrix_data = pd.DataFrame({
    'Actual Negative': [0, FP],
    'Actual Positive': [FN, TP]
}, index=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix для ответов на Русском языке')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Display the plot in Streamlit
st.pyplot(plt)

# Kazakh language
st.write("Ответы на Казахском языке")

# Assuming you have the following values
TP = (qaz_queries['True Positive'] == True ).sum()
FP = (qaz_queries['False Positive'] == True ).sum()
FN = (qaz_queries['False Negative'] == True ).sum()

# Create a DataFrame for the confusion matrix
confusion_matrix_data = pd.DataFrame({
    'Actual Negative': [0, FP],
    'Actual Positive': [FN, TP]
}, index=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix для ответов на Казахском языке')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Display the plot in Streamlit
st.pyplot(plt)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# CONTEXTUAL RELEVANCE 
st.title("Contextual Relevance")
st.write("Контекстная релевантность измеряет, насколько ответы Алисы соответствуют контексту или значению запросов пользователя. В контексте этого проекта контекстуальная релевантность оценивается на основе качественного суждения людей-оценщиков с учетом того, являются ли ответы Алисы уместными и контекстуально согласованными с заданными запросами.")
st.latex(r"\text{Contextual Relevance} = \frac{\text{Number of Contextually Relevant Responses}}{\text{Total Number of Responses}} \times 100")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Total
# Trimming spaces in answers
df["Contextually relevant"] = df["Contextually relevant"].apply(lambda x: x.strip() if isinstance(x, str) else x)

rel_total = (df['Contextually relevant'] == 'Yes' ).sum()
not_rel_total = (df['Contextually relevant'] == 'No' ).sum()

context_total = rel_total / (rel_total + not_rel_total)

# Russian
rus_queries["Contextually relevant"] = rus_queries["Contextually relevant"].apply(lambda x: x.strip() if isinstance(x, str) else x)

rel_rus = (rus_queries['Contextually relevant'] == 'Yes' ).sum()
not_rel_rus = (rus_queries['Contextually relevant'] == 'No' ).sum()

context_rus = rel_rus / (rel_rus + not_rel_rus)

# Kazakh
qaz_queries["Contextually relevant"] = qaz_queries["Contextually relevant"].apply(lambda x: x.strip() if isinstance(x, str) else x)

rel_qaz = (qaz_queries['Contextually relevant'] == 'Yes' ).sum()
not_rel_qaz = (qaz_queries['Contextually relevant'] == 'No' ).sum()

context_qaz = rel_qaz / (rel_qaz + not_rel_qaz)

# Display contextual relevance
st.subheader("Contextual Relevance Progress Bars")

# Display Contextual Relevance for total using a progress bar
st.write(f"Contextual Relevance Общий: {context_total:.2%}")
st.progress(context_total)

# Display Contextual Relevance for Russian using a progress bar
st.write(f"Contextual Relevance для Русского языка: {context_rus:.2%}")
st.progress(context_rus)

# Display Contextual Relevance for Kazakh using a progress bar
st.write(f"Contextual Relevance для Казахского языка: {context_qaz:.2%}")
st.progress(context_qaz)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# HUMAN LIKE INTERACTION
st.title("Human-like Interaction")
st.write("Метрика «Человекоподобное взаимодействие» оценивает, демонстрируют ли ответы Алисы качества, напоминающие человеческое общение. Если он просто дает мне результат поиска, это считается «Нет», если он дает мне ответ именно на то, что я спросил, то это «Да».")
st.latex(r"\text{Human-like Interaction Rate} = \frac{\text{Number of Responses with Human-like Interaction}}{\text{Total Number of Responses}} \times 100")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Total
hum_total = (df['Human-like interaction'] == 'Yes' ).sum()
not_hum_total = (df['Human-like interaction'] == 'No' ).sum()

human_total = hum_total / (hum_total + not_hum_total)

# Russian
hum_rus = (rus_queries['Human-like interaction'] == 'Yes' ).sum()
not_hum_rus = (rus_queries['Human-like interaction'] == 'No' ).sum()

human_rus = hum_rus / (hum_rus + not_hum_rus)

# Kazakh
hum_qaz = (qaz_queries['Human-like interaction'] == 'Yes' ).sum()
not_hum_qaz = (qaz_queries['Human-like interaction'] == 'No' ).sum()

human_qaz = hum_qaz / (hum_qaz + not_hum_qaz)

# Display Human Like Interaction
st.subheader("Human-like Interaction Progress Bars")

# Display Human-like Interaction for total using a progress bar
st.write(f"Human-like Interaction Общий: {human_total:.2%}")
st.progress(human_total)

# Display Human-like Interaction for Russian using a progress bar
st.write(f"Human-like Interaction для Русского языка: {human_rus:.2%}")
st.progress(human_rus)

# Display Human-like Interaction for Kazakh using a progress bar
st.write(f"Human-like Interaction для Казахского языка: {human_qaz:.2%}")
st.progress(human_qaz)

st.write("Интересно, что казахская версия Алисы в данной метрике показывает немного лучший результат, чем русская версия.")





