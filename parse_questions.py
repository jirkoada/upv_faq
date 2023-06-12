import pandas as pd
from expand_text import expand_text
import re

if __name__ == "__main__":
    df = pd.read_excel("Q50_raw2.xlsx")
    df.to_csv("Q50_raw2.csv", sep="\t")
    #print(df["Group"][df["Group"].isnull()])
    #print(df["Group"][~df["Group"].isnull()])
    #df["Group"][df["Group"].isnull()] = [f"Q{i+1}" for i in range(50)]

    q_json = []
    a_json = []

    for i in range (0, df.shape[0], 2):
        cls = int(i / 2)
        questions = re.split(";|\.|\?;|\?", df["Questions"][i])
        expanded = []
        for q in questions[:-1]:
            q_exp = expand_text(q)
            expanded += q_exp
        #print(len(expanded))
        questions = [q.replace('  ', ' ').strip() for q in expanded]
        #print(questions)
        answer = df["Answer"][i]
        for q in questions:
            q_json.append({"question": q, "class": cls})
        a_json.append({"answer": answer, "class": cls})

    q_data = pd.DataFrame.from_records(q_json)
    a_data = pd.DataFrame.from_records(a_json)

    q_data.to_excel("Q50_questions.xlsx")
    a_data.to_excel("Q50_answers.xlsx")
    q_data.to_json("Q50_questions.json")
    a_data.to_json("Q50_answers.json")
