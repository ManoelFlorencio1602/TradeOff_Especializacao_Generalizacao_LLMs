import pandas as pd
import glob
import numpy as np
import tqdm

def generate_mmlu_prompt(question, subject, choices):
    system_msg = f"""
                 Answer the following multiple-choice question about a specific subject by selecting the correct option: '0', '1', '2', or '3'.
                 The provided answer must be only the correrct option, without reasoning, notes or especifications.

                 Example 1
                     Question: For a stationary autoregressive process, shocks will
                     Subject: econometrics
                     Choices:
                         0: "Eventually die away"
                         1: "Persist indefinitely"
                         2: "Grow exponentially"
                         3: "Never occur"
                     Correct option: 0
                 Example 2
                     Question: What is the sign of the covenant for Jewish males?
                     Subject: world_religions
                     Choices:
                         0: "The rainbow"
                         1: "Circumcision"
                         2: "A son"
                         3: "Bar mitzvah"
                     Correct option: 1
                 Example 3
                     Question: Psychological egoism is:
                     Subject: philosofy
                     Choices:
                         0: "an ethical theory about how we ought to behave."
                         1: "a generalization concerning the way people tend to behave."
                         2: "a claim about human nature and the ways people are capable of behaving."
                         3: "none of the above."
                     Correct option: 2
                 Example 4
                     Question: You are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)
                     Subject: astronomy
                     Choices:
                         0: "It would be harder since the truck is heavier on Mars."
                         1: "It would be easier since the truck is lighter on Mars."
                         2: "It would be harder since the truck is lighter on Mars."
                         3: "It would be the same no matter where you are."
                     Correct option: 3
                 """
    user_msg = f"""
                Question: {question}
                Subject: {subject}
                Choices:
                """
    for i in range(len(choices)):
        user_msg += f'\r{i}: "{choices[i]}"\n'

    user_msg += "Correct option:"

    return system_msg, user_msg

filenames = glob.glob('mmlu_data/*')

prompts = []
for filename in filenames:
    df = pd.read_csv(filename)
    for i in range(len(df)):
        question = df.iloc[i, 0]
        subject = df.iloc[i, 1]
        choices = eval(df.iloc[i, 2])

        prompt = generate_mmlu_prompt(question, subject, choices)

        prompts.append(prompt)
np.save('mmlu_prompts/mmlu_prompts.npy', prompts)    