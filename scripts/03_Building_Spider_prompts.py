import pandas as pd
import glob
import numpy as np
import tqdm

def find_db_schema_filename(db_id):
    filename = glob.glob(f'spider_data/spider_data/database/{db_id}/*.sql')[0]
    return filename

def extract_schema(db_id):
    with open(find_db_schema_filename(db_id), 'r', encoding="utf-8") as file:
        clauses = file.read().split(';')
        clauses = [clause[clause.lower().find("create table"):] for clause in clauses if "create table" in clause.lower()]
    schema = "\n".join(clauses).strip()
    return schema

def generate_text2SQL_prompt(db_id, question, query=''):
    system_msg = f"""
                 You are a helpful assistant who answers questions about database tables by responding with SQL queries.  
                 Users will provide you with a set of tables represented as CREATE TABLE statements.  
                 Each CREATE TABLE statement may optionally be followed by the first few rows from the table in order to help write the correct SQL to answer 
                 questions. After the CREATE TABLE statements users will ask a question using a SQL comment starting with two dashes. 
                 You should answer the user's question by writing a SQL statement starting with SELECT.

                 Example 1:
                     Schema: 
                         CREATE TABLE IF NOT EXISTS "department" (
                        "Department_ID" int,
                        "Name" text,
                        "Creation" text,
                        "Ranking" int,
                        "Budget_in_Billions" real,
                        "Num_Employees" real,
                        PRIMARY KEY ("Department_ID")
                        )
                        CREATE TABLE IF NOT EXISTS "head" (
                        "head_ID" int,
                        "name" text,
                        "born_state" text,
                        "age" real,
                        PRIMARY KEY ("head_ID")
                        )
                        CREATE TABLE IF NOT EXISTS "management" (
                        "department_ID" int,
                        "head_ID" int,
                        "temporary_acting" text,
                        PRIMARY KEY ("Department_ID","head_ID"),
                        FOREIGN KEY ("Department_ID") REFERENCES `department`("Department_ID"),
                        FOREIGN KEY ("head_ID") REFERENCES `head`("head_ID")
                        )
                     Question: How many heads of the departments are older than 56 ?
                     SQL answer: SELECT count(*) FROM head WHERE age  >  56

                 Example 2:
                     Schema: 
                        CREATE TABLE CLASS (
                        CLASS_CODE varchar(5) PRIMARY KEY,
                        CRS_CODE varchar(10),
                        CLASS_SECTION varchar(2),
                        CLASS_TIME varchar(20),
                        CLASS_ROOM varchar(8),
                        PROF_NUM int,
                        FOREIGN KEY (CRS_CODE) REFERENCES COURSE(CRS_CODE)
                        FOREIGN KEY (PROF_NUM) REFERENCES EMPLOYEE(EMP_NUM)
                        )
                        CREATE TABLE COURSE (
                        CRS_CODE varchar(10) PRIMARY KEY,
                        DEPT_CODE varchar(10),
                        CRS_DESCRIPTION varchar(35),
                        CRS_CREDIT float(8),
                        FOREIGN KEY (DEPT_CODE) REFERENCES DEPARTMENT(DEPT_CODE)
                        )
                        CREATE TABLE DEPARTMENT (
                        DEPT_CODE varchar(10) PRIMARY KEY,
                        DEPT_NAME varchar(30),
                        SCHOOL_CODE varchar(8),
                        EMP_NUM int,
                        DEPT_ADDRESS varchar(20),
                        DEPT_EXTENSION varchar(4),
                        FOREIGN KEY (EMP_NUM) REFERENCES EMPLOYEE(EMP_NUM)
                        )
                        CREATE TABLE EMPLOYEE (
                        EMP_NUM int PRIMARY KEY,
                        EMP_LNAME varchar(15),
                        EMP_FNAME varchar(12),
                        EMP_INITIAL varchar(1),
                        EMP_JOBCODE varchar(5),
                        EMP_HIREDATE datetime,
                        EMP_DOB datetime
                        )
                        CREATE TABLE ENROLL (
                        CLASS_CODE varchar(5),
                        STU_NUM int,
                        ENROLL_GRADE varchar(50),
                        FOREIGN KEY (CLASS_CODE) REFERENCES CLASS(CLASS_CODE)
                        FOREIGN KEY (STU_NUM) REFERENCES STUDENT(STU_NUM)
                        )
                        CREATE TABLE PROFESSOR (
                        EMP_NUM int,
                        DEPT_CODE varchar(10),
                        PROF_OFFICE varchar(50),
                        PROF_EXTENSION varchar(4),
                        PROF_HIGH_DEGREE varchar(5),
                        FOREIGN KEY (EMP_NUM) REFERENCES EMPLOYEE(EMP_NUM),
                        FOREIGN KEY (DEPT_CODE) REFERENCES DEPARTMENT(DEPT_CODE)
                        )
                        CREATE TABLE STUDENT (
                        STU_NUM int PRIMARY KEY,
                        STU_LNAME varchar(15),
                        STU_FNAME varchar(15),
                        STU_INIT varchar(1),
                        STU_DOB datetime,
                        STU_HRS int,
                        STU_CLASS varchar(2),
                        STU_GPA float(8),
                        STU_TRANSFER numeric,
                        DEPT_CODE varchar(18),
                        STU_PHONE varchar(4),
                        PROF_NUM int,
                        FOREIGN KEY (DEPT_CODE) REFERENCES DEPARTMENT(DEPT_CODE)
                        )
                     Question: What is the first name, gpa and phone number of the top 5 students with highest gpa?
                     SQL answer: SELECT stu_gpa ,  stu_phone ,  stu_fname FROM student ORDER BY stu_gpa DESC LIMIT 5

                Example 3:
                    Schema: 
                        CREATE TABLE Person (
                        name varchar(20) PRIMARY KEY,
                        age INTEGER,
                        city TEXT,
                        gender TEXT,
                        job TEXT
                        )
                        CREATE TABLE PersonFriend (
                        name varchar(20),
                        friend varchar(20),
                        year INTEGER,
                        FOREIGN KEY (name) REFERENCES Person(name),
                        FOREIGN KEY (friend) REFERENCES Person(name)
                        )
                    Question: Find the name and age of the person who is a friend of both Dan and Alice.
                    SQL Answer: SELECT T1.name ,  T1.age FROM Person AS T1 JOIN PersonFriend AS T2 ON T1.name  =  T2.name WHERE T2.friend  =  'Dan' INTERSECT SELECT T1.name ,   T1.age FROM Person AS T1 JOIN PersonFriend AS T2 ON T1.name  =  T2.name WHERE T2.friend  =  'Alice'
                 """
    user_msg = f"""
                Schema: {extract_schema(db_id)}
                Question: {question}
                SQL Answer: {query}
                """
    return system_msg, user_msg

train_df = pd.read_json('spider_data/spider_data/train_final_without_FS_examples.json')
train_prompts = []
training_missing_schema = []

for i in tqdm.tqdm(range(len(train_df))):
    db_id = train_df.iloc[i, 0]
    query = train_df.iloc[i, 1]
    question = train_df.iloc[i, 2]
    try:
        train_prompts.append(generate_text2SQL_prompt(db_id, question, query))
    except IndexError:
        if(db_id not in training_missing_schema):
            training_missing_schema.append(db_id)

np.save('spider_prompts/training_prompts.npy', train_prompts)

names = ['db_id', 'query', 'question']
test_df = pd.read_json('spider_data/spider_data/dev.json')
test_df = test_df[names]
test_prompts = []
test_missing_schema = []

for i in tqdm.tqdm(range(len(test_df))):
    db_id = test_df.iloc[i, 0]
    query = test_df.iloc[i, 1]
    question = test_df.iloc[i, 2]
    
    try:
        test_prompts.append(generate_text2SQL_prompt(db_id, question))
    except IndexError:
        if(db_id not in test_missing_schema):
            test_missing_schema.append(db_id)

np.save('spider_prompts/test_prompts.npy', test_prompts)