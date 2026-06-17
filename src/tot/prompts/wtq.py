direct_sql_prompt = """You are answering a question over a single SQLite table.

{input}

Write one SQLite SQL query that answers the question.
Use only the table and columns shown in the schema.
The table name is w.
Do not explain your reasoning.
Do not use markdown.
Output only the SQL query, ending with a semicolon.
SQL:
"""

decomp_repair_prompt = """You are repairing a SQLite SQL query for a table question answering task.

{input}

Current SQL:
{sql}

Execution feedback:
{feedback}

Silently check whether the SQL selects the exact answer cells requested by the question.
Fix column names, filters, aggregation, ordering, and LIMIT if needed.
Use only the table w and columns shown in the schema.
Output only one corrected complete SQLite SQL query, ending with a semicolon.
SQL:
"""

execution_value_prompt = """You are judging a candidate SQLite SQL query for a table question answering task.

{input}

Candidate SQL:
{sql}

Execution feedback:
{feedback}

Score from 0 to 10.
Reward valid SQLite, correct column selection, correct filters, correct aggregation/order/limit, and an answer shape matching the question.
Penalize invalid SQL, extra answer columns, unsupported filters, and queries that return too many irrelevant rows.
You must output exactly ONE line: a single number from 0 to 10.
"""
