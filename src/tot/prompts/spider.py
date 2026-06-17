propose_step_prompt = """You are solving a Spider text-to-SQL task for a SQLite database.

{input}

Current candidate SQL:
{solution}

Write ONE improved complete SQLite SQL query that answers the question.
Use only tables and columns from the schema.
Prefer a valid executable query over explanation.
Output only the SQL query, ending with a semicolon.
SQL:
"""

propose_final_prompt = """You are solving a Spider text-to-SQL task for a SQLite database.

{input}

Current candidate SQL:
{solution}

Return the final complete SQLite SQL query.
Use only tables and columns from the schema.
Output only the SQL query, ending with a semicolon.
SQL:
"""

value_prompt = """You are judging whether a candidate SQLite SQL query is likely to correctly answer a Spider database question.

{input}

Candidate SQL:
{solution}

Score the candidate from 0 to 10.
Consider schema correctness, SQL validity, join conditions, filters, aggregation, ordering, and whether it answers the question.
You must output exactly ONE line: a single number from 0 to 10.
"""

direct_sql_prompt = """You are solving a Spider text-to-SQL task for a SQLite database.

{input}

Write the single best SQLite SQL query that answers the question.
Use only tables and columns from the schema.
Do not explain your reasoning.
Do not use markdown.
Output only the SQL query, ending with a semicolon.
SQL:
"""

decomp_repair_prompt = """You are repairing a SQLite SQL query for a Spider text-to-SQL task.

{input}

Current SQL:
{sql}

Execution feedback:
{feedback}

Before writing the final SQL, silently check:
1. Which exact output columns the question asks for, and do not output extra columns.
2. Which tables and join keys are necessary.
3. Which filters come from the question.
4. Whether DISTINCT, GROUP BY, aggregation, ORDER BY, and LIMIT are required.
5. Whether the SQL is valid SQLite.

Output only one corrected complete SQLite SQL query, ending with a semicolon.
SQL:
"""

execution_value_prompt = """You are judging a candidate SQLite SQL query for a Spider database question.

{input}

Candidate SQL:
{sql}

Execution feedback:
{feedback}

Score from 0 to 10.
Reward exact schema usage, correct output columns, correct joins, correct filters, correct aggregation/order/limit, and valid SQLite execution.
Penalize extra output columns, unsupported filters, duplicate rows when DISTINCT is needed, and SQL that only looks plausible.
You must output exactly ONE line: a single number from 0 to 10.
"""
