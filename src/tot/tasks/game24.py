import re
import os
import csv
import sympy
from tot.tasks.base import Task, DATA_PATH
# 使用原始 prompts
from tot.prompts.game24 import *
print("✓ 使用原始 prompts") 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "Puzzles" in reader.fieldnames:
                for row in reader:
                    v = (row.get("Puzzles") or "").strip()
                    if v:
                        self.data.append(v)
            else:
                f.seek(0)
                reader2 = csv.reader(f)
                header = next(reader2, None)
                for row in reader2:
                    if not row:
                        continue
                    v = (row[0] or "").strip()
                    if v and v.lower() != "puzzles":
                        self.data.append(v)
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        
        # 从输出中提取评估关键词（支持Qwen的长输出）
        value_names = []
        for output in value_outputs:
            output_lower = output.lower()
            # 优先查找最后一行
            last_line = output.split('\n')[-1].strip().lower()
            
            # 检查最后一行是否包含关键词
            if 'sure' in last_line and 'impossible' not in last_line:
                value_names.append('sure')
            elif 'impossible' in last_line:
                value_names.append('impossible')
            elif 'likely' in last_line:
                value_names.append('likely')
            # 如果最后一行没有，搜索整个输出
            elif 'impossible' in output_lower:
                value_names.append('impossible')
            elif 'sure' in output_lower:
                value_names.append('sure')
            elif 'likely' in output_lower:
                value_names.append('likely')
            else:
                # 默认为impossible
                value_names.append('impossible')
        
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value