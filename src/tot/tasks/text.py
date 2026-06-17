import os
import re
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.text import *
from tot.models import gpt


def _extract_required_end_sentences(x: str) -> list[str]:
    text = (x or '').strip()
    if not text:
        return []
    sents = [s.strip() for s in re.findall(r"[^.!?\n]+[.!?]", text) if s.strip()]
    if len(sents) >= 4:
        return sents[:4]
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if len(parts) >= 4:
        return parts[:4]
    return [text]


def _split_paragraphs(text: str) -> list[str]:
    t = (text or '').strip()
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return parts


def _norm(s: str) -> str:
    s = (s or '').strip()
    s = s.strip('"\'')
    s = re.sub(r"\s+", " ", s)
    return s


def _truncate_paragraphs(paragraphs: list[str], max_paragraphs: int = 2, max_chars: int = 1200) -> str:
    if not paragraphs:
        return ''
    keep = paragraphs[-max_paragraphs:]
    text = "\n\n".join([p.strip() for p in keep if p and p.strip()]).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[-max_chars:]
    return text


class TextTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='data_100_random_text.txt'):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'text', file)
        self.data = open(path).readlines()
        self.steps = 5
        self.stops = ['\nPassage:\n', None, None, None, None]

        self.backend = os.getenv("TOT_BACKEND", "qwen3-32b-vllm")
        self.temperature = 0.7

        self.propose_stop = None
        self.propose_max_tokens = 256
        self.propose_temperature = 0.9
        self.n_propose_sample = 5
        self.n_max_propose = 5

        self.value_stop = "\n"
        self.value_max_tokens = 16
        self.value_temperature = 0.0
        self.value_cache = {}

        self._passage_split_re = re.compile(r"Passage:\s*\n", re.IGNORECASE)
        self._score_re = re.compile(r"coherenc(?:y|e)\s*score\s*(?:is|:)?\s*(\d+)", re.IGNORECASE)

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def is_solved(self, idx: int, y: str) -> bool:
        try:
            x = self.get_input(idx)
        except Exception:
            x = None

        if not y:
            return False

        if 'Passage:' not in y:
            return False

        passage = self._passage_split_re.split(y)[-1].strip()
        paragraphs = _split_paragraphs(passage)
        if len(paragraphs) != 4:
            return False

        required = _extract_required_end_sentences(x or '')
        if len(required) < 4:
            required = required + [''] * (4 - len(required))

        for i in range(4):
            end_s = (required[i] or '').strip()
            if not end_s:
                return False

            para = (paragraphs[i] or '').strip()
            if not _norm(para).endswith(_norm(end_s)):
                return False

            for j in range(4):
                if j == i:
                    continue
                other = (required[j] or '').strip()
                if other and other in para:
                    return False

        return True
    
    def test_output(self, idx: int, output: str):
        output = output.split('Passage:\n')[-1]
        prompt = (
            'Output ONLY one line in exactly this format (no other text): "Thus the coherency score is {s}"\n'
            + score_prompt
            + output
        )
        score_outputs = gpt(
            prompt,
            n=5,
            model=getattr(self, 'backend', os.getenv("TOT_BACKEND", "qwen3-32b-vllm")),
            temperature=0.0,
            max_tokens=getattr(self, 'value_max_tokens', 16),
            stop='\n',
        )

        scores = []
        for score_output in score_outputs:
            m = self._score_re.search((score_output or '').strip())
            if not m:
                continue
            try:
                score = int(m.group(1))
            except Exception:
                continue
            if score < 1:
                score = 1
            if score > 10:
                score = 10
            scores.append(score)

        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info

    def propose_prompt_wrap(self, x: str, y: str) -> str:
        y = y or ''
        if 'Passage:' not in y:
            return (
                'Generate ONE plan only. Output exactly:\n\n'
                'Plan:\n'
                '- ...\n'
                '- ...\n\n'
                'Passage:\n'
                'Do NOT write the passage yet.\n\n'
                + 'Instruction:\n'
                + (x or '').strip()
                + '\n'
            )

        plan_part = y.split('Passage:', 1)[0].strip()
        passage_part = self._passage_split_re.split(y)[-1]
        paragraphs = _split_paragraphs(passage_part)
        next_idx = len(paragraphs) + 1

        required = _extract_required_end_sentences(x)
        if len(required) < 4:
            required = required + [''] * (4 - len(required))
        end_sentence = required[next_idx - 1] if 1 <= next_idx <= 4 else ''

        forbidden = [s for i, s in enumerate(required, 1) if s and i != next_idx]
        forbidden_lines = ''.join([f'- Do NOT include this sentence anywhere: {s}\n' for s in forbidden])

        required_lines = ''.join([f'- Paragraph {i+1} ends with: {s}\n' for i, s in enumerate(required) if s])

        existing = _truncate_paragraphs(paragraphs, max_paragraphs=2, max_chars=1200)
        return (
            f'Write ONLY paragraph {next_idx} (one short paragraph).\n'
            f'It must end with exactly this sentence (as the final sentence): {end_sentence}\n'
            'Do NOT include any other paragraphs. Do NOT include Plan:. Do NOT include Passage:.\n'
            + forbidden_lines
            + 'Do NOT repeat the required ending sentence earlier in the paragraph; it must appear only once at the end.\n\n'
            + 'Required endings:\n'
            + required_lines
            + '\n'
            + 'Plan:\n'
            + plan_part
            + ('\n\nPassage so far:\n' + existing + '\n' if existing else '\n\nPassage so far:\n(EMPTY)\n')
        )

    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int = -1) -> list:
        y = y or ''
        cands = []

        required = _extract_required_end_sentences(x)
        if len(required) < 4:
            required = required + [''] * (4 - len(required))

        for out in outputs or []:
            text = (out or '').strip()
            if not text:
                continue

            if 'Passage:' not in y:
                # Normalize to a clean plan state: keep only the plan part and force an empty Passage: placeholder.
                if 'Plan:' not in text:
                    text = 'Plan:\n' + text
                plan_only = text
                if 'Passage:' in plan_only:
                    plan_only = plan_only.split('Passage:', 1)[0].rstrip()
                plan_only = plan_only.rstrip()
                cand = plan_only + '\n\nPassage:\n'
            else:
                # Normalize to a clean full state: replace the placeholder passage with generated passage text.
                prefix = y.split('Passage:', 1)[0].rstrip() + '\n\nPassage:\n'
                existing = self._passage_split_re.split(y)[-1].strip()
                existing_paras = _split_paragraphs(existing)
                existing_text = "\n\n".join(existing_paras).strip()

                # Determine which paragraph we are generating next.
                next_idx = len(existing_paras) + 1
                end_sentence = required[next_idx - 1] if 1 <= next_idx <= 4 else ''
                forbidden = [s for i, s in enumerate(required, 1) if s and i != next_idx]

                para = text
                if 'Passage:' in para:
                    # Be robust to variants like "Passage: ..." without a newline.
                    try:
                        para = self._passage_split_re.split(para)[-1]
                    except Exception:
                        para = para.split('Passage:', 1)[-1]
                    if 'Passage:' in para:
                        para = para.split('Passage:', 1)[-1]
                elif 'Plan:' in para:
                    # If the model mistakenly outputs a plan here, drop the plan block.
                    if '\n\n' in para:
                        para = para.split('\n\n', 1)[-1]
                para = re.sub(r"^Paragraph\s*\d+\s*:\s*", "", para.strip(), flags=re.IGNORECASE)

                # Keep only one paragraph.
                para_paras = _split_paragraphs(para)
                para = (para_paras[0] if para_paras else para).strip()

                if not para:
                    continue

                # If the model mentioned forbidden ending sentences, truncate before the first forbidden occurrence.
                para_norm = para
                for s in forbidden:
                    if not s:
                        continue
                    idx = para_norm.find(s)
                    if idx != -1:
                        para_norm = para_norm[:idx].rstrip()
                para = para_norm.strip()

                # Enforce required ending sentence.
                if end_sentence:
                    end_sentence_n = _norm(end_sentence)
                    para_n = _norm(para)
                    pos = para_n.rfind(end_sentence_n)
                    if pos != -1:
                        # Trim to the last occurrence of the required ending.
                        cut = para_n[: pos + len(end_sentence_n)].strip()
                        para = cut
                    else:
                        # Repair by appending the required ending.
                        sep = ''
                        if para and not para.endswith((' ', '\n')):
                            sep = ' '
                        para = (para.strip() + sep + end_sentence_n).strip()

                glue = ('\n\n' if existing_text else '')
                cand = prefix + existing_text + glue + para.strip() + '\n'

            cands.append(cand)

        dedup = []
        seen = set()
        for c in cands:
            k = c.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            dedup.append(c)

        if n_max_propose != -1:
            dedup = dedup[:n_max_propose]
        return dedup

    def value_prompt_wrap(self, x: str, y: str) -> str:
        y = (y or '').strip()
        if not y:
            y = '(empty)'

        if 'Passage:' in y:
            passage = self._passage_split_re.split(y)[-1].strip()
            # If passage is still empty (placeholder), score the plan instead.
            if passage:
                paragraphs = _split_paragraphs(passage)
                if len(paragraphs) >= 4:
                    return (
                        'Output ONLY one line in exactly this format (no other text): "Thus the coherency score is {s}"\n'
                        + score_prompt
                        + passage
                    )

                required = _extract_required_end_sentences(x)
                if len(required) < 4:
                    required = required + [''] * (4 - len(required))
                remaining = required[len(paragraphs):]
                remaining_lines = ''.join([f'- Remaining paragraph {len(paragraphs) + i + 1} must end with: {s}\n' for i, s in enumerate(remaining)])
                plan_part = y.split('Passage:', 1)[0].strip()
                passage_short = _truncate_paragraphs(paragraphs, max_paragraphs=2, max_chars=1200)
                return (
                    'Score how coherent the passage so far is, and how likely it can be completed to satisfy the strict ending-sentence constraints.\n'
                    'Output ONLY one line in exactly this format (no other text): "Thus the coherency score is {s}"\n'
                    'Constraints:\n'
                    + remaining_lines
                    + '\nPlan:\n'
                    + plan_part
                    + '\n\nPassage so far:\n'
                    + passage_short
                )

            plan_only = y.split('Passage:', 1)[0].strip()
            y = plan_only if plan_only else y

        return (
            'Score how promising this plan is for writing a coherent passage that satisfies the instruction.\n'
            'Output ONLY one line in exactly this format (no other text): "Thus the coherency score is {s}"\n'
            'Plan:\n'
            + y
        )

    def value_outputs_unwrap(self, x: str, y: str, value_outputs: list) -> float:
        scores = []
        for out in value_outputs or []:
            t = (out or '').strip()
            m = self._score_re.search(t)
            if not m:
                continue
            try:
                s = int(m.group(1))
            except Exception:
                continue
            if s < 1:
                s = 1
            if s > 10:
                s = 10
            scores.append(s)
        return float(sum(scores) / len(scores)) if scores else 0.0
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1