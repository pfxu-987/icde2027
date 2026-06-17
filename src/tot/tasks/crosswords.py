import json
import os
import re

from tot.models import gpt
from tot.prompts.crosswords import *
from tot.tasks.base import DATA_PATH, Task


_WORDLIST_VOCAB: set[str] | None = None
_WORDLIST_VOCAB_LIST: list[str] | None = None


def _matches_pattern(word: str, pattern: str) -> bool:
    if len(word) != 5 or len(pattern) != 5:
        return False
    for wc, pc in zip(word, pattern):
        if pc == "_":
            continue
        if wc != pc:
            return False
    return True


def _load_wordlist_vocab() -> tuple[set[str], list[str]]:
    global _WORDLIST_VOCAB, _WORDLIST_VOCAB_LIST
    if _WORDLIST_VOCAB is not None and _WORDLIST_VOCAB_LIST is not None:
        return _WORDLIST_VOCAB, _WORDLIST_VOCAB_LIST

    path = os.path.join(DATA_PATH, "crosswords", "mini0505_0_100_5.json")
    vocab: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for _clues, board in data:
            if not isinstance(board, list) or len(board) != 25:
                continue
            letters: list[str] = []
            for c in board:
                s = str(c).strip().lower()
                if len(s) != 1 or not s.isalpha():
                    s = "_"
                letters.append(s)
            for i in range(5):
                w = "".join(letters[i * 5 : (i + 1) * 5])
                if len(w) == 5 and w.isalpha():
                    vocab.add(w)
            for j in range(5):
                w = "".join(letters[j::5])
                if len(w) == 5 and w.isalpha():
                    vocab.add(w)
    except Exception:
        vocab = set()

    _WORDLIST_VOCAB = vocab
    _WORDLIST_VOCAB_LIST = sorted(vocab)
    return _WORDLIST_VOCAB, _WORDLIST_VOCAB_LIST


class MiniCrosswordsEnv:
    def __init__(self, file: str = "mini0505.json"):
        self.file = os.path.join(DATA_PATH, "crosswords", file)
        self.file = json.load(open(self.file))
        self.n = len(self.file)
        self.idx = None
        self.times = 0
        self.prompt_status_cache = {}
        self.backend = None

    def __len__(self):
        return self.n

    def reset(self, idx, board=None, status=None, steps=0):
        self.idx = idx
        self.data, self.board_gt = self.file[idx]
        self.board_gt = [_.lower() for _ in self.board_gt]
        self.board = board or ["_"] * 25
        self.status = status or [0] * 10
        self.steps = steps
        self.ans_gt = self.get_ans(self.board_gt)
        self.ans = self.get_ans(self.board)
        self.new_ans = []
        self.times += 1
        return self.render()

    def render_board(self):
        s = "Current Board:\n"
        for i in range(5):
            s += "".join(self.board[i * 5 : (i + 1) * 5]).upper() + "\n"
        return s

    def render_clues(self, status=None):
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += "h" + str(i + 1) + ". " + self.data[i] + "\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += "v" + str(i - 5 + 1) + ". " + self.data[i] + "\n"
        return s

    def render_ans(self, status=None):
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += "h" + str(i + 1) + ". " + self.data[i] + ": " + self.ans[i] + "\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += "v" + str(i - 5 + 1) + ". " + self.data[i] + ": " + self.ans[i] + "\n"
        return s

    def render(self, status=True):
        if status:
            return (
                self.render_board()
                + "\nUnfilled:\n"
                + self.render_ans(status=0)
                + "\nFilled:\n"
                + self.render_ans(status=1)
                + "\nChanged:\n"
                + self.render_ans(status=2)
            )
        return self.render_board() + "\n" + self.render_ans()

    def get_ans(self, board):
        ans = [""] * 10
        for i in range(5):
            ans[i] = "".join(board[i * 5 : (i + 1) * 5])
        for i in range(5):
            ans[i + 5] = "".join(board[i::5])
        return ans

    def step(self, action: str):
        self.steps += 1
        action = action.split("\n")[-1]
        action = action.split(". ")
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = action

        if len(word) != 5:
            return "Invalid! Word should have 5 letters.", 0, False, {}
        if pos.startswith("h"):
            idx = int(pos[1:]) - 1
            self.board[idx * 5 : (idx + 1) * 5] = list(word.lower())
        elif pos.startswith("v"):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.lower())
            idx += 5
        else:
            return "Invalid! Position should be h1-h5 or v1-v5", 0, False, {}

        self.new_ans = self.get_ans(self.board)
        self.status = [
            2 if any(letter != new_letter and letter != "_" for letter, new_letter in zip(ans, new_ans)) else status
            for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)
        ]
        self.status[idx] = 1
        self.ans = self.new_ans
        r_all = self.board == self.board_gt
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return self.render(), r_all, (r_all or self.steps >= 20), {"r_letter": r_letter, "r_word": r_word, "r_game": r_all}

    def prompt_status(self):
        count = {"sure": 0, "maybe": 0, "impossible": 0}
        for ans, data, status in zip(self.ans, self.data, self.status):
            if ans.count("_") >= 4:
                continue
            ans_spaced = " ".join(ans.lower())
            line = f"{data}: {ans_spaced}"
            prompt = value_prompt.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                model = self.backend or os.getenv("TOT_BACKEND", "qwen3-32b-vllm")
                res = gpt(prompt, model=model)[0]
                self.prompt_status_cache[prompt] = res
            out = (res or "").strip().lower()
            last = out.split("\n")[-1].strip()
            if last in count:
                count[last] += 1
            elif "impossible" in out:
                count["impossible"] += 1
            elif "sure" in out and "impossible" not in out:
                count["sure"] += 1
            else:
                count["maybe"] += 1
        return count


class MiniCrosswordsTask(Task):
    def __init__(self, file: str = "mini0505.json"):
        super().__init__()
        self.env = MiniCrosswordsEnv(file)
        self.xs = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())

        self.steps = 10
        self.stops = ["\n"] * self.steps
        self.cache_proposals = {}
        self.value_cache = {}

        self.backend = os.getenv("TOT_BACKEND", "qwen3-32b-vllm")
        self.temperature = 0.7

        self.propose_stop = None
        self.propose_max_tokens = 96
        self.propose_temperature = 0.7
        self.n_propose_sample = 5
        self.n_max_propose = 20

        self.value_stop = "\n"
        self.value_max_tokens = 8
        self.value_temperature = 0.0

        self.word_vocab, self.word_vocab_list = _load_wordlist_vocab()

    def __len__(self) -> int:
        return len(self.env)

    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()

    def test_output(self, idx: int, output: str):
        self.env.reset(idx)
        output = output.split("Output:\n")[-1]
        info = {"r_word": 0, "r_letter": 0, "r_game": 0}

        lines = [ln.strip() for ln in (output or "").split("\n") if ln.strip()]
        has_actions = any(ln.lower().startswith("h") or ln.lower().startswith("v") for ln in lines)
        if has_actions:
            for line in lines:
                if not (line.lower().startswith("h") or line.lower().startswith("v")):
                    continue
                _obs, _r_all, _done, step_info = self.env.step(line)
                if isinstance(step_info, dict) and "r_word" in step_info:
                    info = step_info
            info["r"] = info.get("r_word", 0)
            return info

        for i, line in enumerate((output or "").strip().split("\n")[-5:], 1):
            letters = line.split(" ")[:5]
            word = "".join(letters)
            word = word + "_" * (5 - len(word))
            action = f"h{i}. {word}"
            _obs, _r_all, _done, step_info = self.env.step(action)
            if isinstance(step_info, dict) and "r_word" in step_info:
                info = step_info
        info["r"] = info.get("r_word", 0)
        return info

    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self.env.backend = getattr(self, "backend", None) or os.getenv("TOT_BACKEND", "qwen3-32b-vllm")
        self.test_output(idx, y)

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    def propose_prompt_wrap(self, x: str, y: str = '') -> str:
        self.set_status(x, y)
        prompt = propose_prompt.format(input=self.env.render())
        prompt += (
            "\nReturn ONLY candidate lines (no markdown, no explanations).\n"
            "Each line MUST be exactly: <h1-h5 or v1-v5>. <5 letters> (certain/high/medium/low)\n"
            "Do NOT output any other text.\n"
        )
        return prompt

    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
        if (x, y, n_max_propose) in self.cache_proposals:
            return self.cache_proposals[(x, y, n_max_propose)]

        try:
            self.set_status(x, y)
        except Exception:
            pass

        confidence_to_value = {"certain": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}
        proposals_to_scores: dict[str, float] = {}

        pos_header_re = re.compile(r"\b([hv][1-5])\.", flags=re.IGNORECASE)
        proposal_re = re.compile(
            r"([hv][1-5])\.\s*[`*]{0,3}\s*([a-zA-Z]{5})\s*[`*]{0,3}\s*"
            r"(?:\(|\*\(|\*\s*\(|\*\s*)\s*(certain|high|medium|low)\s*(?:\)|\)\*|\*\s*\)|\*)",
            flags=re.IGNORECASE,
        )
        word_only_re = re.compile(
            r"\b[`*]{0,3}\s*([a-zA-Z]{5})\s*[`*]{0,3}\s*"
            r"(?:\(|\*\(|\*\s*\(|\*\s*)\s*(certain|high|medium|low)\s*(?:\)|\)\*|\*\s*\)|\*)",
            flags=re.IGNORECASE,
        )
        no_conf_re = re.compile(
            r"\b([hv][1-5])\.\s*[`*]{0,3}\s*([a-zA-Z]{5})\s*[`*]{0,3}\b",
            flags=re.IGNORECASE,
        )

        def _slot_index(pos: str) -> int | None:
            try:
                pos = pos.lower()
                if pos.startswith("h"):
                    return int(pos[1:]) - 1
                if pos.startswith("v"):
                    return 5 + int(pos[1:]) - 1
            except Exception:
                return None
            return None

        for output in outputs or []:
            current_pos: str | None = None
            for line in (output or "").split("\n"):
                text = (line or "").strip()

                mpos = pos_header_re.search(text)
                if mpos:
                    current_pos = mpos.group(1).lower()

                m = proposal_re.search(text)
                if m:
                    pos, word, conf = m.group(1), m.group(2), m.group(3)
                    si = _slot_index(pos)
                    if si is not None and 0 <= si < len(self.env.status) and self.env.status[si] != 0:
                        continue
                    proposal = f"{pos.lower()}. {word.lower()}"
                    proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0.0) + confidence_to_value.get(conf.lower(), 0.0)
                    continue

                m0 = no_conf_re.search(text)
                if m0:
                    pos, word = m0.group(1), m0.group(2)
                    si = _slot_index(pos)
                    if si is not None and 0 <= si < len(self.env.status) and self.env.status[si] != 0:
                        continue
                    proposal = f"{pos.lower()}. {word.lower()}"
                    proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0.0) + confidence_to_value.get("low", 0.0)
                    continue

                if current_pos:
                    m2 = word_only_re.search(text)
                    if not m2:
                        continue
                    word, conf = m2.group(1), m2.group(2)
                    si = _slot_index(current_pos)
                    if si is not None and 0 <= si < len(self.env.status) and self.env.status[si] != 0:
                        continue
                    proposal = f"{current_pos}. {word.lower()}"
                    proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0.0) + confidence_to_value.get(conf.lower(), 0.0)

        try:
            if self.word_vocab_list:
                max_per_slot = 8
                for i, ans in enumerate(self.env.ans):
                    if i < len(self.env.status) and self.env.status[i] != 0:
                        continue
                    pat = (ans or "").strip().lower()
                    if len(pat) != 5:
                        continue
                    unknown = pat.count("_")
                    if unknown == 5:
                        continue
                    if unknown >= 4:
                        continue
                    pos = f"h{i+1}" if i < 5 else f"v{i-4}"
                    added = 0
                    for w in self.word_vocab_list:
                        if _matches_pattern(w, pat):
                            proposal = f"{pos}. {w}"
                            proposals_to_scores.setdefault(proposal, 0.05)
                            added += 1
                            if added >= max_per_slot:
                                break
        except Exception:
            pass

        proposals = sorted(proposals_to_scores.items(), key=lambda x: x[1], reverse=True)
        if n_max_propose != -1:
            proposals = proposals[:n_max_propose]

        # Optional dictionary filter (soft): keep only words that exist in the crossword wordlist.
        # If filtering would remove everything, fall back to unfiltered proposals so the search
        # can continue and let fast_value prune aggressively.
        if getattr(self, 'word_vocab', None):
            filtered = [p for p in proposals if p[0].split('. ', 1)[-1] in self.word_vocab]
            if filtered:
                proposals = filtered
                if n_max_propose != -1:
                    proposals = proposals[:n_max_propose]

        proposals = [y + proposal[0] + '\n' for proposal in proposals]
        self.cache_proposals[(x, y, n_max_propose)] = proposals
        return proposals

    def value_prompt_wrap(self, x: str, y: str) -> str:
        self.set_status(x, y)
        lines = []
        for ans, clue, _status in zip(self.env.ans, self.env.data, self.env.status):
            if ans.count("_") >= 4:
                continue
            ans = " ".join(ans.lower())
            lines.append(f"{clue}: {ans}")
        input_block = "\n".join(lines)
        return (
            value_prompt.format(input=input_block)
            + "\nReturn ONLY labels (sure/maybe/impossible), one per line, in the same order as the lines above.\n"
        )

    def fast_value(self, x: str, y: str) -> float:
        self.set_status(x, y)
        if not self.word_vocab_list:
            return 0.0
        count = {"sure": 0, "maybe": 0, "impossible": 0}
        for ans, _clue, _status in zip(self.env.ans, self.env.data, self.env.status):
            if ans.count("_") >= 4:
                continue
            pat = ans.lower()
            matches = 0
            for w in self.word_vocab_list:
                if _matches_pattern(w, pat):
                    matches += 1
                    if matches > 1:
                        break
            if matches == 0:
                count["impossible"] += 1
            elif matches == 1:
                count["sure"] += 1
            else:
                count["maybe"] += 1
        value = 20.0 * count.get("sure", 0) + 1.0 * count.get("maybe", 0) - 20.0 * count.get("impossible", 0)
        return float(value)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if not value_outputs:
            return 0.0
        text = "\n".join([str(o or "") for o in value_outputs]).strip().lower()
        raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        labels = []
        for ln in raw_lines:
            if ln in {"sure", "maybe", "impossible"}:
                labels.append(ln)
            else:
                if "impossible" in ln:
                    labels.append("impossible")
                elif "sure" in ln and "impossible" not in ln:
                    labels.append("sure")
                elif "maybe" in ln:
                    labels.append("maybe")
        if not labels:
            if "impossible" in text:
                labels = ["impossible"]
            elif "sure" in text and "impossible" not in text:
                labels = ["sure"]
            else:
                labels = ["maybe"]
        count = {"sure": 0, "maybe": 0, "impossible": 0}
        for lab in labels:
            if lab in count:
                count[lab] += 1
        value = 1.0 * count.get("sure", 0) + 0.2 * count.get("maybe", 0) - 2.0 * count.get("impossible", 0)
        return float(value)

    def is_solved(self, idx: int, y: str) -> bool:
        info = self.test_output(idx, y)
        if info.get("r_game"):
            return True
        return info.get("r_word", 0) >= 1.0
