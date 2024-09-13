import os
import json
import logging
from time import sleep
import openai
import opencc
from tqdm import tqdm
from .evaluator import Evaluator


class ChatGPT_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(ChatGPT_Evaluator, self).__init__(choices, model_name, k)
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.converter = None
        self.switch_zh_hans = switch_zh_hans
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")
        self.remove_all_params = False
        if 'o1' in self.model_name:
            self.remove_all_params = True

    def format_example(self, line, include_answer=True, cot=False):
        example = line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += "\n答案："
        if include_answer:
            if cot:
                ans = line["answer"]
                content = "讓我們一步一步思考，\n" + line["explanation"] + f"\n所以答案是{ans}。"
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": content},
                ]
            else:
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": line["answer"]},
                ]
        else:
            return [
                {"role": "user", "content": example},
            ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = [
            {
                "role": "system",
                "content": f"你是一位專業的中文AI助理，以下是關於{subject}考試單選題，請選出正確的答案。",
            }
        ]
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            if i == 0:
                tmp[0]["content"] = (
                    f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp[0]["content"]
                )
                if self.converter:
                    tmp[0]["content"] = self.converter.convert(tmp[0]["content"])
                prompt += tmp

        return prompt

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=False,
    ):
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(
                subject_name, dev_df, cot=cot
            )
        else:
            few_shot_prompt = [
                {
                    "role": "system",
                    "content": f"你是一位專業的中文AI助理，以下是關於{subject_name}考試單選題，請選出正確的答案。",
                }
            ]
        system_prompt = None
        if self.remove_all_params:
            system_prompt = few_shot_prompt[0]['content']
            few_shot_prompt = []
        answers = list(test_df["answer"])
        added = set()
        with open('cache.jsonl', 'r') as f:
            for line in f:
                payload = json.loads(line)
                if 'subject_name' in payload and payload['subject_name'] == subject_name:
                    added.add(payload['row_id'])
                    correct_num += payload['correct']
                    result.append(payload['response'])
                    score.append(payload['correct'])

        for row_index, row in tqdm(
            test_df.iterrows(), total=len(test_df), dynamic_ncols=True
        ):
            if row_index in added:
                continue
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            if system_prompt:
                full_prompt[-1]["content"] = (
                    system_prompt + '\n' + full_prompt[-1]["content"]
                )
            if not few_shot:
                full_prompt[-1]["content"] = (
                    f"以下是關於{subject_name}考試單選題，請選出正確的答案。\n\n"
                    + full_prompt[-1]["content"]
                )
            response = None
            timeout_counter = 0
            if self.converter:  # convert to simplified chinese
                for idx, prompt in enumerate(full_prompt):
                    full_prompt[idx]["content"] = self.converter.convert(
                        prompt["content"]
                    )
            params = {
                'temperature': 0.0,
                'max_tokens': 800
            }
            if self.remove_all_params:
                params = {}
            while response is None and timeout_counter <= 30:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=full_prompt,
                        **params
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    logging.error(msg)
                    sleep(5)
                    continue
            if response == None:
                response_str = ""
            else:
                response_str = response.choices[0].message.content
            raw_response = response_str
            if cot:
                ans_list = self.cot_match_response_choice(response_str,
                            is_simplified= self.switch_zh_hans)

                if len(ans_list) == 0:
                    correct = 0
                else:
                    if self.exact_match(ans_list[-1], row["answer"]):
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
            else:
                response_str = response_str.strip()
                if len(response_str) > 0:
                    ans_list = self.extract_ans(response_str)
                    if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
                else:
                    correct = 0

            with open('cache.jsonl', 'a') as f:
                f.write(json.dumps({
                    'model':self.model_name,
                    'question': question,
                    'prompt': full_prompt[-1],
                    'raw_answer': raw_response,
                    'response': response_str,
                    'correct': correct,
                    'subject_name': subject_name,
                    'row_id': row_index,
                    'answer': row["answer"]
                })+'\n')
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_val.csv"),
                encoding="utf-8",
                index=False,
            )
        return correct_ratio

