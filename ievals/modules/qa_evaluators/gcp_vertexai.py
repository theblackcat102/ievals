import os
import opencc
from tqdm import tqdm
import vertexai
from google import genai
from google.genai import types
from .evaluator import Evaluator
from ..answer_parser import cot_match_response_choice
from ...helper import retry_with_exponential_backoff
if 'GCP_PROJECT_NAME' in os.environ:
    vertexai.init(project=os.environ['GCP_PROJECT_NAME'], location="us-central1")

class Vertex_Evaluator(Evaluator):

    SAFETY_SETTINGS=[types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )]

    def __init__(self, choices, k, api_key=None, model_name='gemini-1.5-flash', switch_zh_hans=False):
        super(Vertex_Evaluator, self).__init__(choices, model_name, k)
        self.client = genai.Client(
            vertexai=True,
            project=os.environ['GCP_PROJECT_NAME'],
            location="us-central1",
        )
        # self.model = GenerativeModel(model_name)

        self.model_name = model_name
        self.converter = None
        self.switch_zh_hans = switch_zh_hans
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")

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
            prompt += tmp
        return prompt

    def convert_prompt_to_contents(self, prompt):
        """Convert the old prompt format to the new contents format for the Vertex AI API"""
        contents = []
        current_content = None
        current_role = None
        
        for message in prompt:
            role = message["role"]
            content = message["content"]
            
            # Convert 'system' role to 'user' as Vertex AI doesn't have a system role
            if role == "system":
                role = "user"
            
            # Map to the new API's role naming
            if role == "user":
                api_role = "user"
            elif role == "assistant":
                api_role = "model"
            else:
                api_role = role
            
            # If this is a new role, create a new content object
            if api_role != current_role:
                if current_content is not None:
                    contents.append(current_content)
                
                current_content = types.Content(
                    role=api_role,
                    parts=[types.Part.from_text(text=content)]
                )
                current_role = api_role
            else:
                # Same role, append to parts
                current_content.parts.append(types.Part.from_text(text=content))
        
        # Add the last content object if it exists
        if current_content is not None:
            contents.append(current_content)
        
        return contents
    
    @retry_with_exponential_backoff
    def infer_with_backoff(self, prompt, max_tokens=1024, temperature=0.0, top_p=1, top_k=1):
        # Convert the prompt to the new format
        if isinstance(prompt, list):
            # This is the old format with roles
            contents = self.convert_prompt_to_contents(prompt)
        else:
            # This is already in string format (from the text array in eval_subject)
            print(prompt)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
        
        # Setup generation config
        generate_content_config = types.GenerateContentConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_output_tokens=int(max_tokens),
            safety_settings=self.SAFETY_SETTINGS
        )
        
        # Generate content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generate_content_config
        )
        
        # Get the result text
        result = response.text
        
        # Count tokens for both input and output
        input_tokens = self.client.models.count_tokens(model=self.model_name, contents=contents).total_tokens
        output_tokens = self.client.models.count_tokens(model=self.model_name, contents=[
            types.Content(role="model", parts=[types.Part.from_text(text=result)])
        ]).total_tokens
        
        res_info = {
            "input": prompt,
            "output": result,
            "num_input_tokens": input_tokens,
            "num_output_tokens": output_tokens,
            "logprobs": []  # NOTE: currently the Gemini API does not provide logprobs
        }
        
        return result, res_info


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
        answers = list(test_df["answer"])
        for row_index, row in tqdm(
            test_df.iterrows(), total=len(test_df), dynamic_ncols=True
        ):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"] = (
                    f"以下是關於{subject_name}考試單選題，請選出正確的答案。\n\n"
                    + full_prompt[-1]["content"]
                )
            for prompt in full_prompt:
                if self.converter:
                    prompt["content"] = self.converter.convert(prompt["content"])

            response_str, _ = self.infer_with_backoff(full_prompt, max_tokens=512, temperature=0)

            if cot:
                ans_list = cot_match_response_choice(response_str,
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
                if few_shot:
                    if len(response_str) > 0:
                        if self.exact_match(response_str, row["answer"]):
                            correct_num += 1
                            correct = 1
                        else:
                            ans_list = self.extract_ans(response_str)
                            if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                                correct_num += 1
                                correct = 1
                            else:
                                correct = 0
                    else:
                        correct = 0
                else:
                    if len(response_str) > 0:
                        ans_list = self.extract_ans(response_str)
                        if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                            correct_num += 1
                            correct = 1
                        else:
                            correct = 0
                    else:
                        correct = 0
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

