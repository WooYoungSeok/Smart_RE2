import os
import pickle
import re
import time

from tqdm import tqdm

def get_parse_fn(parsing_strategy):
    def parse_fn_math(output):
        cleaned = output.replace('*', '').replace('#', '').lower().split('answer: ')[-1].replace(',', '')
        match = re.search(r'-?[0-9.]*[0-9]', cleaned)
        if match:
            result = match.group()
            return re.sub(r"\.0+$", "", result)
        else:
            # 예상한 패턴이 없을 때 빈 문자열이나 적절한 기본값 반환
            return ""

    def parse_fn_multiple_choice(output):
        """Used for mmlu math and winograd schema challenge"""
        #return output.replace('*', '').lower().split("answer: ")[-1].replace(".", "").strip()[0:1].lower()

        x = output.replace('*', '').lower().split("answer: ")[-1].replace(".", "").strip()
        
        pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(pattern, x)
        if match:
            return match.group(1)[0:1]
        else:
            return x[0:1]
    
    def parse_bbh_multiple_choice(output):
        """Used for BBH multiple choice questions, where the answer is in the form (A)""" 
        result = output.replace('*', '').replace('#', '').lower().split('answer: ')[-1].replace('.', '').replace('\'', '').replace('\"', '').strip().lower()
        result = re.search(r'\([a-z]\)', result).group(0)
        return result

    def parse_fn_text(output):
        """Used by DROP and hotpotqa, where the answer is a string"""
        return (output.replace("#","").replace("*","").replace("\"", "").replace('\xa0', ' ')
                      .lower().split("answer: ")[-1].split('\n')[0].replace(",", "")
                      .replace(".","").split("}")[0].strip())
    
    def parse_fn_squad(output):
        """Like rext parsing, but explicitly handles the case when there is text after n/a"""
        output_clean = parse_fn_text(output)
        if output_clean.startswith('n/a '):
            return 'n/a'
        return output_clean
    
    def create_parse_fn(specific_parsing_fn):

        def parse_fn(output):
            # tex_pattern = r'\\boxed\{([^{}]+)\}|\\boxed\{\\text\{([^}]+)\}\}'
            tex_pattern = r'\\boxed\{(\\text\{)?([^\\{}]+)\}'

            # If answer is on the last line as expected, run as usual
            if "answer:" in output.lower().replace("*", ""):
                # If the answer is wrapped in latex (e.g., \boxed{...}), extract the content
                answer_section = output.lower().split("answer: ")[-1]
                if re.search(tex_pattern, answer_section):
                    match = re.search(tex_pattern, answer_section).group(2)
                    output = "Answer: " + match
            elif re.search(tex_pattern, output):
                # If the answer is not on the last line, try to recover by looking for a box
                output = "Answer: " + re.search(tex_pattern, output).group(2)
            else: 
                # Otherwise, just return the last line
                last_line = output.strip("\n").split("\n")[-1].lower()
                output = "Answer: " + last_line
            return specific_parsing_fn(output)
        
        return parse_fn

        
    if parsing_strategy == 'math':
        return create_parse_fn(parse_fn_math)
    elif parsing_strategy == 'multiple_choice':
        return create_parse_fn(parse_fn_multiple_choice)
    elif parsing_strategy == 'bbh_multiple_choice':
        return create_parse_fn(parse_bbh_multiple_choice)
    elif parsing_strategy == 'text':
        return create_parse_fn(parse_fn_text)
    elif parsing_strategy == 'squad':
        return create_parse_fn(parse_fn_squad)
    else:
        raise ValueError(f"Invalid parsing strategy: {parsing_strategy}")
    