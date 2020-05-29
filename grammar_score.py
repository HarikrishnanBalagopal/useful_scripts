import json

from pylanguagetool import api
from argparse import ArgumentParser\

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=str, help='path to input file')
    parser.add_argument('--strip', default=False, action='store_true', help='strip special tokens from input')
    parser.add_argument('-o', dest='output_path', type=str, default='grammar_checked.json', help='output filename')
    return parser.parse_args()

def get_lines(input_path):
    with open(input_path, 'r') as f:
        return f.read()

def strip_lines(lines):
    special_tokens = set(['<SOS>', '<EOS>', '<PAD>'])
    lines = [line.strip().split() for line in lines.split('\n')]
    lines = [[word for word in line if word not in special_tokens] for line in lines if len(line) > 0]
    return '\n'.join(' '.join(line) for line in lines)

def run(args):
    # for rules and rule ids see https://community.languagetool.org/rule/list?lang=en
    lines = get_lines(args.input_path)
    if args.strip:
        lines = strip_lines(lines)
    print(lines)
    results = api.check(
        input_text=lines,
        api_url='http://localhost:8081/v2/',
        lang='en',
        disabled_rules='UPPERCASE_SENTENCE_START,I_LOWERCASE,ENGLISH_WORD_REPEAT_BEGINNING_RULE,EN_COMPOUNDS,COMMA_PARENTHESIS_WHITESPACE',
        pwl=['UNK']
    )

    print('grammatical errors:', len(results['matches']))
    with open(args.output_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    run(parse_args())
