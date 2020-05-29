from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=str, help='path to input file')
    parser.add_argument('-o', dest='output_path', type=str, default=None, help='output filename')
    return parser.parse_args()

def run(args):
    special_tokens = set(['<SOS>', '<EOS>', '<PAD>'])
    with open(args.input_path, 'r') as f:
        lines = [line.strip().split() for line in f]
    lines = [line for line in lines if len(line) > 0]
    lines = [[word for word in line if word not in special_tokens] for line in lines]
    output_path = args.output_path
    if output_path is None:
        output_path = Path(args.input_path)
        output_path = output_path.parent / (output_path.stem + '_stripped.txt')
    lines = '\n'.join(' '.join(line) for line in lines)
    with open(output_path, 'w') as f:
        f.write(lines)

if __name__ == '__main__':
    run(parse_args())
