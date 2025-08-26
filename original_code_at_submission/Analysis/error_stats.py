import os
import re
from argparse import ArgumentParser

a_parser = ArgumentParser()
a_parser.add_argument("-id", dest="input_dirs", required=True)

annotation_re = re.compile('^\\s*(fn|tn)\\s*:\\s*([^\\s:#]*).*')
lf_name_re = re.compile('^(.*)_errors.*')

stats = {
    'tn': {'acception': [], 'oov': [], 'lf': [], 'pos': [], 'collocate':[], 'other': []},
    'fn': {'granularity': [], 'resource': [], 'other': []}
}

def main(cl_args):
    dirs =  cl_args.input_dirs.split(';')
    for dir in dirs:
        model = dir.split('+')[0]

        for fname in os.listdir(dir):
            lf = lf_name_re.match(fname).group(1)
            # if lf == 'Anti':
            #    continue

            with open(os.path.join(dir, fname), 'r') as f_in:
                for i, line in enumerate(f_in.readlines()):
                    m = annotation_re.match(line)
                    if m:
                        cat = m.group(2)
                        if m.group(2) not in stats[m.group(1)]:
                            cat = 'other'

                        stats[m.group(1)][cat].append((model, lf, i - 1, m.group(2)))


    print(stats['fn']['other'])
    print(stats['tn']['other'])


    total_fn = sum([len(stats['fn'][c]) for c in stats['fn']])
    total_tn = sum([len(stats['tn'][c]) for c in stats['tn']])
    totals = {'tn': {c: len(stats['tn'][c]) for c in stats['tn']},
              'fn': {c: len(stats['fn'][c]) for c in stats['fn']}
    }
    print("%_tn: {:f}".format(100 * total_tn/(total_fn + total_tn)))
    print("%_fn: {:f}".format(100 * total_fn/(total_fn + total_tn)))

    print('total', total_fn + total_tn)

    for x, t in [('tn', total_tn), ('fn', total_fn)]:
        for c in totals[x]:
            print("%{:s}/{:s}: {:f}".format(c, x, 100 * totals[x][c]/t))



if __name__ == "__main__":
    cl_args = a_parser.parse_args()
    main(cl_args)