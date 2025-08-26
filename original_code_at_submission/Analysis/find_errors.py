import re
from argparse import ArgumentParser
from numpy.random import default_rng
import os
from pathlib import Path

rng = default_rng(42)
entry = re.compile("(IN:[^\\n]* \\|\\| «(.+)»\\s+(\\[.*]))\n-+\n")

a_parser = ArgumentParser()
a_parser.add_argument("-id", dest="input_ds", required=True)
a_parser.add_argument("-od", dest="out_dir", default=".")

def main(cl_args):
    input_dirs = cl_args.input_ds.split(";")
    for dir in input_dirs:
        m_out_files = {}
        for (parent, subdirs, filenames) in os.walk(dir):
            for f_name in [f for f in filenames if f.endswith('.txt')]:
                print(f_name)
                if re.search('synta1', parent):
                    code = re.sub("_output", "*_output", f_name)
                else:
                    code = f_name
                m_out_files[code[:-4]]= os.path.join(dir, parent, f_name)


        for lf_name, f_name in m_out_files.items():
            print(f_name)
            print(lf_name)
            with open(f_name) as f_in:
                txt = f_in.read()

            incorrects = []
            for (a, pred, gold) in entry.findall(txt):
                contained = False
                for g in eval(gold):
                    if re.search(re.escape(g), pred):
                        contained = True

                if not contained:
                    incorrects.append(a)


            d_out_name = os.path.join(cl_args.out_dir, re.sub("/(.)", "+\\1",
                                                              re.search("((qwen|gpt|llama|fast).*)outputs", dir)
                                                              .group(1)))
            Path(d_out_name).mkdir(parents=True, exist_ok=True)

            print(os.path.join(d_out_name, lf_name))
            print(os.getcwd())
            with open(os.path.join(d_out_name, re.sub("output", "errors.txt", lf_name)), 'w') as f_out:
                f_out.write("\n".join(incorrects))


            with open(os.path.join(d_out_name,
                                   re.sub("output", "errors_10_.txt", lf_name)), 'w') as f_out:
                n_samples = min(len(incorrects), 10)
                f_out.write("\n".join(rng.choice(incorrects, size=n_samples, replace=False)))





if __name__ == "__main__":
    cl_args = a_parser.parse_args()
    main(cl_args)