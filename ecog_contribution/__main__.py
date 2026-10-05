import argparse
from .artifacts import ROOT,DEFAULT_OUTPUT


def main():
    parser=argparse.ArgumentParser(description="Local, exploratory ECoG brain-hand contribution suite")
    commands=parser.add_subparsers(dest="command",required=True)
    run=commands.add_parser("run",help="Compute all analyses and presentation artifacts")
    run.add_argument("--data",default=str(ROOT/"source/ECoG_Handpose.mat"))
    run.add_argument("--output",default=str(DEFAULT_OUTPUT))
    args=parser.parse_args()
    from .suite import main_run
    main_run(args.data,args.output)


if __name__=="__main__":
    main()
