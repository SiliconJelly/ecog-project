import sys, os, runpy, time
os.environ.setdefault('OMP_NUM_THREADS','1')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
name=os.path.splitext(os.path.basename(sys.argv[1]))[0]
cnt=[0]
def show(*a,**k):
    for n in plt.get_fignums():
        cnt[0]+=1
        plt.figure(n).savefig(f'verify_out/{name}_fig{cnt[0]}.png',dpi=150,bbox_inches='tight')
    plt.close('all')
plt.show=show
t=time.time()
runpy.run_path(sys.argv[1],run_name='__main__')
print(f'[runtime {time.time()-t:.0f}s]')
