"""Extra checks for finger_context.py: within-gesture finger prediction, and whether
the fist/peace decoder errors fall on unusual index+middle postures more than chance."""
import numpy as np, json
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
import common as C
y,cue,glove,on,off,labels=C.load_raw()
fs=C.fs
hold=np.array([glove[:,o+int(.75*fs):o+int(1.75*fs)].mean(1) for o in on])
X=C.decoder_features(C.band_power('highgamma',True),on)
ridge=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,5,25)))
out={}
print('Within-gesture: brain -> finger bend (cross-validated r, 5 repeats)')
for k,n in [(1,'Fist'),(2,'Peace'),(3,'Open')]:
    m=labels==k; out[n]={}
    for i,f in enumerate(C.FINGERS):
        rs=[stats.pearsonr(cross_val_predict(ridge,X[m],hold[m,i],cv=KFold(10,shuffle=True,random_state=s)),hold[m,i]).statistic for s in range(5)]
        out[n][f]=float(np.mean(rs))
    print(f'  {n:6s}'+''.join(f'  {f} {r:+.2f}' for f,r in out[n].items()))
# errors: signed z toward the other gesture
im=hold[:,1:3].mean(1)
z=np.zeros(90)
for k in (1,2):
    m=labels==k; z[m]=(im[m]-im[m].mean())/im[m].std()
toward=np.where(labels==1,-z,z)   # positive = posture leaning toward the OTHER gesture
wrong=np.array([2,17,65,78])-1
obs=toward[wrong].mean()
rng=np.random.default_rng(0)
fi=np.where(labels==1)[0]; pi=np.where(labels==2)[0]
null=np.array([np.r_[toward[rng.choice(fi,3,replace=False)],toward[rng.choice(pi,1)]].mean() for _ in range(20000)])
p=(np.sum(null>=obs)+1)/20001
print(f'Errors lean toward the other gesture: mean z {obs:+.2f}; random same-size sets {null.mean():+.2f}; p={p:.4f}')
out['errors_toward_z']=float(obs); out['errors_p']=float(p)
json.dump(out,open('results/finger_extra.json','w'),indent=2)
