import pathlib, random, string
import os
import warnings
os.environ['PROJ_LIB'] = '/usr/local/envs/zrp_env/share/proj'
warnings.filterwarnings("ignore", category=DeprecationWarning)
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.metrics import accuracy_score, f1_score, log_loss
from zrp import ZRP

# ───────────── paths / config ─────────────────────────────────────
HERE        = pathlib.Path(__file__).resolve().parent
print(f"HERE: {HERE}")
VOTER_CSV   = HERE / "nc_voter_cleaned_2022.csv"
ZCTA_DIR    = HERE / "tl_2022_us_zcta520"
ART_DIR     = HERE / "artifacts" ; ART_DIR.mkdir(exist_ok=True)
PRED_DIR    = HERE / "preds"     ; PRED_DIR.mkdir(exist_ok=True)
OUT_METRIC  = HERE / "zrp_metrics_partA.csv"

GRID = [(10,0), (20,0), (0,5)]

SEED      = 0                    # single replica
EARTH_MI  = 3958.8
LETTERS   = string.ascii_uppercase
BUCKETS   = ["WHITE","BLACK","HISPANIC","AAPI","OTHER"]
lab2idx   = dict(White=0, Black=1, Hispanic=2, AAPI=3, Other=4)

# ───────────── helper fns ─────────────────────────────────────────
def bucketize(r,e):
    r,e = r.upper(), e.upper()
    if r=="W":  return "White"
    if r=="B":  return "Black"
    if r=="A":  return "AAPI"
    if e=="HL": return "Hispanic"
    return "Other"

def ece10(p,y):
    conf=p.max(1); pred=p.argmax(1); acc=(pred==y).astype(float)
    bins=np.linspace(0,1,11); e=0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any():
            e+=abs(conf[m].mean()-acc[m].mean())*m.mean()
    return e

def metrics(df):
    ok=df[BUCKETS].notna().all(1)
    yp=df.loc[ok,BUCKETS].to_numpy(float)
    yp/=yp.sum(1,keepdims=True)
    yt=df.loc[ok,"bucket"].map(lab2idx).to_numpy()
    return dict(
        coverage = ok.mean(),
        accuracy = accuracy_score(yt, yp.argmax(1)),
        macro_f1 = f1_score(yt, yp.argmax(1), average="macro"),
        logloss  = log_loss(yt, yp),
        ece10    = ece10(yp, yt),
        dropped  = len(df)-ok.sum()
    )

def swap_adjacent(s):
    if len(s)<2: return s
    i=random.randint(0,len(s)-2); lst=list(s)
    lst[i],lst[i+1] = lst[i+1],lst[i]
    return "".join(lst)
def delete_char(s):
    if len(s)<2: return s
    i=random.randrange(len(s)); return s[:i]+s[i+1:]
def add_char(s):
    i=random.randrange(len(s)+1)
    return s[:i]+random.choice(LETTERS)+s[i:]
PERTURB_FN=[swap_adjacent,delete_char,add_char]

# ───────────── BallTree of 10-mile neighbours ────────────────────
print("• building BallTree …")
zcta=gpd.read_file(next(ZCTA_DIR.glob("*.shp")))
for c in ("ZCTA5CE10","ZCTA5CE20","GEOID10","GEOID20","ZCTA5CE"):
    if c in zcta.columns:
        zcta=zcta.rename(columns={c:"zip"})
        break
zcta=zcta[zcta["zip"].str.startswith("27")].reset_index(drop=True)
zcta["centroid"]=zcta.geometry.centroid
coords=np.vstack([zcta.centroid.y,zcta.centroid.x]).T
tree=BallTree(np.deg2rad(coords),metric="haversine")
def neighbours_10mi(z):
    row=zcta.loc[zcta["zip"]==z]
    if row.empty: return [z]
    idx=row.index[0]
    ind=tree.query_radius(coords[[idx]], r=10/EARTH_MI)[0]
    return zcta.iloc[ind]["zip"].tolist()
ZIP_NEIGHB={z: neighbours_10mi(z) for z in zcta["zip"]}
def pert_zip(z):
    nb=[n for n in ZIP_NEIGHB.get(z,[z]) if n!=z] or [z]
    return random.choice(nb)

# ───────────── load voter file once ───────────────────────────────
print("• loading voter CSV …")
voters=pd.read_csv(VOTER_CSV,dtype={"zip_code":str})
voters["bucket"]=voters.apply(lambda r: bucketize(r["race_code"],
                                                  r["ethnic_code"]),axis=1)
voters["ZEST_KEY"]=voters["ncid"].astype(str)

rows=[]

# ───────────── main experiment loop ───────────────────────────────
for alpha,gamma in GRID:
    feat = ART_DIR/f"proxy_output_exp_{alpha}_{gamma}.feather"
    if feat.exists():
        print(f"✓ exp_{alpha}_{gamma} already done")
        continue

    random.seed(SEED); np.random.seed(SEED)
    print(f"\nα={alpha}%  γ={gamma}%  (single seed)")
    df=voters.copy()

    # balanced sampling mask
    def mask(pct):
        m=np.zeros(len(df),bool)
        for b,grp in df.groupby("bucket",observed=True):
            n=int(len(grp)*pct/100)
            if n: m[np.random.choice(grp.index,n,False)]=True
        return m

    if alpha:
        m=mask(alpha)
        df.loc[m,"zip_code"]=df.loc[m,"zip_code"].apply(pert_zip)
    if gamma:
        m=mask(gamma)
        df.loc[m,"surname"]=df.loc[m,"surname"].str.upper().apply(
            lambda s: random.choice(PERTURB_FN)(s))

    zin=pd.DataFrame({
        "first_name":df["first"],
        "last_name" :df["surname"],
        "state"     :"NC",
        "zip_code"  :df["zip_code"].str.zfill(5),
        "ZEST_KEY"  :df["ZEST_KEY"],
        "middle_name":"","house_number":"","street_address":""
    })

    zest=ZRP(geocode=False,bisg=False,
             runname=f"exp_{alpha}_{gamma}")
    zest.fit()
    preds=zest.transform(zin)
    preds.columns=preds.columns.str.upper()

    # save artefacts
    preds.to_feather(feat)
    pd.concat([df.reset_index(drop=True),preds],axis=1)\
      .to_csv(PRED_DIR/f"exp_{alpha}_{gamma}.csv.gz",
              index=False,compression="gzip")

    merged=df[["ZEST_KEY","bucket"]].merge(preds,
                                           on="ZEST_KEY",how="inner")
    row=metrics(merged)
    row.update(alpha=alpha,gamma=gamma)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_METRIC,index=False)

print(f"\n✅ part-A finished → {OUT_METRIC}")
