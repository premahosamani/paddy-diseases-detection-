from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
# main.py  –  UAV Multispectral Disease Detection
# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT MODES
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. LOCAL  (same machine)
#    pip install fastapi uvicorn pillow numpy onnxruntime tifffile
#    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#    open → http://localhost:8000
#
# 2. LAN  (anyone in the same network/institute)
#    sudo ufw allow 8000/tcp          ← open firewall port once
#    uvicorn main:app --host 0.0.0.0 --port 8000
#    other PCs open → http://<YOUR_IP>:8000
#    find your IP:  hostname -I   (Linux)  |  ipconfig (Windows)
#
# 3. GOOGLE COLAB  (temporary public URL via ngrok)
#    pip install pyngrok nest_asyncio fastapi uvicorn pillow numpy onnxruntime
#    Set COLAB_MODE = True  below and run this file as a script:
#        !python main.py
#    A public URL like https://xxxx.ngrok.io will be printed.
#    Free ngrok: 1 tunnel, URL changes every restart.
#
# 4. HUGGING FACE SPACES  (persistent, free)
#    - Create a Space → SDK: Docker  or  SDK: Gradio (use Docker)
#    - Upload this main.py, index.html, dinov2.onnx, requirements.txt
#    - requirements.txt:
#        fastapi uvicorn pillow numpy onnxruntime tifffile python-multipart
#    - Dockerfile (copy from bottom of this file)
#    - Space runs on port 7860 by default → set PORT = 7860 below
#
# 5. RENDER.COM  (persistent, free tier — good for <100 MB TIFF uploads)
#    - New Web Service → connect GitHub repo
#    - Build command: pip install -r requirements.txt
#    - Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
#    - Render sets $PORT automatically
#    - Free tier sleeps after 15 min inactivity → upgrade for always-on
#
# ─────────────────────────────────────────────────────────────────────────────
# DOCKERFILE  (for Hugging Face Spaces / Render / any container platform)
# ─────────────────────────────────────────────────────────────────────────────
# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# EXPOSE 7860
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
# ═════════════════════════════════════════════════════════════════════════════

# ── ALL IMPORTS FIRST ────────────────────────────────────────────────────────
import os, io, json, traceback
import numpy as np
from PIL import Image, ImageFilter

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict

# ── DEPLOYMENT CONFIGURATION ─────────────────────────────────────────────────
# os is now imported — safe to use here
COLAB_MODE  = False   # Set True when running on Google Colab
NGROK_TOKEN = ""      # Free token at https://dashboard.ngrok.com/get-started/your-authtoken
PORT        = int(os.environ.get("PORT", 8000))   # Render/HF inject $PORT automatically
# ─────────────────────────────────────────────────────────────────────────────

# ── Upload limit patch (500 MB) ───────────────────────────────────────────────
import starlette.requests as _sr
_orig = _sr.Request._get_form
async def _big(self, *, max_files=1000, max_fields=1000,
               max_part_size=500*1024*1024):
    return await _orig(self, max_files=max_files,
                       max_fields=max_fields, max_part_size=max_part_size)
_sr.Request._get_form = _big

# ── tifffile ──────────────────────────────────────────────────────────────────
try:
    import tifffile; HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
import onnxruntime as ort
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "dinov2.onnx")

if os.path.exists(ONNX_PATH):
    _opts = ort.SessionOptions()
    _opts.intra_op_num_threads     = os.cpu_count() or 4
    _opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ORT_SESSION  = ort.InferenceSession(ONNX_PATH, sess_options=_opts,
                                        providers=["CPUExecutionProvider"])
    ORT_IN_NAME  = ORT_SESSION.get_inputs()[0].name
    MODEL_LOADED = True
    print(f"[ONNX] loaded: {ONNX_PATH}")
else:
    ORT_SESSION = None; ORT_IN_NAME = "pixel_values"; MODEL_LOADED = False
    print("[ONNX] not found – handcrafted features will be used")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="UAV Disease Detection", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_embeddings: Optional[Dict[str, list]] = None

TILES_DIR     = os.path.join(BASE_DIR, "tiles")
TILES_RAW_DIR = os.path.join(BASE_DIR, "tiles_raw")
UPLOADS_DIR   = os.path.join(BASE_DIR, "uploads")

EXTRACT_SIZE = 300
DISPLAY_SIZE = 768
DINO_SIZE    = 224
MIN_CROP_PX  = 500

CLUSTER_BALANCE_THRESH = 0.85
VEG_DISEASE_STDDEVS    = 1.5


# ═════════════════════════════════════════════════════════════════════════════
# VALID CROP PIXEL MASK
# ═════════════════════════════════════════════════════════════════════════════
def valid_mask(tile: np.ndarray) -> np.ndarray:
    R = tile[:,:,0].astype(np.int32)
    G = tile[:,:,1].astype(np.int32)
    B = tile[:,:,2].astype(np.int32)
    sat = np.maximum(np.maximum(R,G),B) - np.minimum(np.minimum(R,G),B)
    return (
        (G > R) & (G > B) &
        (G >= 30) & (G <= 220) &
        (G - R >= 8) & (G - B >= 5) &
        (sat >= 15)
    )

def enough_crop(mask: np.ndarray) -> bool:
    return int(mask.sum()) >= MIN_CROP_PX


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_image(data: bytes) -> np.ndarray:
    if HAS_TIFFFILE:
        try:
            a = tifffile.imread(io.BytesIO(data))
            if a.ndim == 4: a = a[0]
            if a.ndim == 3:
                if a.shape[0]<=10 and a.shape[1]>a.shape[0] and a.shape[2]>a.shape[0]:
                    a = np.transpose(a,(1,2,0))
                if a.shape[2]>3: a = a[:,:,:3]
                elif a.shape[2]==1: a = a[:,:,0]
            if a.ndim==2: a = np.stack([a]*3,2)
            a = a.astype(np.float32)
            lo,hi = a.min(),a.max()
            if hi>lo: a=(a-lo)/(hi-lo)*255
            return a.clip(0,255).astype(np.uint8)
        except Exception: pass
    img = Image.open(io.BytesIO(data))
    return np.array(img.convert("RGB"))

def get_dims(data):
    try: a=load_image(data); h,w=a.shape[:2]; return w,h
    except: return 0,0


# ═════════════════════════════════════════════════════════════════════════════
# FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def handcrafted(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = tile.astype(np.float32); eps = 1e-6
    R = arr[:,:,0][mask]; G = arr[:,:,1][mask]; B = arr[:,:,2][mask]
    if len(R) < 10: return np.zeros(107, dtype=np.float32)
    feats = [R.mean()/255,R.std()/255,G.mean()/255,G.std()/255,B.mean()/255,B.std()/255]
    ndvi=((G-R)/(G+R+eps)).mean(); exg=((2*G-R-B)/255).mean()
    gli=((2*G-R-B)/(2*G+R+B+eps)).mean(); vari=((G-R)/(G+R-B+eps)).mean()
    ngrdi=((G-R)/(G+R+eps)).mean()
    feats+=[float(ndvi),float(exg),float(gli),float(vari),float(ngrdi)]
    px=tile[mask].reshape(-1,1,3)
    if len(px)>0:
        hsv=np.array(Image.fromarray(px).convert("HSV"),dtype=np.float32).reshape(-1,3)
        for i in range(3): feats+=[hsv[:,i].mean()/255,hsv[:,i].std()/255]
    else: feats+=[0.0]*6
    for c in range(3):
        ch=arr[:,:,c].astype(np.uint8)
        lap=np.array(Image.fromarray(ch).filter(ImageFilter.FIND_EDGES),np.float32)
        feats.append(float(lap[mask].var())/(255.0**2))
    for ch_v in [R,G,B]:
        hist,_=np.histogram(ch_v,bins=16,range=(0,256))
        hist=hist.astype(np.float32)/(hist.sum()+eps); feats.extend(hist.tolist())
    h2,w2=arr.shape[0]//2,arr.shape[1]//2
    for sa,sm in [(arr[:h2,:w2],mask[:h2,:w2]),(arr[:h2,w2:],mask[:h2,w2:]),
                  (arr[h2:,:w2],mask[h2:,:w2]),(arr[h2:,w2:],mask[h2:,w2:])]:
        gv=sa[:,:,1][sm]; rv=sa[:,:,0][sm]
        feats+=[gv.mean()/255 if len(gv)>0 else 0.0,
                rv.mean()/255 if len(rv)>0 else 0.0,
                gv.std()/255  if len(gv)>0 else 0.0]
    return np.array(feats[:107], dtype=np.float32)

def preprocess_onnx(tile,mask):
    out=tile.copy().astype(np.float32)
    if mask.sum()>0:
        for c in range(3):
            mv=float(out[:,:,c][mask].mean())
            out[:,:,c]=np.where(mask,out[:,:,c],mv)
    img=Image.fromarray(out.clip(0,255).astype(np.uint8)).resize((DINO_SIZE,DINO_SIZE),Image.LANCZOS)
    a=np.array(img,dtype=np.float32)/255.0
    a=(a-_MEAN)/_STD
    return a.transpose(2,0,1)[np.newaxis].astype(np.float32)

def veg_score(tile,mask):
    if mask.sum()<10: return -999.0
    arr=tile.astype(np.float32); eps=1e-6
    R=arr[:,:,0][mask]; G=arr[:,:,1][mask]; B=arr[:,:,2][mask]
    return float(((G-R)/(G+R+eps)).mean())*0.4 + \
           float(((2*G-R-B)/255.0).mean())*0.35 + \
           float(((2*G-R-B)/(2*G+R+B+eps)).mean())*0.25


# ═════════════════════════════════════════════════════════════════════════════
# PCA + KMEANS
# ═════════════════════════════════════════════════════════════════════════════
def pca(X,n):
    n=min(n,X.shape[0]-1,X.shape[1])
    nrm=np.linalg.norm(X,axis=1,keepdims=True)
    X=X/np.where(nrm>0,nrm,1.0); mu=X.mean(0); Xc=X-mu
    _,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    vr=float((S[:n]**2).sum()/((S**2).sum()+1e-12))
    return (Xc@Vt[:n].T).astype(np.float32),vr

def scale(X):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1.0; return (X-mu)/sd

def kmeans(X,k=2,max_iter=500,n_init=15):
    best_lbl,best_ctr,best_in=None,None,np.inf
    rng=np.random.RandomState(42)
    for _ in range(n_init):
        idx=[rng.randint(0,len(X))]
        for _ in range(1,k):
            d=np.array([min(np.sum((x-X[i])**2) for i in idx) for x in X])
            idx.append(rng.choice(len(X),p=d/d.sum()))
        ctr=X[np.array(idx)].astype(np.float64); lbl=np.zeros(len(X),np.int32)
        for it in range(max_iter):
            d=np.stack([np.sum((X-c)**2,1) for c in ctr],1)
            nl=np.argmin(d,1).astype(np.int32)
            if np.array_equal(nl,lbl) and it>0: break
            lbl=nl
            nc=np.array([X[lbl==c].mean(0) if (lbl==c).any() else ctr[c] for c in range(k)])
            if np.max(np.linalg.norm(nc-ctr,axis=1))<1e-5: ctr=nc; break
            ctr=nc
        inn=sum(float(np.sum((X[lbl==c]-ctr[c])**2)) for c in range(k))
        if inn<best_in: best_in,best_lbl,best_ctr=inn,lbl.copy(),ctr.copy()
    return best_lbl,best_ctr,best_in

def get_conf(X,lbl,ctr):
    d0=np.sum((X-ctr[0])**2,1); d1=np.sum((X-ctr[1])**2,1)
    out=np.zeros(len(X),np.float32)
    for i,l in enumerate(lbl):
        own,oth=(d0[i],d1[i]) if l==0 else (d1[i],d0[i])
        tot=own+oth; out[i]=float(oth/tot) if tot>0 else 0.5
    return out

def label_by_veg_threshold(fnames,vscores):
    scores=np.array(vscores,dtype=np.float32)
    mean_s=scores.mean(); std_s=scores.std()
    thresh=mean_s-VEG_DISEASE_STDDEVS*std_s
    results,h_cnt,d_cnt=[],0,0
    for fname,score in zip(fnames,vscores):
        lab="diseased" if score<thresh else "healthy"
        conf=min(float(abs(score-thresh)/(std_s+1e-6)),1.0)
        if lab=="healthy": h_cnt+=1
        else: d_cnt+=1
        results.append({"filename":fname,"label":lab,"cluster":-1,"confidence":round(conf,3)})
    return results,h_cnt,d_cnt


# ═════════════════════════════════════════════════════════════════════════════
# HTML — served by FastAPI at GET /
# Accessible from any machine that can reach this server.
# ═════════════════════════════════════════════════════════════════════════════
def _html() -> HTMLResponse:
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/",                response_class=HTMLResponse)
def root():  return _html()          # ← main entry point for all clients

@app.get("/orthomosaic",     response_class=HTMLResponse)
def ortho(): return _html()

@app.get("/tile-extraction", response_class=HTMLResponse)
def te():    return _html()

@app.get("/dinov2",          response_class=HTMLResponse)
def d2():    return _html()

@app.get("/health")
def health():
    """Simple health-check endpoint — useful for Render / HF to verify app is up."""
    return {"status": "ok", "model_loaded": MODEL_LOADED,
            "onnx_path": ONNX_PATH, "port": PORT}


# ═════════════════════════════════════════════════════════════════════════════
# POST /upload
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        data=await file.read()
        os.makedirs(UPLOADS_DIR,exist_ok=True)
        for f in os.listdir(UPLOADS_DIR):
            try: os.remove(os.path.join(UPLOADS_DIR,f))
            except: pass
        name=os.path.basename(file.filename or "ortho.tif")
        open(os.path.join(UPLOADS_DIR,name),"wb").write(data)
        w,h=get_dims(data)
        return JSONResponse({"status":"success","filename":name,
                             "size_mb":round(len(data)/1024/1024,2),
                             "width":w,"height":h,"image_size":f"{w}x{h}"})
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error":str(e)},500)


# ═════════════════════════════════════════════════════════════════════════════
# POST /extract
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    global _embeddings
    try:
        data=await file.read()
        os.makedirs(UPLOADS_DIR,exist_ok=True)
        name=os.path.basename(file.filename or "ortho.tif")
        open(os.path.join(UPLOADS_DIR,name),"wb").write(data)
        _embeddings=None

        img=load_image(data); h,w=img.shape[:2]

        for d in [TILES_DIR,TILES_RAW_DIR]:
            os.makedirs(d,exist_ok=True)
            for f in os.listdir(d):
                try: os.remove(os.path.join(d,f))
                except: pass

        skip_set=set(); tiles,n=[],0

        for row in range(0,h,EXTRACT_SIZE):
            for col in range(0,w,EXTRACT_SIZE):
                tile=img[row:row+EXTRACT_SIZE,col:col+EXTRACT_SIZE].copy()
                ph=EXTRACT_SIZE-tile.shape[0]; pw=EXTRACT_SIZE-tile.shape[1]
                if ph>0 or pw>0:
                    tile=np.pad(tile,((0,ph),(0,pw),(0,0)),mode="constant")
                fname=f"tile_{n:04d}_r{row}_c{col}.tif"
                mask=valid_mask(tile)
                is_bg=not enough_crop(mask)
                if is_bg: skip_set.add(fname)
                Image.fromarray(tile).save(os.path.join(TILES_RAW_DIR,fname),
                                           format="TIFF",compression="raw")
                Image.fromarray(tile).resize((DISPLAY_SIZE,DISPLAY_SIZE),
                    Image.LANCZOS).save(os.path.join(TILES_DIR,fname),
                                        format="TIFF",compression="raw")
                tiles.append({"id":n,"filename":fname,"row":row,"col":col,
                               "is_background":is_bg})
                n+=1

        with open(os.path.join(BASE_DIR,"bg_tiles.json"),"w") as f:
            json.dump(list(skip_set),f)

        crop_n=n-len(skip_set)
        print(f"[extract] total={n} crop={crop_n} bg={len(skip_set)}")
        return JSONResponse({"status":"success","total_tiles":n,
                             "crop_tiles":crop_n,"background_tiles":len(skip_set),
                             "image_size":f"{w}x{h}",
                             "tile_size":f"{DISPLAY_SIZE}x{DISPLAY_SIZE}",
                             "tiles":tiles})
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error":str(e)},500)


# ═════════════════════════════════════════════════════════════════════════════
# POST /predict
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/predict")
async def predict():
    global _embeddings
    try:
        src=TILES_RAW_DIR if os.path.isdir(TILES_RAW_DIR) else TILES_DIR
        if not os.path.isdir(src):
            return JSONResponse({"error":"No tiles. Run /extract first."},400)
        files=sorted(f for f in os.listdir(src) if f.endswith(".tif"))
        if not files:
            return JSONResponse({"error":"Tiles empty."},400)

        skip_path=os.path.join(BASE_DIR,"bg_tiles.json")
        skip_set=set(json.load(open(skip_path))) if os.path.exists(skip_path) else set()

        embs,skipped={},0; mode="onnx" if MODEL_LOADED else "handcrafted"

        for fname in files:
            if fname in skip_set: skipped+=1; continue
            with open(os.path.join(src,fname),"rb") as f: tile=load_image(f.read())
            mask=valid_mask(tile)
            if not enough_crop(mask): skipped+=1; skip_set.add(fname); continue
            if MODEL_LOADED:
                inp=preprocess_onnx(tile,mask)
                outs=ORT_SESSION.run(None,{ORT_IN_NAME:inp}); out=outs[0]
                emb=(out[0,0,:] if out.ndim==3 else out[0,:] if out.ndim==2 else out.flatten())
            else:
                emb=handcrafted(tile,mask)
            embs[fname]=emb.tolist()

        _embeddings=embs
        print(f"[predict] mode={mode} crop={len(embs)} skipped={skipped}")
        return JSONResponse({"status":"success","green_tiles":len(embs),
                             "skipped":skipped,"mode":mode,
                             "embedding_dim":len(next(iter(embs.values()))) if embs else 0})
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error":str(e)},500)


# ═════════════════════════════════════════════════════════════════════════════
# POST /classify
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/classify")
async def classify():
    global _embeddings
    try:
        if _embeddings is None:
            r=await predict(); j=json.loads(r.body)
            if j.get("error"):
                return JSONResponse({"error":f"predict: {j['error']}"},500)
        if not _embeddings:
            return JSONResponse({"error":"No embeddings."},400)

        fnames=sorted(_embeddings.keys())
        X=np.array([_embeddings[f] for f in fnames],dtype=np.float32)
        src=TILES_RAW_DIR if os.path.isdir(TILES_RAW_DIR) else TILES_DIR

        vscores=[]
        for fname in fnames:
            path=os.path.join(src,fname)
            if os.path.exists(path):
                with open(path,"rb") as f: tile=load_image(f.read())
                mask=valid_mask(tile); vscores.append(veg_score(tile,mask))
            else: vscores.append(0.0)

        n_comp=min(32,X.shape[0]-1,X.shape[1])
        Xp,vr=pca(X,n_comp); Xs=scale(Xp)
        lbl,ctr,inn=kmeans(Xs,k=2)
        print(f"[classify] PCA {X.shape[1]}→{n_comp}D var={vr*100:.1f}% inertia={inn:.2f}")

        n_total=len(lbl)
        ratio_max=max((lbl==0).sum(),(lbl==1).sum())/n_total
        use_fallback=ratio_max>CLUSTER_BALANCE_THRESH
        print(f"[classify] ratio_max={ratio_max:.2f} fallback={use_fallback}")

        if use_fallback:
            results,h_cnt,d_cnt=label_by_veg_threshold(fnames,vscores)
        else:
            m0=float(np.mean([s for s,l in zip(vscores,lbl) if l==0 and s>-999])) if any(l==0 for l in lbl) else 0.0
            m1=float(np.mean([s for s,l in zip(vscores,lbl) if l==1 and s>-999])) if any(l==1 for l in lbl) else 0.0
            dis_cl=0 if m0<m1 else 1
            print(f"[classify] dis_cl={dis_cl} veg={min(m0,m1):.4f} | healthy_cl={1-dis_cl} veg={max(m0,m1):.4f}")
            confs=get_conf(Xs,lbl,ctr)
            results,h_cnt,d_cnt=[],0,0
            for fname,l,c in zip(fnames,lbl,confs):
                lab="diseased" if int(l)==dis_cl else "healthy"
                if lab=="healthy": h_cnt+=1
                else: d_cnt+=1
                results.append({"filename":fname,"label":lab,
                                 "cluster":int(l),"confidence":round(float(c),3)})

        skip_path=os.path.join(BASE_DIR,"bg_tiles.json")
        skip_set=set(json.load(open(skip_path))) if os.path.exists(skip_path) else set()
        for f in skip_set:
            results.append({"filename":f,"label":"background","cluster":-1,"confidence":1.0})

        total=h_cnt+d_cnt
        return JSONResponse({
            "status":"success","total_tiles":len(results),
            "healthy":h_cnt,"diseased":d_cnt,
            "healthy_pct":round(h_cnt/total*100,1) if total else 0.0,
            "diseased_pct":round(d_cnt/total*100,1) if total else 0.0,
            "pca_variance":round(vr*100,2),
            "method":"threshold_fallback" if use_fallback else "kmeans",
            "model_mode":"onnx" if MODEL_LOADED else "handcrafted",
            "results":results,
        })
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error":str(e)},500)


# ═════════════════════════════════════════════════════════════════════════════
# POST /run
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/run")
async def run():
    try:
        r=await predict(); j=json.loads(r.body)
        if j.get("error"): return JSONResponse({"error":f"predict: {j['error']}"},500)
        r2=await classify(); j2=json.loads(r2.body)
        if j2.get("error"): return JSONResponse({"error":f"classify: {j2['error']}"},500)
        return JSONResponse(j2)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error":str(e)},500)


# ═════════════════════════════════════════════════════════════════════════════
# GET /tiles/{filename}
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/tiles/{filename}")
async def get_tile(filename: str):
    if ".." in filename or "/" in filename:
        return JSONResponse({"error":"Invalid filename"},400)
    path=os.path.join(TILES_DIR,filename)
    if not os.path.exists(path): return JSONResponse({"error":"Not found"},404)
    return FileResponse(path,media_type="image/tiff")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
# Running as a script (python main.py) handles:
#   • Normal local run
#   • Google Colab + ngrok (when COLAB_MODE = True)
# For production use uvicorn directly:
#   uvicorn main:app --host 0.0.0.0 --port 8000
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    import socket

    def get_local_ip():
        """Get the machine's LAN IP address automatically."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    if COLAB_MODE:
        # ── Google Colab → ngrok public URL ──────────────────────────────
        try:
            import nest_asyncio
            from pyngrok import ngrok, conf

            nest_asyncio.apply()

            if NGROK_TOKEN:
                conf.get_default().auth_token = NGROK_TOKEN

            public_url = ngrok.connect(PORT)
            print("\n" + "=" * 60)
            print("  GOOGLE COLAB — PUBLIC URL")
            print("=" * 60)
            print(f"  URL     :  {public_url}")
            print(f"  Open the above URL in any browser, anywhere.")
            print(f"  NOTE    :  URL changes every time Colab restarts.")
            print("=" * 60 + "\n")

        except ImportError:
            print("ERROR: pyngrok or nest_asyncio not installed.")
            print("Run:  !pip install pyngrok nest_asyncio")
            raise

        uvicorn.run(app, host="0.0.0.0", port=PORT)

    else:
        # ── Local / LAN / Server run ──────────────────────────────────────
        lan_ip = get_local_ip()
        print("\n" + "=" * 60)
        print("  UAV DISEASE DETECTION — SERVER STARTED")
        print("=" * 60)
        print(f"  Step 1 — Open on THIS computer:")
        print(f"           http://localhost:{PORT}")
        print(f"           http://127.0.0.1:{PORT}")
        print()
        print(f"  Step 2 — Open from ANY PC in the same network:")
        print(f"           http://{lan_ip}:{PORT}")
        print()
        print(f"  Step 3 — If Step 2 fails, allow firewall once:")
        print(f"           sudo ufw allow {PORT}/tcp")
        print(f"           sudo ufw reload")
        print()
        print(f"  IMPORTANT: Do NOT open http://0.0.0.0:{PORT}")
        print(f"             0.0.0.0 is the bind address, not a URL.")
        print(f"             Use localhost or {lan_ip} instead.")
        print("=" * 60 + "\n")

        uvicorn.run(
            app,
            host="0.0.0.0",    # binds to all interfaces — do not visit this
            port=PORT,
            reload=False,
        )
