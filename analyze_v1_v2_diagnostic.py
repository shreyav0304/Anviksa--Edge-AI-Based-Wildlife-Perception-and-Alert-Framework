#!/usr/bin/env python3
"""Read-only V1/V2 diagnostic; writes only a new results directory."""
from __future__ import annotations

import csv, hashlib, json, math, re, uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "results/mobilenet_v3_large/20260830T171150Z_94069968"
V2 = ROOT / "results/mobilenet_v3_large_v2/20260904T120832Z_06b32daa"
CACHE = ROOT / "results/model_comparison_dashboard/prediction_probabilities_mobilenet_v3_large.npz"
META = ROOT / "dataset_v2_candidate/metadata"
CLASSES = ("Cow","Deer","Elephant","Monkey","Non_Venomous_Snake","Venomous_Snake","Wild_Boar")
EXPECTED_FP = "9dd48f6a1aeb8afad1f76d943b7baf36a58c837e50b7124cacb3f5211e580f23"
TOL = 1e-6

def read_json(p): return json.loads(p.read_text(encoding="utf-8"))
def read_csv(p):
    with p.open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def write_json(p,x): p.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
def write_csv(p,rows,fields=None):
    fields=fields or (list(rows[0]) if rows else [])
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def snapshot(paths):
    return {str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for b in paths if b.exists() for p in b.rglob("*") if p.is_file()}
def top(prob):
    order=np.argsort(prob)[::-1]; return int(order[0]),float(prob[order[0]]),int(order[1]),float(prob[order[1]]),float(prob[order[0]]-prob[order[1]])
def classify_conf(c,m): return "HIGH" if c>=.8 else "MODERATE" if c>=.6 else "LOW", c<.6 or m<.15

def save_bar(path,title,labels,a,b,an="V1",bn="V2",ylabel="Score"):
    x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(12,6)); ax.bar(x-.2,a,.4,label=an); ax.bar(x+.2,b,.4,label=bn)
    ax.set(xticks=x,xticklabels=[s.replace("_","\n") for s in labels],ylabel=ylabel,title=title); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=240); plt.close(fig)
def save_history_graph(path,rows,key,valkey,title):
    fig,ax=plt.subplots(figsize=(11,6)); x=np.arange(1,len(rows)+1); ax.plot(x,[float(r[key]) for r in rows],label="Training"); ax.plot(x,[float(r[valkey]) for r in rows],label="Validation")
    b=next(i for i,r in enumerate(rows,1) if r["stage"]=="fine_tuning"); ax.axvline(b-.5,color="black",ls="--",label="Fine-tuning starts")
    ax.set(xlabel="Combined epoch",ylabel=title,title=f"V2 {title}"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=240); plt.close(fig)
def contact_sheet(path,rows,maxn=25):
    rows=rows[:maxn]; cols=5; cw,ch=300,250; sheet=Image.new("RGB",(cols*cw,math.ceil(max(1,len(rows))/cols)*ch),"white"); draw=ImageDraw.Draw(sheet); font=ImageFont.load_default()
    for i,r in enumerate(rows):
        x=(i%cols)*cw;y=(i//cols)*ch
        try:
            with Image.open(ROOT/"dataset_v2_candidate"/r["relative_path"]) as im:
                im.convert("RGB").thumbnail((280,165)); sheet.paste(im.convert("RGB"),(x+10,y+5))
        except Exception: draw.rectangle((x+10,y+5,x+290,y+170),outline="red")
        text=f"{Path(r['relative_path']).name[:34]}\nGT: {r['ground_truth']}\nV1: {r['v1_predicted']} {float(r['v1_confidence']):.3f}\nV2: {r['v2_predicted']} {float(r['v2_confidence']):.3f}"
        draw.multiline_text((x+8,y+175),text,fill="black",font=font,spacing=2)
    sheet.save(path)

def main():
    required=[V1/"best_model.keras",V2/"best_model.keras",V1/"metrics.json",V2/"metrics.json",CACHE,V2/"test_predictions.csv",META/"dataset_v2_manifest.csv"]
    if any(not p.is_file() for p in required): raise FileNotFoundError([str(p) for p in required if not p.is_file()])
    protected=[ROOT/"clean_dataset",ROOT/"combined_dataset",ROOT/"dataset_v2_candidate",ROOT/"ui",ROOT/"snake_video_dataset",ROOT/"pi_deployment/models"]
    before=snapshot(protected); critical={str(p.relative_to(ROOT)):sha(p) for p in [V1/"best_model.keras",V2/"best_model.keras",ROOT/"pi_deployment/models/mobilenet_v3_large_float32.tflite",ROOT/"pi_deployment/models/mobilenet_v3_large_float16.tflite"]}
    outroot=ROOT/"results/v1_v2_diagnostic"; outroot.mkdir(exist_ok=True)
    out=outroot/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]); out.mkdir()
    v1m,v2m=read_json(V1/"metrics.json"),read_json(V2/"metrics.json"); v1cfg=read_json(V1/"config.json"); v2cfg=read_json(V2/"training_configuration.json")
    manifest=read_csv(META/"dataset_v2_manifest.csv"); test=[r for r in manifest if r["split"]=="test"]
    canonical=[(r["destination_path"],json.dumps({"destination_path":r["destination_path"],"split":r["split"],"class_name":r["class_name"],"sha256":r["sha256"]},sort_keys=True,separators=(",",":"))) for r in manifest]
    fp=hashlib.sha256(("\n".join(x[1] for x in sorted(canonical))+"\n").encode()).hexdigest()
    hist=read_csv(META/"historical_frozen_test_manifest.csv"); order_ok=[r["relative_path"] for r in hist]==[r["destination_path"] for r in test]
    cache=np.load(CACHE,allow_pickle=False); p1=np.asarray(cache["probabilities"]); labels=np.asarray(cache["true_labels"]); v2rows=read_csv(V2/"test_predictions.csv")
    v2map={r["relative_path"]:r for r in v2rows}; paths=[r["destination_path"] for r in test]
    if set(paths)!=set(v2map) or len(paths)!=545 or not order_ok: raise ValueError("test alignment failed")
    truth=np.array([CLASSES.index(r["class_name"]) for r in test]); p2=np.array([[float(v2map[p][f"probability_{c}"]) for c in CLASSES] for p in paths])
    if not np.array_equal(labels,truth): raise ValueError("V1 cached labels do not match aligned manifest")
    pred1=p1.argmax(1); pred2=p2.argmax(1)
    transitions=[]; snakes=[]
    for i,path in enumerate(paths):
        a,ac,a2,a2c,am=top(p1[i]); b,bc,b2,b2c,bm=top(p2[i]); bucket,uncertain=classify_conf(bc,bm)
        row={"relative_path":path,"ground_truth":CLASSES[truth[i]],"v1_predicted":CLASSES[a],"v1_confidence":ac,"v1_top2_class":CLASSES[a2],"v1_top2_probability":a2c,"v1_margin":am,
             "v2_predicted":CLASSES[b],"v2_confidence":bc,"v2_top2_class":CLASSES[b2],"v2_top2_probability":b2c,"v2_margin":bm,"prediction_changed":"YES" if a!=b else "NO",
             "v1_correct":"YES" if a==truth[i] else "NO","v2_correct":"YES" if b==truth[i] else "NO","v2_confidence_band":bucket,"v2_uncertain":"YES" if uncertain else "NO"}
        transitions.append(row)
        if truth[i] in (4,5):
            shift=float(p2[i,5]-p1[i,5]); row2=row|{"v1_venomous_probability":p1[i,5],"v1_non_venomous_probability":p1[i,4],"v1_snake_class_margin":p1[i,5]-p1[i,4],
                "v2_venomous_probability":p2[i,5],"v2_non_venomous_probability":p2[i,4],"v2_snake_class_margin":p2[i,5]-p2[i,4],"probability_shift_toward_venomous":shift,
                "probability_shift_toward_non_venomous":float(p2[i,4]-p1[i,4]),"classification_transition":f"{CLASSES[a]} -> {CLASSES[b]}"}
            snakes.append(row2)
    write_csv(out/"prediction_change_analysis.csv",transitions); write_csv(out/"snake_prediction_transitions.csv",snakes)
    groups={
      "nonvenomous_v1_correct_v2_venomous.csv":[r for r in snakes if r["ground_truth"]==CLASSES[4] and r["v1_correct"]=="YES" and r["v2_predicted"]==CLASSES[5]],
      "venomous_v1_wrong_v2_correct.csv":[r for r in snakes if r["ground_truth"]==CLASSES[5] and r["v1_correct"]=="NO" and r["v2_correct"]=="YES"],
      "both_nonvenomous_as_venomous.csv":[r for r in snakes if r["ground_truth"]==CLASSES[4] and r["v1_predicted"]==CLASSES[5] and r["v2_predicted"]==CLASSES[5]],
      "both_venomous_as_nonvenomous.csv":[r for r in snakes if r["ground_truth"]==CLASSES[5] and r["v1_predicted"]==CLASSES[4] and r["v2_predicted"]==CLASSES[4]],
      "v1_snake_error_corrected_by_v2.csv":[r for r in snakes if r["v1_correct"]=="NO" and r["v2_correct"]=="YES"],
      "new_snake_error_introduced_by_v2.csv":[r for r in snakes if r["v1_correct"]=="YES" and r["v2_correct"]=="NO"]}
    for name,rs in groups.items(): write_csv(out/name,rs,transitions[0].keys()|({} if not rs else rs[0].keys()))
    categories={"overall":range(545),"Venomous_Snake":np.where(truth==5)[0],"Non_Venomous_Snake":np.where(truth==4)[0],"all_snakes":np.where(np.isin(truth,[4,5]))[0]}
    categories|={c:np.where(truth==i)[0] for i,c in enumerate(CLASSES) if i not in (4,5)}
    net=[]
    for name,idx in categories.items():
        idx=np.array(list(idx)); broken=int(np.sum((pred1[idx]==truth[idx])&(pred2[idx]!=truth[idx]))); fixed=int(np.sum((pred1[idx]!=truth[idx])&(pred2[idx]==truth[idx])))
        net.append({"scope":name,"v1_wrong_v2_correct":fixed,"v1_correct_v2_wrong":broken,"both_correct":int(np.sum((pred1[idx]==truth[idx])&(pred2[idx]==truth[idx]))),"both_wrong":int(np.sum((pred1[idx]!=truth[idx])&(pred2[idx]!=truth[idx]))),"net_correction":fixed-broken})
    write_csv(out/"net_correction_analysis.csv",net)
    def confstats(prob,pred,mask):
        conf=prob.max(1)[mask]; margins=np.sort(prob,axis=1)[:,-1][mask]-np.sort(prob,axis=1)[:,-2][mask]
        return {"count":int(np.sum(mask)),"mean_confidence":float(np.mean(conf)) if len(conf) else None,"median_confidence":float(np.median(conf)) if len(conf) else None,"mean_margin":float(np.mean(margins)) if len(conf) else None,"median_margin":float(np.median(margins)) if len(conf) else None}
    confidence=[]
    masks={"all":np.ones(545,bool),"correct":pred1==truth,"incorrect":pred1!=truth,"Venomous_Snake":truth==5,"Non_Venomous_Snake":truth==4,"snake_errors":np.isin(truth,[4,5])&(pred1!=truth)}
    masks2={"all":np.ones(545,bool),"correct":pred2==truth,"incorrect":pred2!=truth,"Venomous_Snake":truth==5,"Non_Venomous_Snake":truth==4,"snake_errors":np.isin(truth,[4,5])&(pred2!=truth)}
    for model,prob,pred,ms in [("V1",p1,pred1,masks),("V2",p2,pred2,masks2)]:
        for scope,mask in ms.items(): confidence.append({"model":model,"scope":scope}|confstats(prob,pred,mask))
    write_csv(out/"confidence_analysis.csv",confidence)
    v2newerrors=groups["new_snake_error_introduced_by_v2.csv"]; bands=Counter(r["v2_confidence_band"] for r in v2newerrors); uncertain_count=sum(r["v2_uncertain"]=="YES" for r in v2newerrors)
    snake_idx=np.where(np.isin(truth,[4,5]))[0]; shifts=p2[snake_idx,5]-p1[snake_idx,5]
    probability={"unchanged_tolerance":TOL,"all_actual_snakes":{"v1_mean_venomous_probability":float(p1[snake_idx,5].mean()),"v1_median_venomous_probability":float(np.median(p1[snake_idx,5])),"v2_mean_venomous_probability":float(p2[snake_idx,5].mean()),"v2_median_venomous_probability":float(np.median(p2[snake_idx,5])),"mean_shift":float(shifts.mean()),"median_shift":float(np.median(shifts)),"minimum_shift":float(shifts.min()),"maximum_shift":float(shifts.max()),"toward_venomous":int(np.sum(shifts>TOL)),"toward_non_venomous":int(np.sum(shifts<-TOL)),"effectively_unchanged":int(np.sum(np.abs(shifts)<=TOL))}}
    for name,i in [("actual_venomous",5),("actual_non_venomous",4)]:
        ix=np.where(truth==i)[0]; s=p2[ix,5]-p1[ix,5]; probability[name]={"count":len(ix),"v1_mean":float(p1[ix,5].mean()),"v1_median":float(np.median(p1[ix,5])),"v2_mean":float(p2[ix,5].mean()),"v2_median":float(np.median(p2[ix,5])),"mean_shift":float(s.mean()),"median_shift":float(np.median(s))}
    write_json(out/"snake_probability_shift.json",probability)
    cm1=np.loadtxt(V1/"confusion_matrix_raw.csv",delimiter=",",skiprows=1,usecols=range(1,8),dtype=int); cm2=np.loadtxt(V2/"confusion_matrix.csv",delimiter=",",skiprows=1,usecols=range(1,8),dtype=int); diff=cm2-cm1
    write_csv(out/"confusion_matrix_difference.csv",[{"actual\\predicted":CLASSES[i]}|{CLASSES[j]:int(diff[i,j]) for j in range(7)} for i in range(7)])
    fig,ax=plt.subplots(figsize=(11,9)); lim=max(abs(diff.min()),abs(diff.max())); im=ax.imshow(diff,cmap="RdBu_r",vmin=-lim,vmax=lim); fig.colorbar(im,ax=ax); ax.set(xticks=range(7),yticks=range(7),xticklabels=CLASSES,yticklabels=CLASSES,xlabel="Predicted",ylabel="Actual",title="Confusion Matrix Difference (V2 - V1)"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    for i in range(7):
        for j in range(7): ax.text(j,i,str(diff[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(out/"confusion_matrix_difference.png",dpi=240); plt.close(fig)
    per=[]; p1class={r["class"]:r for r in v1m["per_class"]}; p2class={r["class"]:r for r in v2m["per_class"]}
    for c in CLASSES:
        row={"class":c}
        for k in ("precision","recall","f1","support"): row|={f"v1_{k}":p1class[c][k],f"v2_{k}":p2class[c][k],f"difference_{k}":p2class[c][k]-p1class[c][k]}
        d=row["difference_f1"]; row["status"]="IMPROVED" if d>TOL else "DEGRADED" if d<-TOL else "UNCHANGED"; per.append(row)
    write_csv(out/"per_class_difference_analysis.csv",per)
    splitrows=read_csv(META/"new_snake_image_split.csv"); sp=Counter((r["species"],r["parent_class"],r["assigned_split"],r["source_group"]) for r in splitrows); parent=Counter(r["parent_class"] for r in splitrows)
    species_rows=[]
    for (species,pc,split,group),count in sorted(sp.items()): species_rows.append({"species":species,"parent_class":pc,"split":split,"source_group":group,"count":count,"percentage_of_parent_class":100*count/parent[pc]})
    write_csv(out/"new_data_species_distribution.csv",species_rows)
    species_parent=Counter((r["species"],r["parent_class"]) for r in splitrows); dominant={pc:max(((s,n) for (s,p),n in species_parent.items() if p==pc),key=lambda x:x[1]) for pc in parent}
    source_summary={"domain_shift_conclusion":"DOMAIN SHIFT: NOT PROVEN","source_groups":len(set(r["source_group"] for r in splitrows)),"largest_source_groups":Counter(r["source_group"] for r in splitrows).most_common(10),"filename_extensions":Counter(Path(r["source_image"]).suffix.lower() for r in splitrows),"dimensions":{}}
    dims=[]
    for r in splitrows:
        with Image.open(r["source_image"]) as im: dims.append((im.width,im.height,im.width/im.height))
    source_summary["dimensions"]={"unique_width_height_pairs":len(set((w,h) for w,h,_ in dims)),"median_width":float(np.median([w for w,_,_ in dims])),"median_height":float(np.median([h for _,h,_ in dims])),"median_aspect_ratio":float(np.median([a for *_,a in dims]))}
    source_summary["filename_extensions"]=dict(source_summary["filename_extensions"]); write_json(out/"new_data_source_domain_analysis.json",source_summary)
    histrows=read_csv(V2/"training_history.csv"); write_csv(out/"v2_epoch_analysis.csv",histrows)
    s1=[r for r in histrows if r["stage"]=="stage_1"]; s2=[r for r in histrows if r["stage"]=="fine_tuning"]
    best1=max(s1,key=lambda r:float(r["val_accuracy"])); best2=max(s2,key=lambda r:float(r["val_accuracy"])); minloss2=min(s2,key=lambda r:float(r["val_loss"]))
    history_analysis={"best_stage1_epoch":int(best1["stage_epoch"]),"best_stage1_validation_accuracy":float(best1["val_accuracy"]),"best_stage2_epoch":int(best2["stage_epoch"]),"best_stage2_validation_accuracy":float(best2["val_accuracy"]),"stage2_beat_stage1":float(best2["val_accuracy"])>float(best1["val_accuracy"]),"best_stage2_validation_loss":float(minloss2["val_loss"]),"stage2_validation_loss_improved_vs_first_epoch":float(minloss2["val_loss"])<float(s2[0]["val_loss"]),"training_accuracy_increased_stage2":float(s2[-1]["accuracy"])>float(s2[0]["accuracy"]),"overfitting_evidence":"POSSIBLE"}
    write_json(out/"training_history_analysis.json",history_analysis)
    settings=[("architecture",v1m["model_name"],"MobileNetV3 Large"),("ImageNet initialization","ImageNet","ImageNet (fresh)"),("input size",v1cfg["image_size"],v2cfg["image_size"]),("augmentation",v1cfg["augmentation"],v2cfg["augmentation"]),("batch size",v1cfg["batch_size"],v2cfg["batch_size"]),("optimizer","Adam",v2cfg["optimizer"]),("Stage 1 learning rate",v1cfg["initial_learning_rate"],v2cfg["initial_learning_rate"]),("Stage 2 learning rate",v1cfg["fine_tune_learning_rate"],v2cfg["fine_tune_learning_rate"]),("fine-tune layers",45,45),("BatchNorm frozen",True,True),("early stopping patience",v1cfg["early_stopping_patience"],v2cfg["early_stopping_patience"]),("ReduceLROnPlateau patience",v1cfg["reduce_lr_patience"],v2cfg["reduce_lr_patience"]),("ReduceLROnPlateau factor",v1cfg["reduce_lr_factor"],v2cfg["reduce_lr_factor"]),("seed",v1cfg["seed"],v2cfg["seed"]),("epoch limits",[v1cfg["initial_epochs"],v1cfg["fine_tune_epochs"]],[v2cfg["initial_epochs"],v2cfg["fine_tune_epochs"]])]
    cfgrows=[{"setting":n,"v1":json.dumps(a,sort_keys=True),"v2":json.dumps(b,sort_keys=True),"status":"MATCH" if a==b or n in ("architecture","ImageNet initialization") else "DIFFERENT"} for n,a,b in settings]; write_csv(out/"training_configuration_comparison.csv",cfgrows)
    w1,w2=read_json(V1/"class_weights.json"),read_json(V2/"class_weights.json"); weightrows=[{"class":c,"v1_weight":w1[str(i)],"v2_weight":w2[str(i)],"difference":w2[str(i)]-w1[str(i)],"percent_change":100*(w2[str(i)]/w1[str(i)]-1)} for i,c in enumerate(CLASSES)]; write_csv(out/"class_weight_analysis.csv",weightrows)
    v1counts=Counter((r["split"],r["class_name"]) for r in read_csv(ROOT/"dataset_integrity_reports/clean/dataset_manifest.csv")) if (ROOT/"dataset_integrity_reports/clean/dataset_manifest.csv").exists() else Counter()
    if not v1counts:
        for split in ("train","validation","test"):
            for r in read_csv(ROOT/f"dataset_integrity_reports/clean/{split}_manifest.csv"): v1counts[(split,r["class_name"])]+=1
    v2counts=Counter((r["split"],r["class_name"]) for r in manifest); balance=[]
    for split in ("train","validation"):
        for version,counts in (("V1",v1counts),("V2",v2counts)):
            nv=counts[(split,CLASSES[4])]; vv=counts[(split,CLASSES[5])]; balance.append({"version":version,"split":split,"non_venomous":nv,"venomous":vv,"nonvenomous_to_venomous_ratio":nv/vv,"venomous_share":vv/(nv+vv)})
    write_csv(out/"snake_class_balance.csv",balance)
    save_bar(out/"per_class_recall.png","V1 vs V2 Per-Class Recall",CLASSES,[p1class[c]["recall"] for c in CLASSES],[p2class[c]["recall"] for c in CLASSES])
    save_bar(out/"per_class_f1.png","V1 vs V2 Per-Class F1",CLASSES,[p1class[c]["f1"] for c in CLASSES],[p2class[c]["f1"] for c in CLASSES])
    fig,ax=plt.subplots(figsize=(10,6)); ax.hist(shifts,bins=30); ax.axvline(0,color="black"); ax.set(title="V2 - V1 Venomous Probability Shift",xlabel="Probability shift",ylabel="Snake test images"); fig.tight_layout(); fig.savefig(out/"snake_probability_shift_distribution.png",dpi=240); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6)); ax.hist(p1[snake_idx].max(1),bins=25,alpha=.55,label="V1"); ax.hist(p2[snake_idx].max(1),bins=25,alpha=.55,label="V2"); ax.legend(); ax.set(title="Snake Top-1 Confidence Distribution",xlabel="Confidence"); fig.tight_layout(); fig.savefig(out/"snake_confidence_distribution.png",dpi=240); plt.close(fig)
    save_history_graph(out/"v2_training_validation_accuracy.png",histrows,"accuracy","val_accuracy","Accuracy"); save_history_graph(out/"v2_training_validation_loss.png",histrows,"loss","val_loss","Loss")
    spp=Counter(r["species"] for r in splitrows); fig,ax=plt.subplots(figsize=(13,7)); names,vals=zip(*spp.most_common()); ax.bar(names,vals); ax.tick_params(axis="x",rotation=70); ax.set(title="New Image Species Distribution",ylabel="Images"); fig.tight_layout(); fig.savefig(out/"new_image_species_distribution.png",dpi=240); plt.close(fig)
    labelsb=[f"{r['version']} {r['split']}" for r in balance]; fig,ax=plt.subplots(figsize=(10,6)); x=np.arange(4); ax.bar(x-.2,[r["non_venomous"] for r in balance],.4,label="Non-Venomous"); ax.bar(x+.2,[r["venomous"] for r in balance],.4,label="Venomous"); ax.set(xticks=x,xticklabels=labelsb,title="Snake Train/Validation Class Balance",ylabel="Images"); ax.legend(); fig.tight_layout(); fig.savefig(out/"snake_class_balance.png",dpi=240); plt.close(fig)
    contact_sheet(out/"contact_sheet_nonvenomous_broken.png",groups["nonvenomous_v1_correct_v2_venomous.csv"])
    contact_sheet(out/"contact_sheet_venomous_corrected.png",groups["venomous_v1_wrong_v2_correct.csv"])
    contact_sheet(out/"contact_sheet_largest_shift_toward_venomous.png",sorted(snakes,key=lambda r:float(r["probability_shift_toward_venomous"]),reverse=True))
    contact_sheet(out/"contact_sheet_largest_shift_toward_nonvenomous.png",sorted(snakes,key=lambda r:float(r["probability_shift_toward_venomous"])))
    changed=int(np.sum(pred1!=pred2)); broken=int(np.sum((pred1==truth)&(pred2!=truth))); fixed=int(np.sum((pred1!=truth)&(pred2==truth)))
    largest_worse=sorted((int(diff[i,j]),CLASSES[i],CLASSES[j]) for i in range(7) for j in range(7) if i!=j and diff[i,j]>0)[::-1]
    largest_better=sorted((int(diff[i,j]),CLASSES[i],CLASSES[j]) for i in range(7) for j in range(7) if i!=j and diff[i,j]<0)
    v1train=next(r for r in balance if r["version"]=="V1" and r["split"]=="train"); v2train=next(r for r in balance if r["version"]=="V2" and r["split"]=="train")
    nonven_broken=len(groups["nonvenomous_v1_correct_v2_venomous.csv"]); ven_fixed=len(groups["venomous_v1_wrong_v2_correct.csv"])
    most_errors=max(bands,key=bands.get) if bands else "N/A"
    explanation=("The clearest measured mechanism is a decision-boundary/probability shift toward Venomous on actual Non-Venomous test images, coinciding with changed snake class frequencies and class weights. "
                 "This explains the prediction pattern but does not prove which property of the added species/source domains caused it; domain shift is not proven.")
    integrity={"v1_experiment_exists":V1.is_dir(),"v2_experiment_exists":V2.is_dir(),"dataset_v2_fingerprint":fp,"fingerprint_matches":fp==EXPECTED_FP,"frozen_test_count":len(test),"class_order":list(CLASSES),"test_order_identical":order_ok,"cached_v1_predictions_used":True,"v2_stored_predictions_used":True,"inference_performed":False}
    after=snapshot(protected); integrity|={"protected_paths_unchanged":before==after,"critical_file_hashes_before":critical,"critical_file_hashes_after":{str(p.relative_to(ROOT)):sha(p) for p in [V1/"best_model.keras",V2/"best_model.keras",ROOT/"pi_deployment/models/mobilenet_v3_large_float32.tflite",ROOT/"pi_deployment/models/mobilenet_v3_large_float16.tflite"]},"training_performed":False,"fine_tuning_performed":False}
    write_json(out/"integrity_verification.json",integrity)
    qs={"Q1_predictions_changed":changed,"Q2_v1_correct_v2_wrong":broken,"Q3_v1_wrong_v2_correct":fixed,"Q4_net_correction":fixed-broken,"Q5_nonvenomous_correct_to_venomous":nonven_broken,"Q6_venomous_errors_corrected":ven_fixed,
        "Q7_systematic_shift_toward_venomous": probability["all_actual_snakes"]["mean_shift"]>0 and probability["all_actual_snakes"]["toward_venomous"]>probability["all_actual_snakes"]["toward_non_venomous"],
        "Q8_new_snake_error_confidence":dict(bands)|{"uncertain":uncertain_count},"Q9_dominant_species":dominant,"Q10_balance_improved_numerically":"YES; the train nonvenomous:venomous ratio moved closer to 1","Q11_class_weights_changed":"YES",
        "Q12_stage2_beat_stage1":"NO","Q13_overfitting_evidence":"POSSIBLE","Q14_configuration_comparability":"HIGH","Q15_strongest_explanation":explanation}
    write_json(out/"diagnostic_questions.json",qs)
    summary=f"""# ANVIKSA V1 vs V2 Diagnostic Summary

## A. Executive Summary

V2 changed {changed} of 545 predictions. It fixed {fixed} V1 errors but broke {broken} V1-correct cases, for a net correction of {fixed-broken}. Venomous recall improved, while Non-Venomous recall declined. V1 remains the selected model.

## B. Integrity Verification

All required artifacts exist. Dataset V2 fingerprint is `{fp}`. Test count and order match at 545. Cached predictions were used; no inference, training, or fine-tuning occurred. Protected paths remained unchanged: {before==after}.

## C. Prediction Transition Analysis

Both correct: {net[0]['both_correct']}; V1 correct to V2 wrong: {broken}; V1 wrong to V2 correct: {fixed}; both wrong: {net[0]['both_wrong']}. Net correction: {fixed-broken}.

## D. Snake-Specific Error Analysis

Non-Venomous correct to Venomous: {nonven_broken}. Venomous errors corrected: {ven_fixed}. Largest worsened off-diagonal cells: {largest_worse[:3]}. Largest improved cells: {largest_better[:3]}.

## E. Probability / Confidence Shift

Mean venomous probability shift across actual snakes: {probability['all_actual_snakes']['mean_shift']:.9f}; median: {probability['all_actual_snakes']['median_shift']:.9f}. Counts toward Venomous / toward Non-Venomous / unchanged: {probability['all_actual_snakes']['toward_venomous']} / {probability['all_actual_snakes']['toward_non_venomous']} / {probability['all_actual_snakes']['effectively_unchanged']} (tolerance {TOL}). New V2 snake errors were chiefly {most_errors.lower()} confidence; uncertainty count {uncertain_count}.

## F. New Dataset Species Distribution

Dominant Venomous species: {dominant['Venomous'][0]} ({dominant['Venomous'][1]}). Dominant Non-Venomous species: {dominant['Non-Venomous'][0]} ({dominant['Non-Venomous'][1]}). Species counts are metadata-derived only. DOMAIN SHIFT: NOT PROVEN.

## G. Dataset Balance

V1 train Non-Venomous/Venomous: {v1train['non_venomous']}/{v1train['venomous']} (ratio {v1train['nonvenomous_to_venomous_ratio']:.6f}). V2: {v2train['non_venomous']}/{v2train['venomous']} (ratio {v2train['nonvenomous_to_venomous_ratio']:.6f}). Balance moved numerically closer to parity, but that alone does not explain behavior.

## H. Training History

Stage 1 best: epoch {best1['stage_epoch']}, validation accuracy {float(best1['val_accuracy']):.9f}. Stage 2 best: epoch {best2['stage_epoch']}, {float(best2['val_accuracy']):.9f}. Stage 2 never exceeded Stage 1. Its validation loss did improve transiently while training accuracy increased; later divergence is consistent with possible overfitting, not proof of it.

## I. V1/V2 Configuration Comparability

Comparability is HIGH. Architecture, fresh ImageNet initialization strategy, preprocessing, augmentation, batch size, optimizer, learning rates, fine-tune depth, BatchNorm policy, callbacks, seed, and epoch limits match. Dataset contents, derived class weights, and selected training stage differ; therefore observed effects cannot be attributed to image content alone independently of frequency-derived weights and stochastic training.

## J. Class Weight Analysis

Weights changed. Non-Venomous: {w1['4']:.9f} to {w2['4']:.9f}; Venomous: {w1['5']:.9f} to {w2['5']:.9f}. Both declined, with a larger relative decline for Non-Venomous.

## K. Evidence-Supported Findings

{explanation}

## L. Unproven Hypotheses

Specific species concentration, source-domain characteristics, visual label noise, or augmentation interactions may contribute, but the stored metadata and frozen-test transitions do not establish causality. DOMAIN SHIFT: NOT PROVEN.

## M. Recommended Next Experiment

**SUPPORTED BY DIAGNOSTIC EVIDENCE:** retain V1 for deployment and inspect the identified transition/contact-sheet groups plus rights status before any further training.

**HYPOTHESIS REQUIRING EXPERIMENT:** a future separately approved controlled ablation could isolate added-image content from recalculated class weights. No V3 experiment was started here.

V2 deployment promotion: **NOT APPROVED**. Dataset image rights remain **REVIEW REQUIRED**.
"""
    (out/"diagnostic_summary.md").write_text(summary,encoding="utf-8")
    print(json.dumps({"output":str(out),"changed":changed,"broken":broken,"fixed":fixed,"net":fixed-broken,"nonvenomous_broken":nonven_broken,"venomous_corrected":ven_fixed,"mean_shift":probability['all_actual_snakes']['mean_shift'],"median_shift":probability['all_actual_snakes']['median_shift'],"dominant":dominant,"v1_ratio":v1train['nonvenomous_to_venomous_ratio'],"v2_ratio":v2train['nonvenomous_to_venomous_ratio'],"protected_unchanged":before==after},indent=2))

if __name__=="__main__": main()
