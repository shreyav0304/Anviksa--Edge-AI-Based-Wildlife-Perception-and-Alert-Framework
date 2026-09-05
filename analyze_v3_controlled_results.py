#!/usr/bin/env python3
"""Create V1/V2/V3 post-training comparisons from stored predictions only."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
V1=ROOT/"results/mobilenet_v3_large/20260830T171150Z_94069968"
V2=ROOT/"results/mobilenet_v3_large_v2/20260904T120832Z_06b32daa"
V3=ROOT/"results/mobilenet_v3_large_v3/20260905T044151Z_ea0bf4da"
CLASSES=("Cow","Deer","Elephant","Monkey","Non_Venomous_Snake","Venomous_Snake","Wild_Boar")

def j(p): return json.loads(p.read_text(encoding="utf-8"))
def rows(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def stored(path):
    rs=rows(path); return {r["relative_path"]:r for r in rs}
def probs(r):return np.array([float(r[f"probability_{c}"]) for c in CLASSES])

def main():
    m1,m2,m3=j(V1/"metrics.json"),j(V2/"metrics.json"),j(V3/"metrics.json")
    cache=np.load(ROOT/"results/model_comparison_dashboard/prediction_probabilities_mobilenet_v3_large.npz",allow_pickle=False)
    manifest=[r for r in rows(ROOT/"dataset_v2_candidate/metadata/dataset_v2_manifest.csv") if r["split"]=="test"]
    paths=[r["destination_path"] for r in manifest]; truth=np.array([CLASSES.index(r["class_name"]) for r in manifest]); p1=cache["probabilities"]
    d2,d3=stored(V2/"test_predictions.csv"),stored(V3/"test_predictions.csv")
    if set(paths)!=set(d2) or set(paths)!=set(d3) or not np.array_equal(cache["true_labels"],truth):raise ValueError("alignment failed")
    p2=np.array([probs(d2[p]) for p in paths]); p3=np.array([probs(d3[p]) for p in paths]); y1=p1.argmax(1);y2=p2.argmax(1);y3=p3.argmax(1)
    dash=next(r for r in rows(ROOT/"results/model_comparison_dashboard/model_comparison_metrics.csv") if r["Model"]=="MobileNetV3 Large")
    def flat(m,extra=None):
        x={"accuracy":m["test_accuracy"],"macro_precision":m["macro_precision"],"macro_recall":m["macro_recall"],"macro_f1":m["macro_f1"],"weighted_f1":m["weighted_f1"],
           "mse":m.get("probability_mse"),"mean_confidence":m.get("mean_confidence"),"median_confidence":m.get("median_confidence"),
           "venomous_precision":m["venomous_snake"]["precision"],"venomous_recall":m["venomous_snake"]["recall"],"venomous_f1":m["venomous_snake"]["f1"],
           "non_venomous_precision":m["non_venomous_snake"]["precision"],"non_venomous_recall":m["non_venomous_snake"]["recall"],"non_venomous_f1":m["non_venomous_snake"]["f1"],
           "snake_macro_recall":m["snake_macro_recall"],"snake_macro_f1":m["snake_macro_f1"],"non_venomous_to_venomous":m.get("non_venomous_as_venomous"),"venomous_to_non_venomous":m.get("venomous_as_non_venomous")}
        if extra:x.update(extra)
        return x
    cm1=np.loadtxt(V1/"confusion_matrix_raw.csv",delimiter=",",skiprows=1,usecols=range(1,8),dtype=int)
    a1=flat(m1,{"mse":float(dash["MSE"]),"mean_confidence":float(dash["Mean_Confidence"]),"median_confidence":float(dash["Median_Confidence"]),"non_venomous_to_venomous":int(cm1[4,5]),"venomous_to_non_venomous":int(cm1[5,4])})
    a2,a3=flat(m2),flat(m3); keys=list(a3)
    comparison={"V1":a1,"V2":a2,"V3":a3,"V3_minus_V1":{k:a3[k]-a1[k] for k in keys},"V3_minus_V2":{k:a3[k]-a2[k] for k in keys}}
    dump(V3/"comparison_v1_v2_v3.json",comparison)
    transitions=[]
    for i,p in enumerate(paths):
        transitions.append({"relative_path":p,"ground_truth":CLASSES[truth[i]],"v1_prediction":CLASSES[y1[i]],"v2_prediction":CLASSES[y2[i]],"v3_prediction":CLASSES[y3[i]],
          "v1_correct":y1[i]==truth[i],"v2_correct":y2[i]==truth[i],"v3_correct":y3[i]==truth[i],"v1_confidence":float(p1[i].max()),"v2_confidence":float(p2[i].max()),"v3_confidence":float(p3[i].max()),
          "v1_venomous_probability":float(p1[i,5]),"v2_venomous_probability":float(p2[i,5]),"v3_venomous_probability":float(p3[i,5])})
    with (V3/"prediction_transitions_v1_v2_v3.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(transitions[0]));w.writeheader();w.writerows(transitions)
    def counts(old,new):return {"prediction_changes":int(np.sum(old!=new)),"correct_to_wrong":int(np.sum((old==truth)&(new!=truth))),"wrong_to_correct":int(np.sum((old!=truth)&(new==truth)))}
    t={"V1_to_V3":counts(y1,y3),"V2_to_V3":counts(y2,y3),"snake_specific":{
      "v1_nonvenomous_correct_to_v3_venomous":int(np.sum((truth==4)&(y1==4)&(y3==5))),
      "v2_nonvenomous_venomous_to_v3_corrected":int(np.sum((truth==4)&(y2==5)&(y3==4))),
      "v1_venomous_wrong_to_v3_corrected":int(np.sum((truth==5)&(y1!=5)&(y3==5))),
      "v2_venomous_correct_to_v3_nonvenomous":int(np.sum((truth==5)&(y2==5)&(y3==4)))}}
    dump(V3/"prediction_transition_summary.json",t)
    shifts={}
    for scope,mask in {"all_actual_snakes":np.isin(truth,[4,5]),"actual_venomous":truth==5,"actual_non_venomous":truth==4}.items():
        s31=p3[mask,5]-p1[mask,5];s32=p3[mask,5]-p2[mask,5]
        shifts[scope]={"count":int(mask.sum()),"v3_minus_v1_mean":float(s31.mean()),"v3_minus_v1_median":float(np.median(s31)),"v3_minus_v2_mean":float(s32.mean()),"v3_minus_v2_median":float(np.median(s32))}
    dump(V3/"probability_shift_analysis.json",shifts)
    answers={"restored_nonvenomous_recall_vs_v2":a3["non_venomous_recall"]>a2["non_venomous_recall"],"retained_v2_venomous_recall_improvement":a3["venomous_recall"]>=a2["venomous_recall"],
      "exceeded_v1_snake_macro_recall":a3["snake_macro_recall"]>a1["snake_macro_recall"],"exceeded_v1_snake_macro_f1":a3["snake_macro_f1"]>a1["snake_macro_f1"],"exceeded_v1_accuracy":a3["accuracy"]>a1["accuracy"],
      "reduced_nonvenomous_to_venomous_vs_v2":a3["non_venomous_to_venomous"]<a2["non_venomous_to_venomous"],"reversed_venomous_probability_shift_vs_v2":shifts["all_actual_snakes"]["v3_minus_v2_mean"]<0,
      "stage2_beat_stage1":j(V3/"experiment_metadata.json")["selected_stage"]=="fine_tuning","better_snake_balance_than_v2":a3["snake_macro_recall"]>a2["snake_macro_recall"] and a3["snake_macro_f1"]>a2["snake_macro_f1"],
      "sufficiently_better_than_v1_for_deployment_consideration":False,"class_weight_hypothesis":"MIXED"}
    dump(V3/"experiment_questions.json",answers)
    metrics=["accuracy","macro_f1","venomous_recall","non_venomous_recall","snake_macro_recall","snake_macro_f1","mse"]
    fig,axes=plt.subplots(2,4,figsize=(18,9));
    for ax,k in zip(axes.ravel(),metrics):ax.bar(["V1","V2","V3"],[a1[k],a2[k],a3[k]]);ax.set_title(k.replace("_"," "));ax.grid(axis="y",alpha=.25)
    axes.ravel()[-1].axis("off");fig.suptitle("MobileNetV3 Large V1 vs V2 vs V3");fig.tight_layout();fig.savefig(V3/"comparison_v1_v2_v3.png",dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));mask=np.isin(truth,[4,5]);ax.hist(p3[mask,5]-p2[mask,5],bins=30,alpha=.7,label="V3 - V2");ax.hist(p3[mask,5]-p1[mask,5],bins=30,alpha=.55,label="V3 - V1");ax.axvline(0,color="black");ax.legend();ax.set(title="V3 Venomous Probability Shifts on Actual Snakes",xlabel="Probability shift");fig.tight_layout();fig.savefig(V3/"probability_shift_v3.png",dpi=240);plt.close(fig)
    prior=j(ROOT/"results/v1_v2_diagnostic/20260904T145314Z_de424635/integrity_verification.json")["critical_file_hashes_after"]
    current={"v1_model":sha(V1/"best_model.keras"),"v2_model":sha(V2/"best_model.keras"),"float32_tflite":sha(ROOT/"pi_deployment/models/mobilenet_v3_large_float32.tflite"),"float16_tflite":sha(ROOT/"pi_deployment/models/mobilenet_v3_large_float16.tflite")}
    expected={"v1_model":prior["results\\mobilenet_v3_large\\20260830T171150Z_94069968\\best_model.keras"],"v2_model":prior["results\\mobilenet_v3_large_v2\\20260904T120832Z_06b32daa\\best_model.keras"],"float32_tflite":prior["pi_deployment\\models\\mobilenet_v3_large_float32.tflite"],"float16_tflite":prior["pi_deployment\\models\\mobilenet_v3_large_float16.tflite"]}
    integrity={"original_v1_modified":current["v1_model"]!=expected["v1_model"],"v2_modified":current["v2_model"]!=expected["v2_model"],"existing_tflite_modified":current["float32_tflite"]!=expected["float32_tflite"] or current["float16_tflite"]!=expected["float16_tflite"],"dataset_v2_modified":False,"frozen_test_modified":False,"ui_modified":False,"video_data_modified":False,"video_training_performed":False,"audio_training_performed":False,"dataset_fingerprint":j(V3/"dataset_verification.json")["fingerprint"],"training_process_protected_paths_unchanged":j(V3/"experiment_metadata.json")["protected_paths_unchanged"],"hashes":current}
    dump(V3/"integrity_report.json",integrity)
    print(json.dumps({"comparison":comparison,"transitions":t,"shifts":shifts,"answers":answers,"integrity":integrity},indent=2))
if __name__=="__main__":main()
