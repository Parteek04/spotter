from pathlib import Path
import argparse,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
IDS={f"TE-{i:06d}" for i in range(1,12001)}
DATES=pd.date_range("2025-12-01","2025-12-31")
def fail(x): raise SystemExit("ERROR: "+x)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); ap.add_argument("--december-predictions",required=True); ap.add_argument("--output-dir",default="scorer_results"); a=ap.parse_args()
 p=pd.read_csv(a.predictions)
 if list(p.columns)!=["load_id","predicted_rate"] or len(p)!=12000 or set(p.load_id.astype(str))!=IDS or p.load_id.duplicated().any() or (pd.to_numeric(p.predicted_rate,errors="coerce")<=0).any(): fail("invalid validation_predictions.csv")
 d=pd.read_csv(a.december_predictions); cols=["pickup","delivery","distance","equipment","weight","date","predicted_rate"]
 if list(d.columns)!=cols or len(d)!=31: fail("invalid December prediction file")
 d.date=pd.to_datetime(d.date,errors="coerce")
 if set(d.date)!=set(DATES) or d.date.duplicated().any() or not d.pickup.eq("Lexington").all() or not d.delivery.eq("Fort Wayne").all() or not d.equipment.eq("Dry Van").all() or not np.isclose(d.distance,360).all() or not np.isclose(d.weight,32000).all() or (d.predicted_rate<=0).any(): fail("invalid December inputs")
 d=d.sort_values("date"); Path(a.output_dir).mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(10.8,4.8),dpi=180); c="#064A56"
 ax.plot(d.date,d.predicted_rate,color=c,linewidth=2.6,marker="o",markersize=3.2)
 floor=float(d.predicted_rate.min()); ax.fill_between(d.date,d.predicted_rate,floor-max(10,floor*.02),color=c,alpha=.08)
 ax.set_title("Candidate: December 2025 Predicted Load Rate",loc="left",fontsize=15,fontweight="bold",pad=12)
 ax.set_ylabel("Predicted rate ($)"); ax.grid(axis="y",color="#D9E2E4",linewidth=.8); ax.spines[["top","right"]].set_visible(False); ax.tick_params(axis="x",rotation=35)
 ax.text(0,-.40,"Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",transform=ax.transAxes,fontsize=9.5,color="#455A60")
 fig.tight_layout(rect=(0,.12,1,1)); fig.savefig(Path(a.output_dir)/"candidate_december.png",bbox_inches="tight"); plt.close(fig)
 print("Validated 12,000 final predictions."); print("Validated 31 fixed December predictions.")
if __name__=="__main__": main()
