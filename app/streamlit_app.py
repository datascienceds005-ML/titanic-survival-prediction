"""Titanic Survival Prediction — Streamlit Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib, json, os, sys
sys.path.insert(0, ".")

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
.stApp{background:linear-gradient(135deg,#0a0f1e,#0d1929,#0f2040);min-height:100vh;}
.hero{background:linear-gradient(135deg,rgba(59,130,246,0.12),rgba(16,185,129,0.12));
      border:1px solid rgba(59,130,246,0.25);border-radius:20px;
      padding:2.5rem 3rem;margin-bottom:2rem;text-align:center;}
.hero-title{font-size:2.8rem;font-weight:700;
    background:linear-gradient(135deg,#3b82f6,#10b981,#f59e0b);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;line-height:1.2;}
.hero-sub{color:rgba(255,255,255,0.5);font-size:1rem;margin-top:.5rem;}
.metric-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
    border-radius:16px;padding:1.5rem;text-align:center;transition:all .3s;}
.metric-card:hover{border-color:rgba(59,130,246,0.4);background:rgba(59,130,246,0.08);}
.metric-value{font-size:2rem;font-weight:700;color:#3b82f6;line-height:1;}
.metric-label{font-size:.75rem;color:rgba(255,255,255,0.45);margin-top:.3rem;
    text-transform:uppercase;letter-spacing:.08em;}
.survived-card{background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(16,185,129,0.05));
    border:2px solid rgba(16,185,129,0.5);border-radius:20px;padding:2rem;text-align:center;}
.died-card{background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(239,68,68,0.05));
    border:2px solid rgba(239,68,68,0.5);border-radius:20px;padding:2rem;text-align:center;}
.uncertain-card{background:linear-gradient(135deg,rgba(245,158,11,0.2),rgba(245,158,11,0.05));
    border:2px solid rgba(245,158,11,0.5);border-radius:20px;padding:2rem;text-align:center;}
.prob-survived{font-size:3.5rem;font-weight:700;color:#10b981;line-height:1;}
.prob-died{font-size:3.5rem;font-weight:700;color:#ef4444;line-height:1;}
.prob-uncertain{font-size:3.5rem;font-weight:700;color:#f59e0b;line-height:1;}
.section-hdr{font-size:1.3rem;font-weight:600;color:white;margin:1.5rem 0 1rem;
    padding-bottom:.4rem;border-bottom:2px solid rgba(59,130,246,0.3);}
.feat-card{background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.18);
    border-radius:10px;padding:.875rem 1rem;margin-bottom:.5rem;}
[data-testid="stSidebar"]{background:rgba(10,15,30,0.97)!important;
    border-right:1px solid rgba(59,130,246,0.15)!important;}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#10b981)!important;
    color:white!important;border:none!important;border-radius:12px!important;
    padding:.75rem 2rem!important;font-weight:600!important;width:100%!important;font-size:1rem!important;}
.stButton>button:hover{opacity:.85!important;}
</style>
""", unsafe_allow_html=True)

PLOT_BG = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,0.02)",
               font=dict(family="Inter",color="rgba(255,255,255,0.75)",size=12),
               margin=dict(l=20,r=20,t=40,b=20))

@st.cache_resource
def load_model(name="best"):
    paths={"best":"models/best_model.joblib","rf":"models/randomforest.joblib",
           "gb":"models/gradientboosting.joblib","lr":"models/logisticregression.joblib",
           "xgb":"models/xgboost.joblib","et":"models/extratrees.joblib"}
    p=paths.get(name,"models/best_model.joblib")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_metrics():
    p="reports/metrics/results.json"
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_raw():
    p="data/raw/titanic.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🚢</div>
        <div style='color:white;font-weight:700;font-size:1.1rem;margin-top:.5rem;'>Titanic AI</div>
        <div style='color:rgba(255,255,255,0.4);font-size:.75rem;'>Survival Predictor</div>
    </div>""", unsafe_allow_html=True)

    page = st.selectbox("Nav", ["🏠 Home","🔮 Predict Survival",
                                 "📊 Data Explorer","📈 Model Analytics","ℹ️ About"],
                        label_visibility="collapsed")
    st.markdown("---")
    model_choice = st.radio("Model",["Best Model","Random Forest","Gradient Boosting",
                                     "Logistic Regression","Extra Trees"],
                            label_visibility="collapsed")
    model_key = {"Best Model":"best","Random Forest":"rf","Gradient Boosting":"gb",
                 "Logistic Regression":"lr","Extra Trees":"et"}[model_choice]

    metrics = load_metrics()
    if metrics:
        best = metrics.get("best_model","RandomForest")
        m = metrics.get(best,{})
        st.markdown("---")
        st.markdown("<div style='color:rgba(255,255,255,0.4);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.75rem;'>Best Model Stats</div>", unsafe_allow_html=True)
        c1,c2=st.columns(2)
        c1.metric("Accuracy",f"{m.get('Accuracy',0):.3f}")
        c2.metric("AUC",f"{m.get('ROC_AUC',0):.3f}")
        c1.metric("F1",f"{m.get('F1',0):.3f}")
        c2.metric("Recall",f"{m.get('Recall',0):.3f}")

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.3);font-size:.7rem;text-align:center;line-height:1.7;'>
        Built by <strong style='color:rgba(255,255,255,0.6);'>Darsh Kumar</strong><br>
        B.Tech Data Science<br>Ensemble ML · SHAP
    </div>""", unsafe_allow_html=True)

# HOME
if page == "🏠 Home":
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>🚢 Titanic Survival Predictor</div>
        <div class='hero-sub'>Advanced Feature Engineering · Ensemble Models · SHAP Explainability<br>
        891 passengers · 13 raw features · 13 engineered features</div>
    </div>""", unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics:
        best = metrics.get("best_model","RandomForest")
        m = metrics.get(best,{})
        st.markdown("<div class='section-hdr'>🏆 Best Model Performance</div>", unsafe_allow_html=True)
        cols=st.columns(5)
        for col,(k,lbl) in zip(cols,[("Accuracy","Accuracy"),("ROC_AUC","ROC-AUC"),
                                      ("F1","F1 Score"),("Recall","Recall"),("Precision","Precision")]):
            col.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{m.get(k,0):.3f}</div>
                <div class='metric-label'>{lbl}</div>
                <div style='color:rgba(255,255,255,0.3);font-size:.7rem;margin-top:2px;'>{best}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Models not trained. Run `python src/models/train.py` first.")

    st.markdown("---")
    st.markdown("<div class='section-hdr'>⚙️ Engineered Features</div>", unsafe_allow_html=True)
    feats=[("FamilySize","SibSp+Parch+1","Large families struggled more"),
           ("IsAlone","Traveling solo flag","Alone passengers had lower survival"),
           ("Title","Extracted from Name","Proxy for gender+age+status"),
           ("FarePerPerson","Fare÷FamilySize","Normalised wealth indicator"),
           ("AgeGroup","Child/Teen/Adult etc","Children had priority"),
           ("IsFemale","Binary sex flag","#1 survival predictor"),
           ("CabinKnown","Cabin recorded?","Known=higher class=better odds"),
           ("PclassSex","Pclass×Sex combo","1st class female=97% survival"),
           ("AgePclass","Age×Pclass","Young 3rd class=worst odds"),
           ("IsChild","Age<12 flag","Children prioritised in evacuation"),
           ("Deck","Cabin letter","Deck proximity to lifeboats"),
           ("FamilyType","Solo/Small/Medium/Large","Family category survival rate"),]
    c1,c2=st.columns(2)
    for i,(name,desc,insight) in enumerate(feats):
        t=c1 if i%2==0 else c2
        t.markdown(f"""<div class='feat-card'>
            <span style='color:#60a5fa;font-weight:600;font-size:.85rem;'>{name}</span>
            <div style='color:rgba(255,255,255,0.55);font-size:.78rem;margin-top:2px;'>{desc}</div>
            <div style='color:#86efac;font-size:.72rem;margin-top:2px;'>💡 {insight}</div>
        </div>""", unsafe_allow_html=True)

# PREDICT
elif page == "🔮 Predict Survival":
    st.markdown("""
    <div class='hero' style='padding:1.75rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem;font-weight:700;color:white;'>🔮 Predict Survival</div>
        <div class='hero-sub'>Enter passenger details to predict survival probability</div>
    </div>""", unsafe_allow_html=True)

    model = load_model(model_key)
    if model is None:
        st.error("Model not found. Run `python src/models/train.py` first.")
        st.stop()

    with st.form("titanic_form"):
        st.markdown("<div class='section-hdr'>👤 Passenger Info</div>", unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        pclass   = c1.selectbox("Passenger Class",["1 - First","2 - Second","3 - Third"])
        sex      = c2.selectbox("Sex",["female","male"])
        age      = c3.number_input("Age",0.17,80.0,28.0,0.5)
        embarked = c4.selectbox("Embarked",["S - Southampton","C - Cherbourg","Q - Queenstown"])

        c1,c2,c3=st.columns(3)
        sibsp = c1.number_input("Siblings/Spouse aboard",0,8,0)
        parch = c2.number_input("Parents/Children aboard",0,6,0)
        fare  = c3.number_input("Ticket Fare ($)",0.0,512.0,32.0,0.5)

        c1,c2=st.columns(2)
        cabin_known = c1.selectbox("Cabin Known?",["No","Yes"])
        deck        = c2.selectbox("Deck",["Unknown","A","B","C","D","E","F","G"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Predict Survival", use_container_width=True)

    if submitted:
        pclass_int = int(pclass[0])
        emb = embarked[0]
        fam_size = sibsp + parch + 1
        title_val = "Miss" if sex=="female" else "Mr"

        features = {
            "Pclass": pclass_int, "Sex": sex, "Age": age,
            "SibSp": sibsp, "Parch": parch, "Fare": fare,
            "Embarked": emb,
            "FamilySize": fam_size,
            "IsAlone": int(fam_size==1),
            "Title": title_val,
            "FarePerPerson": round(fare/fam_size, 4),
            "AgeGroup": "Child" if age<12 else "Teen" if age<18 else "Adult" if age<35 else "MiddleAge" if age<60 else "Senior",
            "FareBin": "VeryHigh" if fare>50 else "High" if fare>25 else "Mid" if fare>12 else "Low",
            "CabinKnown": int(cabin_known=="Yes"),
            "Deck": deck,
            "FamilyType": "Solo" if fam_size==1 else "Small" if fam_size<=3 else "Medium" if fam_size<=5 else "Large",
            "PclassSex": f"{pclass_int}_{sex}",
            "AgePclass": round(age*pclass_int, 2),
            "IsChild": int(age<12),
            "IsFemale": int(sex=="female"),
        }

        df_input = pd.DataFrame([features])
        prob = float(model.predict_proba(df_input)[0][1])
        pred = int(model.predict(df_input)[0])

        if prob >= 0.70:
            card="survived-card"; prob_cls="prob-survived"
            verdict="🟢 LIKELY SURVIVED"; color="#10b981"
        elif prob >= 0.40:
            card="uncertain-card"; prob_cls="prob-uncertain"
            verdict="🟡 UNCERTAIN"; color="#f59e0b"
        else:
            card="died-card"; prob_cls="prob-died"
            verdict="🔴 LIKELY DID NOT SURVIVE"; color="#ef4444"

        c1,c2=st.columns([1,1])
        with c1:
            st.markdown(f"""
            <div class='{card}'>
                <div style='font-size:.9rem;color:rgba(255,255,255,0.5);text-transform:uppercase;
                            letter-spacing:.1em;margin-bottom:.5rem;'>Survival Probability</div>
                <div class='{prob_cls}'>{prob*100:.1f}%</div>
                <div style='color:rgba(255,255,255,0.5);font-size:1rem;font-weight:600;
                            margin-top:.75rem;'>{verdict}</div>
                <div style='color:rgba(255,255,255,0.35);font-size:.78rem;margin-top:.5rem;'>
                    Class {pclass_int} · {sex.capitalize()} · Age {age:.0f} · ${fare:.0f} fare
                </div>
            </div>""", unsafe_allow_html=True)

            fig_g=go.Figure(go.Indicator(
                mode="gauge+number",value=prob*100,
                number={"suffix":"%","font":{"size":28,"color":color}},
                title={"text":"Survival Probability","font":{"size":13,"color":"rgba(255,255,255,0.6)"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"rgba(255,255,255,0.3)"},
                       "bar":{"color":color,"thickness":.25},
                       "bgcolor":"rgba(255,255,255,0.03)","borderwidth":0,
                       "steps":[{"range":[0,40],"color":"rgba(239,68,68,0.15)"},
                                 {"range":[40,70],"color":"rgba(245,158,11,0.15)"},
                                 {"range":[70,100],"color":"rgba(16,185,129,0.15)"}],
                       "threshold":{"line":{"color":"white","width":2},"value":prob*100}}))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=210,
                                margin=dict(l=20,r=20,t=30,b=10),
                                font=dict(color="rgba(255,255,255,0.7)"))
            st.plotly_chart(fig_g, use_container_width=True)

        with c2:
            st.markdown("<div class='section-hdr' style='margin-top:0;'>📊 Key Factors</div>", unsafe_allow_html=True)
            factors=[
                ("Sex",f"{sex.capitalize()}","Strongest predictor — women first",
                 "#10b981" if sex=="female" else "#ef4444"),
                ("Class",f"{pclass_int}st/2nd/3rd",
                 "1st class had most lifeboats access",
                 "#10b981" if pclass_int==1 else "#f59e0b" if pclass_int==2 else "#ef4444"),
                ("Age",f"{age:.0f} years",
                 "Children prioritised in evacuation",
                 "#10b981" if age<12 else "#f59e0b"),
                ("Family",f"Size: {fam_size}",
                 "Solo or small family best odds",
                 "#10b981" if fam_size in [2,3] else "#f59e0b"),
                ("Fare",f"${fare:.0f}",
                 "Higher fare = better cabin location",
                 "#10b981" if fare>50 else "#f59e0b" if fare>20 else "#ef4444"),
            ]
            for fname,fval,finsight,fcolor in factors:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px;padding:.75rem 1rem;margin-bottom:.5rem;
                            display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <span style='color:{fcolor};font-weight:600;font-size:.85rem;'>{fname}: {fval}</span>
                        <div style='color:rgba(255,255,255,0.4);font-size:.75rem;margin-top:1px;'>{finsight}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

# DATA EXPLORER
elif page == "📊 Data Explorer":
    st.markdown("""
    <div class='hero' style='padding:1.75rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem;font-weight:700;color:white;'>📊 Data Explorer</div>
        <div class='hero-sub'>Explore the Titanic dataset — 891 passengers · 12 features</div>
    </div>""", unsafe_allow_html=True)

    df=load_raw()
    if df is None:
        st.error("Dataset not found.")
        st.stop()

    c1,c2,c3,c4=st.columns(4)
    surv_rate = df["Survived"].mean() if "Survived" in df.columns else 0
    for col,(val,lbl) in zip([c1,c2,c3,c4],[
        (f"{len(df):,}","Passengers"),(f"{surv_rate*100:.1f}%","Survival Rate"),
        (f"{df['Age'].mean():.0f} yrs","Avg Age"),(f"${df['Fare'].mean():.0f}","Avg Fare"),
    ]):
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["🚢 Survival Analysis","👥 Demographics","💰 Fare & Class"])

    with tab1:
        c1,c2=st.columns(2)
        sv=df["Survived"].value_counts()
        fig_sv=go.Figure(go.Pie(
            labels=["Died","Survived"],values=[sv.get(0,0),sv.get(1,0)],
            hole=.55,marker=dict(colors=["#ef4444","#10b981"]),
            textinfo="percent+label"))
        fig_sv.update_layout(**PLOT_BG,title="Overall Survival",height=300)
        c1.plotly_chart(fig_sv,use_container_width=True)

        if "Sex" in df.columns:
            sg=df.groupby(["Sex","Survived"]).size().reset_index(name="count")
            fig_sg=px.bar(sg,x="Sex",y="count",color=sg["Survived"].map({0:"Died",1:"Survived"}),
                         color_discrete_map={"Died":"#ef4444","Survived":"#10b981"},
                         barmode="group",title="Survival by Sex")
            fig_sg.update_layout(**PLOT_BG,height=300)
            c2.plotly_chart(fig_sg,use_container_width=True)

        if "Pclass" in df.columns:
            pc=df.groupby(["Pclass","Survived"]).size().reset_index(name="count")
            fig_pc=px.bar(pc,x="Pclass",y="count",color=pc["Survived"].map({0:"Died",1:"Survived"}),
                         color_discrete_map={"Died":"#ef4444","Survived":"#10b981"},
                         barmode="group",title="Survival by Passenger Class",
                         labels={"Pclass":"Class"})
            fig_pc.update_layout(**PLOT_BG,height=300)
            st.plotly_chart(fig_pc,use_container_width=True)

    with tab2:
        c1,c2=st.columns(2)
        fig_age=px.histogram(df,x="Age",color=df["Survived"].map({0:"Died",1:"Survived"}),
                             nbins=30,barmode="overlay",opacity=.7,
                             color_discrete_map={"Died":"#ef4444","Survived":"#10b981"},
                             title="Age Distribution by Survival")
        fig_age.update_layout(**PLOT_BG,height=320)
        c1.plotly_chart(fig_age,use_container_width=True)

        df["FamilySize"]=df["SibSp"]+df["Parch"]+1
        fs=df.groupby("FamilySize")["Survived"].mean().reset_index()
        fig_fs=px.bar(fs,x="FamilySize",y="Survived",
                      title="Survival Rate by Family Size",
                      labels={"Survived":"Survival Rate","FamilySize":"Family Size"},
                      color="Survived",color_continuous_scale="RdYlGn")
        fig_fs.update_layout(**PLOT_BG,height=320)
        c2.plotly_chart(fig_fs,use_container_width=True)

    with tab3:
        c1,c2=st.columns(2)
        fig_fare=px.box(df,x=df["Survived"].map({0:"Died",1:"Survived"}),y="Fare",
                        color=df["Survived"].map({0:"Died",1:"Survived"}),
                        color_discrete_map={"Died":"#ef4444","Survived":"#10b981"},
                        title="Fare Distribution by Survival",
                        labels={"x":"","Fare":"Fare ($)"})
        fig_fare.update_layout(**PLOT_BG,height=320)
        c1.plotly_chart(fig_fare,use_container_width=True)

        emb=df.groupby(["Embarked","Survived"]).size().reset_index(name="count")
        fig_emb=px.bar(emb,x="Embarked",y="count",
                       color=emb["Survived"].map({0:"Died",1:"Survived"}),
                       color_discrete_map={"Died":"#ef4444","Survived":"#10b981"},
                       barmode="stack",title="Embarked Port vs Survival")
        fig_emb.update_layout(**PLOT_BG,height=320)
        c2.plotly_chart(fig_emb,use_container_width=True)

# MODEL ANALYTICS
elif page == "📈 Model Analytics":
    st.markdown("""
    <div class='hero' style='padding:1.75rem 2rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem;font-weight:700;color:white;'>📈 Model Analytics</div>
        <div class='hero-sub'>Baseline · Ensemble · Comparison</div>
    </div>""", unsafe_allow_html=True)

    metrics=load_metrics()
    if metrics is None:
        st.error("No metrics. Run `python src/models/train.py` first.")
        st.stop()

    names=[k for k in metrics if k!="best_model"]
    keys=["Accuracy","F1","ROC_AUC","Recall","Precision"]
    comp={"Model":names}
    for k in keys:
        comp[k]=[f"{metrics[n].get(k,0):.4f}" for n in names]
    st.dataframe(pd.DataFrame(comp),use_container_width=True,hide_index=True)

    st.markdown("<div class='section-hdr'>📊 Visual Comparison</div>", unsafe_allow_html=True)
    colors=["#3b82f6","#10b981","#f59e0b","#ef4444","#a855f7","#06b6d4","#f97316","#84cc16"]

    for metric in ["Accuracy","F1","ROC_AUC"]:
        vals=[metrics.get(n,{}).get(metric,0) for n in names]
        fig=go.Figure(go.Bar(x=names,y=vals,
                             marker=dict(color=colors[:len(names)]),
                             text=[f"{v:.3f}" for v in vals],textposition="outside",
                             textfont=dict(color="rgba(255,255,255,0.9)")))
        fig.update_layout(**PLOT_BG,title=f"{metric} — All Models",
                          yaxis=dict(**PLOT_BG.get("yaxis",{}),range=[0,1.08]),height=300)
        st.plotly_chart(fig,use_container_width=True)

    best=metrics.get("best_model","N/A")
    st.markdown(f"""
    <div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                border-radius:12px;padding:1rem 1.5rem;color:#6ee7b7;margin-top:1rem;'>
        🏆 Best Model: <strong>{best}</strong> — selected by highest Accuracy on held-out test set.
    </div>""", unsafe_allow_html=True)

# ABOUT
elif page == "ℹ️ About":
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>🚢 About This Project</div>
        <div class='hero-sub'>Titanic Survival Prediction · Darsh Kumar · B.Tech Data Science</div>
    </div>""", unsafe_allow_html=True)

    c1,c2=st.columns([2,1])
    c1.markdown("""
    <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                border-radius:12px;padding:1.5rem;'>
        <div style='color:#60a5fa;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;'>Project</div>
        <div style='color:rgba(255,255,255,0.7);font-size:.9rem;line-height:1.8;margin-top:.75rem;'>
            This project builds an end-to-end ML pipeline to predict Titanic passenger survival
            using <strong style='color:#60a5fa;'>ensemble methods</strong> including Random Forest,
            Gradient Boosting, Extra Trees, and XGBoost with 13 engineered features and SHAP explainability.
        </div>
    </div>""", unsafe_allow_html=True)

    c2.markdown("""
    <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                border-radius:12px;padding:1.5rem;'>
        <div style='color:#60a5fa;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;'>Stack</div>
        <div style='margin-top:.75rem;'>""" +
    "".join([f"<div style='display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem;'>"
             f"<div style='width:7px;height:7px;border-radius:50%;background:{c};'></div>"
             f"<span style='color:rgba(255,255,255,0.7);font-size:.83rem;'>{t}</span></div>"
             for t,c in [("Python 3.11","#3b82f6"),("scikit-learn 1.4","#10b981"),
                         ("XGBoost + SHAP","#f59e0b"),("Streamlit + Plotly","#ef4444"),
                         ("GitHub Codespaces","#a855f7")]]) +
    "</div></div>", unsafe_allow_html=True)
