import os, math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")

def download_nifty(period="60d", interval="15m"):
    df = yf.download("^NSEI", period=period, interval=interval, auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no NIFTY data.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    if "Volume" not in df.columns: df["Volume"] = 0
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df

def rsi(s, n=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n, adjust=False).mean()
    al=l.ewm(alpha=1/n, adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    pc=df.Close.shift()
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def adx(df,n=14):
    up=df.High.diff(); dn=-df.Low.diff()
    p=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    m=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    pc=df.Close.shift()
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    av=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*p.ewm(alpha=1/n,adjust=False).mean()/av
    mdi=100*m.ewm(alpha=1/n,adjust=False).mean()/av
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

def indicators(df):
    x=df.copy()
    x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean()
    x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
    x["RSI"]=rsi(x.Close); x["ATR"]=atr(x); x["ATR_PCT"]=x.ATR/x.Close; x["ADX"]=adx(x)
    tp=(x.High+x.Low+x.Close)/3; day=x.index.date
    x["VWAP"]=(tp*x.Volume).groupby(day).cumsum()/x.Volume.groupby(day).cumsum().replace(0,np.nan)
    x["RVOL"]=x.Volume/x.Volume.rolling(20).mean().replace(0,np.nan)
    x["BODY_RATIO"]=(x.Close-x.Open).abs()/(x.High-x.Low).replace(0,np.nan)
    x["HIGH20"]=x.High.rolling(20).max().shift(1); x["LOW20"]=x.Low.rolling(20).min().shift(1)
    return x

def daily_context(df, target):
    d=df.resample("1D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    d["EMA5"]=d.Close.ewm(span=5,adjust=False).mean()
    d["EMA10"]=d.Close.ewm(span=10,adjust=False).mean()
    d["RANGE"]=(d.High-d.Low)/d.Open
    p=d[d.index<pd.Timestamp(target)].tail(7)
    if len(p)<7: return None
    last=p.iloc[-1]; score=0; reasons=[]
    if last.Close>last.EMA5: score+=1; reasons.append("above 5D EMA")
    if last.EMA5>last.EMA10: score+=1; reasons.append("5D trend positive")
    if p.Close.iloc[-1]>p.Close.iloc[0]: score+=1; reasons.append("7D momentum positive")
    if p.RANGE.iloc[-3:].mean()>p.RANGE.mean(): score+=1; reasons.append("range expansion")
    if last.Close>last.Open: score+=1; reasons.append("previous day green")
    direction="CALL" if score>=3 else "PUT" if score<=1 else "WAIT"
    return {"score":score,"direction":direction,"reasons":reasons}

def signal(row):
    l=s=0; lr=[]; sr=[]
    if row.Close>row.VWAP: l+=1; lr.append("above VWAP")
    elif row.Close<row.VWAP: s+=1; sr.append("below VWAP")
    if row.EMA20>row.EMA50: l+=1; lr.append("EMA20 > EMA50")
    elif row.EMA20<row.EMA50: s+=1; sr.append("EMA20 < EMA50")
    if row.ADX>=20:
        if row.EMA20>row.EMA50: l+=1; lr.append("ADX trend strength")
        elif row.EMA20<row.EMA50: s+=1; sr.append("ADX trend strength")
    if row.RVOL>=1.2:
        if row.Close>row.Open: l+=1; lr.append("volume expansion")
        elif row.Close<row.Open: s+=1; sr.append("volume expansion")
    if row.BODY_RATIO>=.60:
        if row.Close>row.Open: l+=1; lr.append("strong bullish candle")
        elif row.Close<row.Open: s+=1; sr.append("strong bearish candle")
    if row.Close>row.HIGH20: l+=2; lr.append("20-bar breakout")
    if row.Close<row.LOW20: s+=2; sr.append("20-bar breakdown")
    if l>=5 and l>s: return "CALL",l,lr
    if s>=5 and s>l: return "PUT",s,sr
    return "WAIT",max(l,s),[]

def normcdf(x): return .5*(1+math.erf(x/math.sqrt(2)))
def bs(spot,strike,t,iv=.18,rate=.06,call=True):
    if t<=0: return max(spot-strike,0) if call else max(strike-spot,0)
    v=max(iv,.01); d1=(math.log(spot/strike)+(rate+.5*v*v)*t)/(v*math.sqrt(t)); d2=d1-v*math.sqrt(t)
    if call: return spot*normcdf(d1)-strike*math.exp(-rate*t)*normcdf(d2)
    return strike*math.exp(-rate*t)*normcdf(-d2)-spot*normcdf(-d1)

def backtest(df, lot=75):
    x=indicators(df); trades=[]; equity=100000; peak=equity; dd=0
    for day in sorted(set(x.index.date)):
        ctx=daily_context(df,day)
        if not ctx or ctx["direction"]=="WAIT": continue
        z=x[x.index.date==day]
        z=z[(z.index.hour>9)|((z.index.hour==9)&(z.index.minute>=45))]
        for i in range(len(z)-1):
            r=z.iloc[i]; side,score,reasons=signal(r)
            if side!=ctx["direction"] or score<5: continue
            ent=z.iloc[i+1]; ep=float(ent.Open); strike=round(ep/50)*50
            pe=bs(ep,strike,1/252,call=side=="CALL")
            stop=max(float(r.ATR),ep*.002); target=2*stop
            xp=float(z.iloc[-1].Close); xt=z.index[-1]; why="EOD"
            for j in range(i+1,len(z)):
                b=z.iloc[j]
                if side=="CALL":
                    if b.Low<=ep-stop: xp=ep-stop; xt=z.index[j]; why="SL"; break
                    if b.High>=ep+target: xp=ep+target; xt=z.index[j]; why="TARGET"; break
                else:
                    if b.High>=ep+stop: xp=ep+stop; xt=z.index[j]; why="SL"; break
                    if b.Low<=ep-target: xp=ep-target; xt=z.index[j]; why="TARGET"; break
            px=bs(xp,strike,max((xt-ent.index).total_seconds()/(365*86400),1/31536000),call=side=="CALL")
            pnl=(px-pe)*lot; equity+=pnl; peak=max(peak,equity); dd=max(dd,peak-equity)
            trades.append({"date":str(day),"entry_time":str(ent.name),"exit_time":str(xt),
                           "side":side,"strike":strike,"spot_entry":round(ep,2),
                           "spot_exit":round(xp,2),"premium_entry":round(pe,2),
                           "premium_exit":round(px,2),"pnl":round(pnl,2),
                           "exit_reason":why,"signal_score":score,"lookback_score":ctx["score"],
                           "reasons":", ".join(reasons),"equity":round(equity,2)})
            break
    wins=sum(t["pnl"]>0 for t in trades)
    metrics={"trades":len(trades),"wins":wins,"losses":len(trades)-wins,
             "win_rate":round(100*wins/len(trades),2) if trades else 0,
             "net_pnl":round(sum(t["pnl"] for t in trades),2),
             "return_pct":round((equity/100000-1)*100,2),"max_drawdown":round(dd,2),
             "final_equity":round(equity,2)}
    return trades,metrics

@app.route("/")
def home():
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)),"static"),"index.html")

@app.get("/health")
def health(): return jsonify({"status":"ok"})

@app.get("/api/scan")
def scan():
    try:
        df=download_nifty(request.args.get("period","30d"),request.args.get("interval","15m"))
        x=indicators(df); r=x.iloc[-1]; s,score,reasons=signal(r)
        return jsonify({"status":"ok","signal":s,"score":int(score),"price":round(float(r.Close),2),
                        "adx":round(float(r.ADX),2),"rvol":round(float(r.RVOL),2),
                        "vwap":round(float(r.VWAP),2),"ema20":round(float(r.EMA20),2),
                        "ema50":round(float(r.EMA50),2),"reasons":reasons,"timestamp":str(x.index[-1])})
    except Exception as e: return jsonify({"status":"error","error":str(e)}),500

@app.get("/api/option-backtest")
def option_backtest():
    try:
        df=download_nifty(request.args.get("period","60d"),request.args.get("interval","15m"))
        trades,metrics=backtest(df)
        return jsonify({"status":"ok","metrics":metrics,"trades":trades,
                        "note":"Option premiums are modeled with Black-Scholes from NIFTY spot history; this is not a historical NSE option-premium feed."})
    except Exception as e: return jsonify({"status":"error","error":str(e)}),500

@app.get("/api/backtest")
def api_backtest(): return option_backtest()

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=False)
