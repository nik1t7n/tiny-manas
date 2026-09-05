"""Deterministic vector PDF figures from the extracted real measurements."""
from pathlib import Path
import csv
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
FONT = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Figure", str(FONT / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("FigureBold", str(FONT / "Arial Bold.ttf")))
INK = "#26333A"
TEAL = "#007F79"
BLUE = "#366AAA"
RUST = "#B85935"
GREY = "#757B80"


def new(name, w, h):
    c = canvas.Canvas(str(OUT / name), pagesize=(w, h), invariant=1,
                      initialFontName="Figure", initialFontSize=8)
    c.setTitle("Tiny Manas — " + name.removesuffix(".pdf"))
    c.setAuthor("Nikita Nosov")
    return c


def text(c, x, y, s, size=8, bold=False, color=INK, align="left"):
    c.setFont("FigureBold" if bold else "Figure", size)
    c.setFillColor(HexColor(color))
    getattr(c, {"left": "drawString", "center": "drawCentredString", "right": "drawRightString"}[align])(x, y, s)


def line(c, points, color=INK, width=1, dashed=False):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
    if isinstance(dashed, tuple):
        c.setDash(*dashed)
    elif dashed:
        c.setDash(3, 2)
    else:
        c.setDash()
    p = c.beginPath(); p.moveTo(*points[0])
    for x, y in points[1:]: p.lineTo(x, y)
    c.drawPath(p)
    c.setDash()


def arrow(c, points, color=INK, width=1):
    import math
    line(c, points, color, width)
    (x0, y0), (x, y) = points[-2:]
    a = math.atan2(y - y0, x - x0)
    d = 4
    p = c.beginPath(); p.moveTo(x, y)
    p.lineTo(x - d * math.cos(a - .45), y - d * math.sin(a - .45))
    p.lineTo(x - d * math.cos(a + .45), y - d * math.sin(a + .45)); p.close()
    c.setFillColor(HexColor(color)); c.drawPath(p, stroke=0, fill=1)


def box(c, x, y, w, h, label, fill="#F2F4F5", size=8):
    c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(INK)); c.setLineWidth(.85)
    c.roundRect(x-w/2, y-h/2, w, h, 4, fill=1, stroke=1)
    labels = label.split("\n")
    for i, s in enumerate(labels): text(c, x, y + (len(labels)-1)*5 - i*10 - 2.7, s, size, align="center")


def plus(c, x, y):
    c.setFillColor(white); c.setStrokeColor(HexColor(INK)); c.setLineWidth(1)
    c.circle(x, y, 7, stroke=1, fill=1)
    line(c, [(x-3, y), (x+3, y)]); line(c, [(x,y-3), (x,y+3)])


def architecture():
    c = new("architecture.pdf", 396, 420)
    text(c, 91, 405, "(a) Decoder architecture", 9, True, align="center")
    text(c, 286, 405, "(b) Inside causal attention", 9, True, align="center")
    x = 94
    # Outer boundary conveys repetition; residuals bypass normalization too.
    c.setStrokeColor(HexColor("#ACB8BB")); c.setFillColor(HexColor("#FAFBFB"))
    c.roundRect(14, 77, 163, 251, 8, fill=1, stroke=1)
    text(c, 23, 313, "8 ×", 8, True)
    text(c, x, 8, "Token IDs", align="center")
    box(c,x,33,112,22,"Token embedding", "#F7E1E3")
    box(c,x,61,80,16,"Dropout", "#F4F0DC",7)
    box(c,x,106,100,20,"LayerNorm", "#EFEDBD")
    box(c,x,141,119,28,"Causal multi-head\nself-attention", "#F9DFB9")
    box(c,x,171,80,16,"Dropout", "#F4F0DC",7)
    plus(c,x,194)
    box(c,x,224,100,20,"LayerNorm", "#EFEDBD")
    box(c,x,254,112,24,"Feed-forward", "#CDE8F3")
    box(c,x,282,80,16,"Dropout", "#F4F0DC",7)
    plus(c,x,306)
    box(c,x,347,100,20,"Final LayerNorm", "#EFEDBD")
    box(c,x,378,112,22,"Tied LM head", "#E2E1F1")
    for a,b in [(12,22),(44,53),(69,96),(116,127),(155,163),(179,187),(201,214),(234,242),(266,274),(290,299),(313,337),(357,367)]:
        arrow(c, [(x,a),(x,b)])
    arrow(c,[(x,85),(30,85),(30,194),(87,194)])
    arrow(c,[(x,206),(30,206),(30,306),(87,306)])
    arrow(c,[(150,378),(176,378)])
    text(c,176,385,"Logits",7,align="right")
    # Detailed attention: Q/K get rotations, V is bypassed unchanged.
    text(c,288,32,"Normalized token vectors",8,align="center")
    for xx,s in [(231,"Q"),(280,"K"),(350,"V")]:
        arrow(c,[(288,38),(288,49),(xx,49),(xx,62)])
        box(c,xx,73,34,22,s,"#ECEBF5")
    for xx in [231,280]:
        arrow(c,[(xx,84),(xx,101)])
        box(c,xx,112,42,22,"RoPE", "#DFECE8")
    box(c,256,157,88,24,"Q × K transpose","#EDEFF0")
    arrow(c,[(231,123),(231,145)]); arrow(c,[(280,123),(280,145)])
    box(c,256,196,98,30,"Scale + causal mask\n+ softmax", "#DCEAD9",7.7)
    arrow(c,[(256,169),(256,181)])
    box(c,256,233,98,22,"Attention dropout", "#F4F0DC",7.4)
    arrow(c,[(256,211),(256,222)])
    box(c,280,275,122,26,"Weighted sum of V","#EDEFF0")
    arrow(c,[(256,244),(256,262)])
    arrow(c,[(350,84),(350,275),(341,275)])
    box(c,280,319,132,30,"Concatenate 8 heads\n+ output projection", "#E2E1F1",7.6)
    arrow(c,[(280,288),(280,304)])
    arrow(c,[(280,334),(280,356)])
    text(c,280,365,"Context message: 384 values",7.6,align="center")
    text(c,280,384,"Each head has 48 features",7,color=GREY,align="center")
    c.save()


def rows(name):
    with (HERE / "data" / name).open() as f:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def axes(c, x, y, w, h, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel):
    def pt(a,b): return x+(a-xmin)/(xmax-xmin)*w, y+(b-ymin)/(ymax-ymin)*h
    for t in yticks:
        yy=pt(xmin,t)[1];line(c,[(x,yy),(x+w,yy)],"#E1E6E7",.5)
        text(c,x-6,yy-2.5,f"{t:g}",7,align="right")
    line(c,[(x,y+h),(x,y),(x+w,y)],INK,.7)
    for t in xticks:
        xx=pt(t,ymin)[0];line(c,[(xx,y),(xx,y-3)],INK,.6);text(c,xx,y-13,f"{t:g}",7,align="center")
    text(c,x+w/2,y-29,xlabel,8,align="center")
    text(c,x,y+h+10,ylabel,8)
    return pt


def series(c, data, xkey, ykey, pt, color, dashed=False):
    line(c,[pt(r[xkey],r[ykey]) for r in data],color,1.5,dashed)


def legend(c, x, y, labels):
    for label,color,dashed in labels:
        line(c,[(x,y+2),(x+15,y+2)],color,1.6,dashed)
        text(c,x+20,y,label,7.6);y-=13


def curves():
    c=new("learning-curves.pdf",396,240)
    pt=axes(c,35,42,142,151,100,1000,0,11,[100,500,1000],[0,2,4,6,8,10],"Updates","Loss (nats/token)")
    text(c,35,223,"(a) 10k-token pilot",9,True)
    d=rows("pilot.csv");series(c,d,"step","train",pt,TEAL);series(c,d,"step","validation",pt,RUST)
    legend(c,48,174,[("Train",TEAL,False),("Validation",RUST,False)])
    pt=axes(c,233,42,153,151,100,3000,3.8,8.2,[100,1500,3000],[4,5,6,7,8],"Updates","Validation loss (nats/token)")
    text(c,233,223,"(b) Full Manas01",9,True)
    for name,col,dash in [("base13",GREY,(1,2)),("context512",RUST,True),("classic27",BLUE,(6,2)),("rope",TEAL,False)]:
        series(c,rows(name+".csv"),"step","validation",pt,col,dash)
    legend(c,277,181,[("13M / 256",GREY,(1,2)),("13M / 512",RUST,True),("27M / learned",BLUE,(6,2)),("27M / RoPE",TEAL,False)])
    c.save()
    c=new("staged-quality.pdf",396,220)
    pt=axes(c,40,44,342,139,100,3000,-.45,.025,[100,600,900,1500,3000],[-.4,-.3,-.2,-.1,0],"Optimizer updates","Validation loss difference (nats/token; candidate minus control)")
    line(c,[pt(100,-.02),pt(3000,-.02)],GREY,.8,True)
    series(c,rows("rope-delta.csv"),"step","delta",pt,TEAL)
    series(c,rows("rmsnorm-delta.csv"),"step","delta",pt,RUST,True)
    legend(c,216,135,[("RoPE: completed 3,000",TEAL,False),("RMSNorm: stopped at 900",RUST,True),("Practical gain floor: −0.02",GREY,True)])
    text(c,40,203,"Lower is better; repeated checks are not independent seeds",8,True)
    c.save()
    c=new("tokenizer-curves.pdf",396,205)
    pt=axes(c,40,40,342,125,1,30,.85,1.55,[1,10,20,30],[.9,1.1,1.3,1.5],"Complete training-text passes","Validation bits per byte")
    for name,col,dash in [(32768,BLUE,(6,2)),(16384,RUST,True),(8192,TEAL,False)]:
        series(c,rows(f"vocab{name}.csv"),"epoch","bpb",pt,col,dash)
    line(c,[pt(1,.8764423051),pt(30,.8764423051)],GREY,1,(1,2))
    legend(c,208,148,[("32k / equal-text recipe",BLUE,(6,2)),("16k / equal-text recipe",RUST,True),("8k / equal-text recipe",TEAL,False),("Incumbent / random windows",GREY,(1,2))])
    text(c,40,187,"Same target bytes; different token boundaries",8,True)
    c.save()


def cache():
    c=new("cache.pdf",396,192)
    for x,title,sub in [(67,"Prefill","Process prompt once"),(202,"Decode","One new query per step"),(335,"At 256-token limit","Rebuild cropped context")]:
        text(c,x,175,title,8.5,True,align="center");text(c,x,161,sub,7,align="center")
    for x,n,active in [(25,5,5),(158,6,1),(291,6,6)]:
        for i in range(n):
            fill="#D9EBE6" if i>=n-active else "#ECEFF1"
            box(c,x+7+i*15,129,12,17,"",fill)
    arrow(c,[(108,129),(149,129)])
    arrow(c,[(244,129),(283,129)])
    box(c,67,84,98,30,"Store each layer’s\nkeys and values", "#DFE9F5",7.5)
    box(c,202,84,103,30,"Read old K/V\nAppend new K/V", "#DFE9F5",7.5)
    box(c,335,84,108,30,"Old states are invalid\nfor exact crop parity", "#F5E2D7",7.4)
    for x in [67,202,335]:arrow(c,[(x,116),(x,100)])
    text(c,198,40,"Cache state belongs to one request; training has no KV cache.",8,align="center")
    text(c,198,20,"MHA: 6 MiB     GQA (2 KV heads): 1.5 MiB     (FP32, one full request)",8,True,align="center")
    c.save()


def followup():
    import json
    import math
    evidence = json.loads((HERE / "data/followup-evidence.json").read_text())
    runs = {name: rows(f"followup-{name}.csv") for name in ("old256", "expanded256", "context512")}
    c = new("data-context-followup.pdf", 396, 350)
    styles = [("old256", GREY, (1, 2)), ("expanded256", TEAL, False), ("context512", RUST, (6, 2))]
    for key, y, title in [("primary", 220, "(a) New book: Orozbakov 4"),
                           ("familiar", 61, "(b) Familiar source: Karalaev suffix")]:
        values = [r[key] for d in runs.values() for r in d]
        lo, hi = math.floor(min(values)), math.ceil(max(values))
        ticks = list(range(lo, hi+1, 2 if hi-lo > 6 else 1))
        pt = axes(c, 39, y, 344, 98, 100, 3000, lo, hi, [100, 1000, 2000, 3000], ticks,
                  "Optimizer updates", "Loss (nats/token)")
        text(c, 39, y+119, title, 8.5, True)
        for name, col, dash in styles:
            series(c, runs[name], "step", key, pt, col, dash)
    context_mixture = evidence["runs"]["context512"]["protocol"]["mixture"]
    context_label = "Original" if context_mixture == "old" else context_mixture.title()
    for x, label, col, dash in [(15, "Original / 256", GREY, (1, 2)), (133, "Expanded / 256", TEAL, False),
                               (278, f"{context_label} / 512", RUST, (6, 2))]:
        line(c, [(x, 14), (x+15, 14)], col, 1.5, dash)
        text(c, x+20, 11, label, 7.3)
    c.save()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--followup", action="store_true", help="Build only the O14-O17 figure from completed observed CSVs")
    args = parser.parse_args()
    if args.followup:
        followup()
        print("Wrote the data/context figure from observed follow-up curves.")
    else:
        architecture()
        curves()
        cache()
        print("Wrote five vector figures with embedded fonts; curves use observed CSV data.")
