"""Local catalog recommendation UI backed by the retained, trained MLP.
Run: .venv/Scripts/python.exe -X utf8 web/server.py
"""
import hashlib
import itertools
import json
import mimetypes
from pathlib import Path
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from outfit_ranker import OutfitRanker

LABELS = {'top':'Áo', 'bottom':'Quần / váy', 'shoes':'Giày'}
CATEGORIES = {'sweater':'Áo len','male T-shirt':'Áo thun','blouse':'Áo blouse','tank':'Áo hai dây','sweathirt':'Áo nỉ','hoodie':'Áo hoodie','vest':'Áo gile','tunic':'Áo dáng dài','male shirt':'Áo sơ mi nam','male sweater':'Áo len nam','male polos':'Áo polo nam','sports bra':'Áo thể thao','skirt':'Chân váy','long skirt':'Chân váy dài','pants':'Quần dài','jeans':'Quần jeans','shorts':'Quần short','sweatpants':'Quần nỉ','male jeans':'Quần jeans nam','male pants':'Quần dài nam','heels':'Giày cao gót','pump':'Giày bít mũi cao gót','sneakers':'Giày sneaker','male sneakers':'Sneaker nam','flat sandals':'Sandal đế bệt','flats':'Giày bệt','boots':'Bốt','heeled boots':'Bốt cao gót','slippers':'Dép','flip-flops':'Dép xỏ ngón','closed shoes':'Giày kín mũi','platform shoes':'Giày đế cao','flat boots':'Bốt đế bệt'}


class Recommender:
    def __init__(self):
        archive = ROOT/'artifacts/embedding/catalog_vectors.npz'
        z = np.load(archive, allow_pickle=False)
        ids = z['ids'].astype(str).tolist()
        self.index = {iid:i for i,iid in enumerate(ids)}
        self.vectors = z['vectors']
        all_items = json.loads((ROOT/'data/processed/polyvore/items.json').read_text(encoding='utf-8'))
        self.items = {iid:all_items[iid] for iid in ids}
        self.ids = ids.copy()
        random.Random(2026).shuffle(self.ids)
        self.pools = {s:[i for i in self.ids if self.items[i]['slot']==s] for s in LABELS}
        self.rules = json.loads((ROOT/'artifacts/mining/fpgrowth_rules.json').read_text(encoding='utf-8'))
        self.model = OutfitRanker(ROOT/'artifacts/outfit_ranker/best.pt', hashlib.sha256(archive.read_bytes()).hexdigest())
        self.lock = threading.Lock()

    def item(self,iid):
        r=self.items[iid]
        return {'id':iid,'slot':r['slot'],'category':r['category'],
                'label':CATEGORIES.get(r['category'],r['category']), 'name':r['description'] or CATEGORIES.get(r['category'],r['category']),
                'image':'/image/'+iid+'.jpg'}

    def catalog(self,slot='top',category='',query='',page=1):
        if slot not in LABELS:raise ValueError('Loại trang phục không hợp lệ.')
        pool=self.pools[slot]
        query=query.casefold().strip()[:100]
        found=[i for i in pool if (not category or self.items[i]['category']==category) and
               (not query or query in (self.items[i]['description']+' '+CATEGORIES.get(self.items[i]['category'],'')+' '+self.items[i]['category']).casefold())]
        pages=max(1,(len(found)+11)//12);page=max(1,min(page,pages))
        categories=sorted(set(self.items[i]['category'] for i in pool),key=lambda c:CATEGORIES.get(c,c))
        return {'items':[self.item(i) for i in found[(page-1)*12:page*12]],'total':len(found),'page':page,'pages':pages,
                'categories':[{'value':c,'label':CATEGORIES.get(c,c)} for c in categories]}

    def recommend(self,iid,diverse=False):
        if iid not in self.items:raise ValueError('Món đồ không có trong danh mục.')
        anchor=self.items[iid];anchor_token=anchor['slot']+':'+anchor['category']
        rng=random.Random(int(hashlib.sha256(iid.encode()).hexdigest()[:12],16))
        pools=[]
        for slot in LABELS:
            if slot==anchor['slot']:
                pools.append([iid]);continue
            # Limit combinations. Prefer categories from rules involving the anchor,
            # while keeping 16 random fallback candidates per remaining slot.
            priorities={}
            for rule in self.rules:
                tokens=set(rule['antecedents'])|set(rule['consequents'])
                if anchor_token in tokens:
                    for token in tokens:
                        if token.startswith(slot+':'):
                            priorities[token.split(':',1)[1]]=max(priorities.get(token.split(':',1)[1],0),rule['confidence'])
            available=self.pools[slot]
            baseline=rng.sample(available,min(16,len(available)))
            sampled=rng.sample(available,min(1000,len(available)))
            sampled.sort(key=lambda i:priorities.get(self.items[i]['category'],0),reverse=True)
            chosen=list(dict.fromkeys(baseline+sampled))[:48]
            pools.append(chosen)
        triples=list(itertools.product(*pools))
        v=self.vectors[np.array([[self.index[i] for i in t] for t in triples])]
        a,b,c=v[:,0],v[:,1],v[:,2]
        x=np.concatenate((a,b,c,a*b,a*c,b*c),axis=1)
        with self.lock: scores=self.model.predict_proba(x)[:,1]
        matched=[];rule_scores=[]
        for triple in triples:
            tokens={self.items[i]['slot']+':'+self.items[i]['category'] for i in triple}
            matches=[r for r in self.rules if set(r['antecedents'])|set(r['consequents'])<=tokens]
            rule=max(matches,key=lambda r:r['confidence']) if matches else None
            matched.append(rule);rule_scores.append(rule['confidence'] if rule else 0)
        relevance=.75*scores+.25*np.array(rule_scores)
        remaining=list(np.argsort(-relevance,kind='stable'));selected=[]
        variable=[i for i,s in enumerate(LABELS) if s!=anchor['slot']]
        lam=.9 if diverse else 1.
        while remaining and len(selected)<5:
            def objective(i):
                overlap=max((sum(triples[i][k]==triples[j][k] for k in variable)/len(variable) for j in selected),default=0)
                return lam*float(relevance[i])-(1-lam)*overlap
            pick=max(remaining,key=objective);selected.append(pick);remaining.remove(pick)
        results=[]
        for rank,j in enumerate(selected,1):
            rule=matched[j]
            reasons=['Giữ lại món đồ bạn đã chọn.','Được xếp hạng bằng đặc trưng ảnh của cả bộ.']
            if rule:
                names=[CATEGORIES.get(t.split(':',1)[1],t.split(':',1)[1]) for t in rule['antecedents']+rule['consequents']]
                reasons.append('Luật kết hợp được sử dụng: '+' + '.join(names)+'.')
            else:reasons.append('Không có luật kết hợp phù hợp; sử dụng điểm mô hình.')
            if diverse:reasons.append('Ưu tiên giảm lặp các món còn lại giữa những gợi ý.')
            results.append({'id':'-'.join(triples[j]),'rank':rank,'items':[self.item(i) for i in triples[j]],'reasons':reasons})
        return {'anchor':self.item(iid),'outfits':results,'candidate_count':len(triples)}


class Handler(BaseHTTPRequestHandler):
    def send(self,status,content,content_type='application/json; charset=utf-8'):
        if not isinstance(content,bytes):content=json.dumps(content,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(content)));self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(content)

    def do_GET(self):
        parsed=urlparse(self.path);q=parse_qs(parsed.query);path=parsed.path
        try:
            if path=='/api/catalog':
                return self.send(200,self.server.engine.catalog(q.get('slot',['top'])[0],q.get('category',[''])[0],q.get('q',[''])[0],int(q.get('page',['1'])[0])))
            if path=='/api/health':return self.send(200,{'status':'ready','items':len(self.server.engine.items),'model':'MLP + FP-Growth'})
            if path.startswith('/image/') and path.endswith('.jpg'):
                iid=path[len('/image/'):-4]
                if iid not in self.server.engine.items:return self.send(404,{'error':'Không tìm thấy ảnh.'})
                file=ROOT/self.server.engine.items[iid]['image_path']
                if not file.resolve().is_relative_to((ROOT/'data').resolve()):return self.send(404,{'error':'Không tìm thấy ảnh.'})
                return self.send(200,file.read_bytes(),'image/jpeg')
            allowed={'/':'index.html','/app.js':'app.js','/styles.css':'styles.css'}
            if path in allowed:
                file=ROOT/'web/static'/allowed[path]
                return self.send(200,file.read_bytes(),(mimetypes.guess_type(str(file))[0] or 'text/plain')+'; charset=utf-8')
            return self.send(404,{'error':'Không tìm thấy trang.'})
        except ValueError as e:return self.send(400,{'error':str(e)})
        except Exception:
            import traceback;traceback.print_exc()
            return self.send(500,{'error':'Không thể đọc dữ liệu. Vui lòng thử lại.'})

    def do_POST(self):
        if self.path!='/api/recommend':return self.send(404,{'error':'Không tìm thấy chức năng.'})
        try:
            if not self.headers.get('Content-Type','').startswith('application/json'):raise ValueError('Yêu cầu phải có định dạng JSON.')
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>16384:raise ValueError('Yêu cầu không hợp lệ.')
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict) or not isinstance(data.get('id'),str):raise ValueError('Chọn một món đồ trước khi gợi ý.')
            return self.send(200,self.server.engine.recommend(data['id'],data.get('diverse') is True))
        except (ValueError,TypeError) as e:return self.send(400,{'error':str(e)})
        except Exception:
            import traceback;traceback.print_exc()
            return self.send(500,{'error':'Không thể tạo gợi ý. Vui lòng thử lại.'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    print('Đang nạp mô hình và danh mục...',flush=True)
    engine=Recommender()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.engine=engine
    print(f'Fashion Suggest sẵn sàng: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
