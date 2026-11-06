"""Meaningful integration checks against real catalog/checkpoint; no retraining."""
import json
from pathlib import Path
import sys
import unittest
from io import BytesIO

sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import Recommender, Handler, ROOT


class AppChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.engine=Recommender()

    def test_catalog_filters_and_empty_result(self):
        for slot in ('top','bottom','shoes'):
            page=self.engine.catalog(slot)
            self.assertEqual(len(page['items']),12)
            self.assertTrue(all(i['slot']==slot for i in page['items']))
            category=page['items'][0]['category']
            self.assertTrue(all(i['category']==category for i in self.engine.catalog(slot,category)['items']))
        self.assertEqual(self.engine.catalog(query='__definitely_no_item__')['total'],0)
        self.assertEqual(self.engine.catalog(page=-2)['page'],1)

    def test_real_recommendations_and_images(self):
        for slot in ('top','bottom','shoes'):
            anchor=self.engine.catalog(slot)['items'][0]
            result=self.engine.recommend(anchor['id'])
            self.assertEqual(len(result['outfits']),5)
            self.assertLessEqual(result['candidate_count'],2304)
            self.assertEqual(len(set(o['id'] for o in result['outfits'])),5)
            for outfit in result['outfits']:
                self.assertEqual([i['slot'] for i in outfit['items']],['top','bottom','shoes'])
                self.assertIn(anchor['id'],[i['id'] for i in outfit['items']])
                self.assertGreaterEqual(len(outfit['reasons']),3)
                for i in outfit['items']:
                    path=ROOT/self.engine.items[i['id']]['image_path']
                    self.assertGreater(path.stat().st_size,0)
            alternate=self.engine.recommend(anchor['id'],True)
            self.assertEqual(len(alternate['outfits']),5)
        with self.assertRaises(ValueError):self.engine.recommend('../secrets')

    def test_http_routes_without_external_socket(self):
        class Connection:
            def __init__(self,data):self.input=BytesIO(data);self.output=BytesIO()
            def makefile(self,*args,**kwargs):return self.input
            def sendall(self,data):self.output.write(data)
        class Server:pass
        server=Server();server.engine=self.engine
        def request(method,path,payload=None,content_type='application/json'):
            body=json.dumps(payload).encode() if payload is not None else b''
            raw=f'{method} {path} HTTP/1.0\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body
            c=Connection(raw);Handler(c,('127.0.0.1',0),server)
            header,body=c.output.getvalue().split(b'\r\n\r\n',1)
            return header,body
        for path in ('/','/app.js','/styles.css','/api/health','/api/catalog?slot=top'):
            header,body=request('GET',path);self.assertIn(b'200 OK',header);self.assertTrue(body)
        for path in ('/../../.env','/image/../../.env.jpg'):
            self.assertIn(b'404',request('GET',path)[0])
        self.assertIn(b'400',request('POST','/api/recommend',{'id':'bad'})[0])
        self.assertIn(b'400',request('POST','/api/recommend',[],content_type='text/plain')[0])
        anchor=self.engine.catalog()['items'][0]['id']
        header,body=request('POST','/api/recommend',{'id':anchor})
        self.assertIn(b'200',header);self.assertEqual(len(json.loads(body)['outfits']),5)


if __name__=='__main__':unittest.main(verbosity=2)
