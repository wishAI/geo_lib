import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from webgui import avp_recordings
from webgui.tests import test_avp_delivery as delivery


def clip():
    return {'format':'avp-motion-clip', 'version':1, 'createdAt':'2026-11-01T12:00:00Z',
            'name':'captured on headset', 'urdfSha256':'a'*64,
            'frames':[{'t':0, 'pose':{'hand_l':0}, 'tracking':None, 'rawTracking':None, 'alignment':None}]}


class RecordingTests(unittest.TestCase):
    def test_immutable_deduplicated_recordings_survive_new_reader(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(avp_recordings,'ROOT',Path(directory)):
            value=clip();first=avp_recordings.save(value)
            self.assertEqual(avp_recordings.save(value),first)
            self.assertEqual(avp_recordings.read(first['id']),value)
            other=copy.deepcopy(value);other['frames'][0]['pose']['hand_l']=.5
            second=avp_recordings.save(other)
            self.assertNotEqual(first['id'],second['id'])
            self.assertEqual(len(avp_recordings.summaries()),2)
            self.assertEqual(avp_recordings.read(first['id']),value)
            self.assertTrue((Path(directory)/(first['id']+'.json.gz')).is_file())

    def test_rejects_bad_timestamps_matrices_and_identifiers(self):
        for mutate in [lambda c:c['frames'][0].update(t=-1),lambda c:c['frames'][0]['pose'].update(hand_l=float('nan')),
                       lambda c:c['frames'][0].update(tracking={'head':[[0]*4]*4}),lambda c:c.update(frames=[]),lambda c:c.update(urdfSha256='../private')]:
            value=clip();mutate(value)
            with self.assertRaises(ValueError):avp_recordings.validate(value)
        with self.assertRaises(ValueError):avp_recordings.read('../private')


class RecordingHttpTests(unittest.TestCase):
    setUp = delivery.AvpDeliveryTests.setUp
    tearDown = delivery.AvpDeliveryTests.tearDown
    request = delivery.AvpDeliveryTests.request
    def test_recording_download_uses_stored_gzip_and_private_immutable_cache(self):
        with patch.object(avp_recordings,'ROOT',self.root/'recordings'):
            value=clip();saved=avp_recordings.save(value)
            path='/api/avp/recordings/'+saved['id']
            # Download must not parse and re-serialize a potentially 96 MB take.
            with patch.object(avp_recordings,'read',side_effect=AssertionError('unnecessary JSON expansion')):
                status,headers,body=self.request(path=path,headers={'Accept-Encoding':'gzip'})
            self.assertEqual(status,200)
            self.assertEqual(headers['Content-Encoding'],'gzip')
            self.assertEqual(body,(avp_recordings.ROOT/(saved['id']+'.json.gz')).read_bytes())
            self.assertEqual(json.loads(gzip.decompress(body)),value)
            self.assertEqual(int(headers['X-Uncompressed-Content-Length']),len(gzip.decompress(body)))
            self.assertIn('private',headers['Cache-Control'])
            self.assertIn('immutable',headers['Cache-Control'])
            self.assertEqual(headers['Vary'],'Accept-Encoding')
            status,_,body=self.request(path=path,headers={'Accept-Encoding':'gzip','If-None-Match':headers['ETag']})
            self.assertEqual((status,body),(304,b''))
            status,headers,body=self.request(path=path,headers={'Accept-Encoding':'gzip;q=0'})
            self.assertEqual(status,200)
            self.assertNotIn('Content-Encoding',headers)
            self.assertEqual(json.loads(body),value)
            self.assertEqual(int(headers['X-Uncompressed-Content-Length']),len(body))

    def test_cross_client_upload_list_read_and_foreign_origin_rejection(self):
        with patch.object(avp_recordings,'ROOT',self.root/'recordings'):
            headers={'Content-Type':'application/json','Origin':'http://'+self.host}
            status,_,body=self.request('POST','/api/avp/recordings',headers,json.dumps(clip()))
            self.assertEqual(status,200);saved=json.loads(body)
            status,_,body=self.request(path='/api/avp/recordings')
            self.assertEqual(status,200);self.assertEqual(json.loads(body)['recordings'][0]['id'],saved['id'])
            status,_,body=self.request(path='/api/avp/recordings/'+saved['id'])
            self.assertEqual(status,200);self.assertEqual(json.loads(body),clip())
            status,_,_=self.request('POST','/api/avp/recordings',dict(headers,Origin='https://foreign.invalid'),json.dumps(clip()))
            self.assertEqual(status,400)
