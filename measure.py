from src import measures
from src.cleaner import DoubleStepCleaner
from src.separator import Separator
from src.gallery import *
from src.errors import CleanerError, MeasureError, FileError, MeasurementError, SeparationError

import numpy as np
from tinydb import TinyDB, Query
from pymongo import MongoClient
from copy import deepcopy
import pandas as pd
import os
from medpy.io import load as medload
import tifffile
from tqdm.auto import tqdm
import yaml
import argparse

import hydra
from omegaconf import DictConfig
from glob import glob
import re

from joblib import delayed, Parallel
import warnings as wrngngs
from collections import defaultdict



is_forced = lambda c: ('force' in c.keys()) and (c['force'] == True)

def dict_to_planar(d):
    n_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_to_planar(v)
            for sk, sv in v.items():
                n_d['.'.join([k, sk])] = sv
        else:
            n_d[k] = v
    return n_d

class MongoDBInterface:
    def __init__(self, address, database):
        self.db = MongoClient(address)[database].measurements
    
    def get_sample_record(self, id):
        return self.db.find_one({'id': id}, projection={'_id': False})
    
    def update_sample_record(self, record):
        self.db.find_one_and_update({'id': record['id']},
                                    {'$set': dict_to_planar(record)},
                                    upsert=True)
    
    def as_table(self, fields, request=None):
        if isinstance(fields, str):
            fields = [fields]
        
        collection = self.db.find(request, projection={**{f_n: True for f_n in fields}, **{'_id': False}})
        collection = [dict_to_planar(document) for document in collection] # flattened dicts in collection
        # collection = {k: [d[k] for d in collection] for k in collection[0].keys()} # collection as dict of lists instead of list of dicts
        return collection
    
class TinyDBInterface:
    def __init__(self, address):
        self.db = address

    def get_sample_record(self, id):
        with TinyDB(self.db) as db:
            record = db.search(Query().id == id)
        if record:
            return record[0]
        else:
            return None
    
    def update_sample_record(self, record):
        with TinyDB(self.db) as db:
            db.upsert(record, Query().id == record['id'])
    
    def as_table(self, fields, request=None):
        if isinstance(fields, str):
            fields = [fields]
        with TinyDB(self.db) as db:
            if request is not None:
                selection = db.search(Query().fragment(request))
            else:
                selection = db.all()
            
        result = defaultdict(list)
        for record in selection:
            for fieldname in fields:
                subfields = fieldname.split('.')
                curview = record
                for subfieldname in subfields:
                    if subfieldname in curview.keys():
                        curview = curview[subfieldname]
                    else:
                        curview = None
                        break
                result[fieldname].append(curview)
        return result

def get_database_interface(address):
    if ('@' in address): # this is MongoDB
        database, address = address.split('@')
        return  MongoDBInterface(address, database)
    else: # this is TinyDB
        return TinyDBInterface(address)


class Measurer:
    def __init__(self, config, db_addr=None, write_db=True, log_file=None, gallery=None):
        self.cleaner = DoubleStepCleaner(**config['cleaning'])
        self.measurement_config = config['measures']
        self.centering_config = config['centering'] if 'centering' in config.keys() else []
        self.force = is_forced(config)
        self.group_name = config['group_name']
        self.gallery = gallery

        self.log_file = log_file
        if self.log_file is not None:
            with open(self.log_file, 'w') as f:
                pass
        
        self.write_db = write_db
        if db_addr is not None:
            self.db = get_database_interface(db_addr)
        else:
            self.db = None
            self.write_db = False

    def _prepare_update_record(self, mc, om):
        if not (self.group_name in om.keys()):
            om[self.group_name] = dict()
        for label in mc:
            if not(label['name'] in om[self.group_name].keys()):
                om[self.group_name][label['name']] = dict()

        return om

    def _filter_configs(self, meta):
        if self.force or (self.db is None): # check if everything is forced to be processed
            return self.measurement_config, self.centering_config, deepcopy(meta)

        record = self.db.get_sample_record(meta['id'])
        if record is None: # there is no such record appaerntly
            return self.measurement_config, self.centering_config, deepcopy(meta)

        if self.group_name in record.keys(): # check if this organ group was measured already
            old_measurements = record[self.group_name]
        else:
            return self.measurement_config, self.centering_config, deepcopy(meta)

        # since we have this group already measured, let start searching for
        # possible duplicates.
        cu = [] # centers used
        mc = [] # current measurement config
        # breakpoint()
        for label in self.measurement_config:
            cc = label['center'] if ('center' in label.keys()) else None
            if not (label['name'] in old_measurements.keys()): # everything for this label should be measured
                mc.append(label)
                cu.append(cc)
            elif is_forced(label):
                # everything for this label forced to be measured
                mc.append(label)
                cu.append(cc)
            else:
                # selecting specific measures
                com = old_measurements[label['name']]
                nl = deepcopy(label)
                nl['measures'] = []
                for measure in label['measures']:
                    if not (measure['function'] in com.keys()):
                        # measure is absent
                        nl['measures'].append(measure)
                    elif is_forced(measure):
                        # measure is forced
                        nl['measures'].append(measure)
                if nl['measures']:
                    mc.append(nl)
                    cu.append(cc)

        cu = set(cu) - {None}
        cc = [c for c in self.centering_config if (c['name'] in cu)]
        return mc, cc, record

    def _load_files(self, mask_addr, volume_addr):
        if not os.path.exists(mask_addr):
            raise FileError('mask file does not exists', addr=mask_addr)

        if not os.path.exists(volume_addr):
            raise FileError('volume file does not exists', addr=volume_addr)

        mask = tifffile.imread(mask_addr)
        volume = tifffile.imread(volume_addr)

        return mask, volume

    def _process_one_volume(self, mask, volume, meta):
        mc, cc, om = self._filter_configs(meta) # measurement config, centering config, old measurements (already was in database)
        warnings = []
        sample_id = meta['id']

        if not mc:
            return om, warnings

        om = self._prepare_update_record(mc, om)
        
        mask, slices, bbox = self.cleaner(mask)
        volume = volume[slices]

        group_bbox = np.array(bbox).astype(int).tolist()
        om['group_bbox'] = group_bbox


        centers = {}
        for centering in cc:
            centers[centering['name']] = Separator((mask == centering['label_id']), centering['function'], centering['count'])
        for label in mc:
            label_mask = (mask == label['id'])

            for measure in label['measures']:
                mf = getattr(measures, measure['function'])
                cc = centers[label['center']] if ('center' in label.keys()) else None
                try:
                    with wrngngs.catch_warnings(record=True) as w:
                        om[self.group_name][label['name']][measure['function']] = mf(label_mask, volume, cc)

                    if len(w) > 0:
                        # breakpoint()
                        warnings.append({'location': w[0].filename + '::' + str(w[0].lineno), 'meta': None, 'message': str(w[0].message), 'type': str(w[0].category)})
                        # warnings += [{'location': 'out', 'meta': None, 'message': str(i), 'type': 'warning'} for i in w]
                except MeasurementError as err:
                    warnings.append(err._as_dict())

        if self.gallery:
            gallery_params = {
                'gallery_path': self.gallery.path,
                'keep_every_slice': self.gallery.keep_every_slice, 
                'sample_id': sample_id, 
                'slices_range': group_bbox,
                'scale_small': self.gallery.scale_small,
                'scale_large': self.gallery.scale_large,
                'blend_alpha': self.gallery.blend_alpha
            }
            
            make_gallery(volume, mask, gallery_params)

        return om, warnings

    def _load_n_process(self, mask_addr, volume_addr, one_fish):
        log = {'mask_addr': mask_addr, 'volume_addr': volume_addr, 'id': one_fish['id'], 'status': 'initiated'}
        try:
            mask, volume = self._load_files(mask_addr, volume_addr)
            measured, warnings = self._process_one_volume(mask, volume, one_fish)

            if self.write_db:
                self.db.update_sample_record(measured)
            if warnings:
                log['status'] = 'warnings'
                log['warnings'] = warnings
            else:
                log['status'] = 'success'
        except (CleanerError, FileError, SeparationError) as err:
            log['status'] = 'error'
            log['error'] = err._as_dict()
        
        if self.log_file is not None:
            with open(self.log_file, 'r+') as f:
                previous_logs = yaml.safe_load(f) or []
                previous_logs.append(log)
                f.seek(0)
                yaml.safe_dump(previous_logs, f)

    # def _from_files(self, mask_dir, volumes_dir, meta_file, processing_range=None, multiprocessing=None):
    #     if processing_range is None:
    #         processing_range = slice(None)

    #     meta_df = pd.read_csv(meta_file).iloc[processing_range]
    #     results = []
    #     internal_errors = []
    #     external_errors = []

    #     if multiprocessing is None:
    #         for i, one_fish in tqdm(meta_df.iterrows(), desc='processing volumes', total=len(meta_df)):
    #             result, status = self._load_n_process(mask_dir, volumes_dir, one_fish)
    #             if status == 2:
    #                 print(status)
    #                 raise Exception('Halting computation!')
    #             results_processing(result, status)
    #     else:
    #         res_pairs = Parallel(n_jobs=multiprocessing, verbose=20)(delayed(self._load_n_process)(mask_dir, volumes_dir, one_fish) for i, one_fish in meta_df.iterrows())
    #         for result, status in res_pairs:
    #             results_processing(result, status)

    #     return results, (internal_errors, external_errors)


def measure_file(measurer_params, triplet):
    Measurer(**measurer_params)._load_n_process(*triplet)

@hydra.main(config_path='measurement_configs', config_name="config")
def measure(cfg : DictConfig) -> None:
    mc = cfg['measurement']
    dc = cfg['dataset']
    pc = cfg['processing']

    readonly = pc.get('readonly', False)
    db = pc.get('db', None)
    gal = pc.get('gallery', None)

    #measurer_params = {'config': mc, 'db_addr': db, 'write_db': (not readonly), 'log_file': 'log.yaml', 'gallery_params': gal}
    measurer = Measurer(mc, db_addr=db, write_db=(not readonly), log_file='log.yaml', gallery = gal)
    
    processing_triplets = []
    for directory in glob(dc['directories']):
        volume_addr = os.path.join(directory, dc['volume_name'])
        mask_addr = os.path.join(directory, dc['mask_name'])
        sample_id = {'id': re.findall(dc['id_regexp'], directory)[0]}

        processing_triplets.append((mask_addr, volume_addr, sample_id))
    
    #Parallel(n_jobs=pc['n_jobs'], verbose=20)(delayed(measure_file)(measurer_params, triplet) for triplet in processing_triplets)
    [measurer._load_n_process(*i) for i in tqdm(processing_triplets)]

    if measurer.gallery:
        print('Building gallery HTML')
        # TODO



if __name__ == "__main__":
    measure()
