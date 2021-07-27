from src import measures
from src.cleaner import DoubleStepCleaner
from src.separator import Separator
from src.errors import CleanerError, MeasureError, FileError, MeasurementError

from tinydb import TinyDB, Query
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


is_forced = lambda c: ('force' in c.keys()) and (c['force'] == True)

class Measurer:
    def __init__(self, config, db_addr=None, write_db=True):
        self.cleaner = DoubleStepCleaner(**config['cleaning'])
        self.measurement_config = config['measures']
        self.centering_config = config['centering'] if 'centering' in config.keys() else []
        self.force = is_forced(config)
        self.group_name = config['group_name']
        self.write_db = write_db

        if db_addr is not None:
            self.db = db_addr
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

        with TinyDB(self.db) as db:
            record = db.search(Query().id == meta['id'])
        if record: # check if there is record for the current fish in db
            record = record[0]
        else:
            return self.measurement_config, self.centering_config, deepcopy(meta)

        if self.group_name in record.keys(): # check if this organ group was measured already
            old_measurements = record[self.group_name]
        else:
            return self.measurement_config, self.centering_config, deepcopy(meta)

        # since we have this group already measured, let start searching for
        # possible duplicates.
        cu = [] # centers used
        mc = [] # current measurement config
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
                nl = []
                for measure in label['measures']:
                    if not (measure['function'] in com.keys()):
                        # measure is absent
                        nl.append(measure)
                        cu.append(cc)
                    elif is_forced(measure):
                        # measure is forced
                        nl.append(measure)
                        cu.append(cc)
                if nl:
                    mc.append(nl)

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

        if not mc:
            return om, warnings

        om = self._prepare_update_record(mc, om)
        
        
        mask, roi = self.cleaner(mask)
        volume = volume[roi]

        centers = {}
        for centering in cc:
            centers[centering['name']] = Separator((mask == centering['label_id']), centering['function'], centering['count'])
        for label in mc:
            label_mask = (mask == label['id'])
            for measure in label['measures']:
                mf = getattr(measures, measure['function'])
                cc = centers[label['center']] if ('center' in label.keys()) else None
                try:
                    om[self.group_name][label['name']][measure['function']] = mf(label_mask, volume, cc)
                except MeasurementError as err:
                    warnings.append(err._as_dict())
        return om, warnings

    def _load_n_process(self, mask_addr, volume_addr, one_fish):
        log = {'mask_addr': mask_addr, 'volume_addr': volume_addr, 'id': one_fish['id'], 'status': 'initiated'}
        try:
            mask, volume = self._load_files(mask_addr, volume_addr)
            measured, warnings = self._process_one_volume(mask, volume, one_fish)

            if self.write_db:
                with TinyDB(self.db) as db:
                    db.upsert(measured, Query().id == measured['id'])
            if warnings:
                log['status'] = 'warnings'
                log['warnings'] = warnings
            else:
                log['status'] = 'success'
        except (CleanerError, FileError) as err:
            log['status'] = 'error'
            log['error'] = err._as_dict()
        return log

    def _from_files(self, mask_dir, volumes_dir, meta_file, processing_range=None, multiprocessing=None):
        if processing_range is None:
            processing_range = slice(None)

        meta_df = pd.read_csv(meta_file).iloc[processing_range]
        results = []
        internal_errors = []
        external_errors = []

        if multiprocessing is None:
            for i, one_fish in tqdm(meta_df.iterrows(), desc='processing volumes', total=len(meta_df)):
                result, status = self._load_n_process(mask_dir, volumes_dir, one_fish)
                if status == 2:
                    print(status)
                    raise Exception('Halting computation!')
                results_processing(result, status)
        else:
            res_pairs = Parallel(n_jobs=multiprocessing, verbose=20)(delayed(self._load_n_process)(mask_dir, volumes_dir, one_fish) for i, one_fish in meta_df.iterrows())
            for result, status in res_pairs:
                results_processing(result, status)

        return results, (internal_errors, external_errors)


@hydra.main(config_path='measurement_configs', config_name="config")
def measure(cfg : DictConfig) -> None:
    mc = cfg['measurement']
    dc = cfg['dataset']
    pc = cfg['processing']

    readonly = pc.get('readonly', False)
    db = pc.get('db', None)
    measurer = Measurer(mc, db_addr=db, write_db=(not readonly))
    
    processing_triplets = []
    for directory in glob(dc['directories']):
        volume_addr = os.path.join(directory, dc['volume_name'])
        mask_addr = os.path.join(directory, dc['mask_name'])
        sample_id = {'id': re.findall(dc['id_regexp'], directory)[0]}

        processing_triplets.append((mask_addr, volume_addr, sample_id))
    measurer._load_n_process(*processing_triplets[0])
    
    log = Parallel(n_jobs=pc['n_jobs'], verbose=20)(delayed(measurer._load_n_process)(*i) for i in processing_triplets)
    # log = [measurer._load_n_process(*i) for i in processing_triplets]

    with open('log.yaml', 'w') as f:
        yaml.safe_dump(log, f)


if __name__ == "__main__":
    measure()
