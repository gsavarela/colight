''' Simpler pipeline script

    1. Preloads a model. (optional: EMIT)
    2. Generates data.
    3. Update network. (optional: not EMIT)

    Set rollouts:
    1. EMIT = True
    2. LIST_MODEL_NEED_TO_UPDATE = []

'''
import json
import os
import shutil
from multiprocessing import Process, Pool
import pickle
import sys
import time
import traceback
from pathlib import Path

from generator import Generator
from construct_sample import ConstructSample
from updater import Updater

class Pipeline:

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            if self.dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.test = self.dic_traffic_env_conf['EMIT'] and \
            not any(self.dic_exp_conf['LIST_MODEL_NEED_TO_UPDATE'])

        # do file operations
        if not self.test:
            self._path_check()
            self._copy_conf_file()
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file()


    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          best_round=None):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            best_round=best_round,
            bar_round=bar_round
        )

        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")
        return

    def downsample(self, path_to_log, i):

        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                os.remove(path_to_pkl)
                with open(path_to_pkl, "wb") as f_subset:
                    try:
                        pickle.dump(subset_data, f_subset)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
                        print("----------------------------")

            except Exception as e:
                # print("CANNOT READ %s"%path_to_pkl)
                print("----------------------------")
                print("Error occurs when READING pickles when down sampling for inter {0}, {1}".format(i, f_logging_data))
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                print("----------------------------")


    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.downsample(path_to_log, i)

    def construct_sample_multi_process(self, train_round, cnt_round, batch_size=200):
        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        if batch_size > self.dic_traffic_env_conf['NUM_INTERSECTIONS']:
            batch_size_run = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'], batch_size_run):
            start = batch
            stop = min(batch + batch_size, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])
            process_list.append(Process(target=self.construct_sample_batch, args=(cs, start, stop)))

        for t in process_list:
            t.start()
        for t in process_list:
            t.join()

    def construct_sample_batch(self, cs, start,stop):
        for inter_id in range(start, stop):
            print("make construct_sample_wrapper for ", inter_id)
            cs.make_reward(inter_id)

    def run(self, multi_process=False):

        best_round, bar_round = None, None

        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        if self.test:
            # Fetch the latest model defines best_round
            round_paths = Path(self.dic_path['PATH_TO_MODEL']).rglob('round_*')
            rounds = [int(p.stem.split('_')[1]) for p in round_paths]
            best_round = max(rounds)

        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                      self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                )
                    print("before p")
                    p.start()
                    print("end p")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    self.generator_wrapper(cnt_round=cnt_round,
                                           cnt_gen=cnt_gen,
                                           dic_path=self.dic_path,
                                           dic_exp_conf=self.dic_exp_conf,
                                           dic_agent_conf=self.dic_agent_conf,
                                           dic_traffic_env_conf=self.dic_traffic_env_conf,
                                           best_round=best_round)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time
            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()

            round_dir = 'test_round' if self.test else 'train_round'
            round_dir = Path(self.dic_path["PATH_TO_WORK_DIRECTORY"]) / round_dir
            round_dir.mkdir(exist_ok=True)

            cs = ConstructSample(path_to_samples=round_dir.as_posix(), cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()


            # EvaluateSample()
            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            print("==============  update network =============")
            update_network_start_time = time.time()
            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round,
                                      bar_round))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round,
                                         bar_round=bar_round)


            round_dir = 'test_round' if self.test else 'train_round'
            round_dir = Path(self.dic_path["PATH_TO_WORK_DIRECTORY"]) / round_dir
            for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                path_to_log = round_dir / f"round_{cnt_round}" / f"generator_{cnt_gen}"
                try:
                    self.downsample_for_system(path_to_log.as_posix(),self.dic_traffic_env_conf)
                except Exception as e:
                    print("----------------------------")
                    print("Error occurs when downsampling for round {0} generator {1}".format(cnt_round, cnt_gen))
                    print("traceback.format_exc():\n%s"%traceback.format_exc())
                    print("----------------------------")
            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            print("best_round: ", best_round)

            print("Generator time: ",generator_total_time)
            print("Making samples time:", making_samples_total_time)
            print("update_network time:", update_network_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")
            f_time.write("{0}\t{1}\t{2}\t{3}\n".format(generator_total_time,making_samples_total_time,
                                                          update_network_total_time,
                                                          time.time()-round_start_time))
            f_time.close()


