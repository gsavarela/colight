from pathlib import Path
from copy import deepcopy
import json
import os
import shutil
import xml.etree.ElementTree as ET
from roller import Roller
from construct_sample import ConstructSample
from multiprocessing import Process, Pool, Manager
import random
import pickle
import model_test
import pandas as pd
import numpy as np
from math import isnan
import sys
import time
import traceback

class RolloutJob:

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
        json.dump(dict(self.dic_traffic_env_conf),
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
        self.dic_traffic_env_conf = Manager().dict(dic_traffic_env_conf) #DicProxy (SharedVariable)
        self.dic_path = dic_path

        # do file operations
        # self._path_check()
        # self._copy_conf_file()
        # self._copy_anon_file()
        # test_duration
        self.test_duration = []

        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)


    def early_stopping(self, dic_path, cnt_round): # Todo multi-process
        print("decide whether to stop")
        early_stopping_start_time = time.time()
        record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_"+str(cnt_round))

        ave_duration_all = []
        # compute duration
        for inter_id in self.sample_inter_id:
            try:
                df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_{0}.csv".format(inter_id)),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])
                duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                ave_duration = np.mean([time for time in duration if not isnan(time)])
                ave_duration_all.append(ave_duration)
            except FileNotFoundError:
                error_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
                if not os.path.exists(error_dir):
                    os.makedirs(error_dir)
                f = open(os.path.join(error_dir, "error_info.txt"), "a")
                f.write("Fail to read csv of inter {0} in early stopping of round {1}\n".format(inter_id, cnt_round))
                f.close()
                pass

        ave_duration = np.mean(ave_duration_all)
        self.test_duration.append(ave_duration)
        early_stopping_end_time = time.time()
        print("early_stopping time: {0}".format(early_stopping_end_time - early_stopping_start_time) )
        if len(self.test_duration) < 30:
            return 0
        else:
            duration_under_exam = np.array(self.test_duration[-15:])
            mean_duration = np.mean(duration_under_exam)
            std_duration = np.std(duration_under_exam)
            max_duration = np.max(duration_under_exam)
            if std_duration/mean_duration < 0.1 and max_duration < 1.5 * mean_duration:
                return 1
            else:
                return 0



    def roller_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf,
                          dic_agent_conf, dic_traffic_env_conf, best_round=None):

        roller = Roller(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )
        print("make roller")
        roller.roll()
        print("roller_wrapper end")
        return

    def downsample(self, path_to_log, i):

        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                print(subset_data)
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


    def run(self, multi_process=False):

        best_round, bar_round = None, None
        path_to_work = Path(self.dic_path["PATH_TO_WORK_DIRECTORY"]) / "running_time.csv"

        with open(path_to_work,"w") as f:
            f.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")

        self.dic_exp_conf["PRETRAIN"] = False
        self.dic_exp_conf["AGGREGATE"] = False

        root_path = Path(self.dic_path["PATH_TO_MODEL"])
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()
            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    # For each generator -- fetch a path.
                    p = Process(
                        target=self.roller_wrapper,
                        args=(cnt_round, cnt_gen, self.dic_path,
                              self.dic_exp_conf, self.dic_agent_conf,
                              self.dic_traffic_env_conf, best_round)
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
                    # For each generator -- fetch a path.
                    self.roller_wrapper(
                        cnt_round=cnt_round,
                        cnt_gen=cnt_gen,
                        dic_path=self.dic_path,
                        dic_exp_conf=self.dic_exp_conf,
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        best_round=best_round
                    )


            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time

            print("Generator time: ",generator_total_time)
            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")
            f_time.write("{0}\n".format(generator_total_time))
            f_time.close()
