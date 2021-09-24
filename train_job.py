import json
import os
import shutil
import xml.etree.ElementTree as ET
from trainer import Trainer
# from construct_sample import ConstructSample
# from updater import Updater
from multiprocessing import Process, Pool, Manager
from model_pool import ModelPool
import random
import pickle
import model_test
import pandas as pd
import numpy as np
from math import isnan
import sys
import time
import traceback

class TrainJob:
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
        self._path_check()
        self._copy_conf_file()
        self._copy_anon_file()
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



    def trainer_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          best_round=None):
        trainer = Trainer(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )
        print("make trainer")
        trainer.train()
        print("trainer_wrapper end")
        return

    def model_pool_wrapper(self, dic_path, dic_exp_conf, cnt_round):
        model_pool = ModelPool(dic_path, dic_exp_conf)
        model_pool.model_compare(cnt_round)
        model_pool.dump_model_pool()


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

        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"w")
        f_time.write("trainer_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        # trainf
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            process_list = []

            print("==============  trainer =============")
            trainer_start_time = time.time()
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    p = Process(target=self.trainer_wrapper,
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
                    print("trainer %d to join" % i)
                    p.join()
                    print("trainer %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    self.trainer_wrapper(
                        cnt_round=cnt_round,
                        cnt_gen=cnt_gen,
                        dic_path=self.dic_path,
                        dic_exp_conf=self.dic_exp_conf,
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        best_round=best_round
                    )
            trainer_end_time = time.time()
            trainer_total_time = trainer_end_time - trainer_start_time

            if not self.dic_exp_conf["DEBUG"]:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                    try:
                        self.downsample_for_system(path_to_log,self.dic_traffic_env_conf)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when downsampling for round {0} trainer {1}".format(cnt_round, cnt_gen))
                        print("traceback.format_exc():\n%s"%traceback.format_exc())
                        print("----------------------------")
            # update_network_end_time = time.time()
            # update_network_total_time = update_network_end_time - update_network_start_time

            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            if multi_process:
                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False))
                p.start()
                if self.dic_exp_conf["EARLY_STOP"]:
                    p.join()
            else:
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, if_gui=False)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            print('==============  early stopping =============')
            if self.dic_exp_conf["EARLY_STOP"]:
                flag = self.early_stopping(self.dic_path, cnt_round)
                if flag == 1:
                    print("early stopping!")
                    print("training ends at round %s" % cnt_round)
                    break

            print('==============  model pool evaluation =============')
            if self.dic_exp_conf["MODEL_POOL"] and cnt_round > 50:
                if multi_process:
                    p = Process(target=self.model_pool_wrapper,
                                args=(self.dic_path,
                                      self.dic_exp_conf,
                                      cnt_round),
                                )
                    p.start()
                    print("model_pool to join")
                    p.join()
                    print("model_pool finish join")
                else:
                    self.model_pool_wrapper(dic_path=self.dic_path,
                                            dic_exp_conf=self.dic_exp_conf,
                                            cnt_round=cnt_round)
                model_pool_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl")
                if os.path.exists(model_pool_dir):
                    model_pool = pickle.load(open(model_pool_dir, "rb"))
                    ind = random.randint(0, len(model_pool) - 1)
                    best_round = model_pool[ind][0]
                    ind_bar = random.randint(0, len(model_pool) - 1)
                    flag = 0
                    while ind_bar == ind and flag < 10:
                        ind_bar = random.randint(0, len(model_pool) - 1)
                        flag += 1
                    # bar_round = model_pool[ind_bar][0]
                    bar_round = None
                else:
                    best_round = None
                    bar_round = None

                # downsample
                if not self.dic_exp_conf["DEBUG"]:
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                               "round_" + str(cnt_round))
                    self.downsample_for_system(path_to_log, self.dic_traffic_env_conf)
            else:
                best_round = None

            print("best_round: ", best_round)

            print("Trainer time: ",trainer_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")
            f_time.write("{0}\t{1}\t{2}\n".format(trainer_total_time,
                                                          test_evaluation_total_time,
                                                          time.time()-round_start_time))
            f_time.close()


