''' Converts ATSC data into experience.
    Based of ConstructSamples
    * Singleton
    * Experience serialization/deserialization
    * Persists data between different rounds.
'''
from pathlib import Path
from multiprocessing import Pool, Process
import os
import traceback

import time
import numpy as np
import pickle
import pandas as pd
# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]

class Sampler(object):

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf, cnt_gen=None):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.logging_data_list_per_gen = []
        self.hidden_states_list = None
        self._sample = [[] for _ in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS'])]
        self.cnt_gen = cnt_gen



    def get_samples(self, i):
        '''Gets both previous and current samples.'''
        # lvl 1: num_intersections
        # lvl 2: sample_size
        return self._sample[i]

    def sample(self):
        '''Samples current episode data and dumps both current and prv.'''
        print("==============  make samples =============")
        # make samples and determine which samples are good
        making_samples_start_time = time.time()

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            target_dir = f'generator_{self.cnt_gen}'
            # loads current data.
            self.load_data_for_system(target_dir)

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            # builds sample
            self.make_reward(target_dir, i)
            target_path = f'{self.path_to_samples}/{target_dir}/samples_inter_{i}.pkl'
            with open(target_path, 'wb') as f:
                pickle.dump(self.get_samples(i), f, -1)
                print(f'Samples per intersection {i}:', len(self._sample[i]))


        # EvaluateSample()
        making_samples_end_time = time.time()
        making_samples_total_time = making_samples_end_time - making_samples_start_time

    def load_data(self, folder, i):
        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception as e:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None


    def load_data_for_system(self, folder):
        '''
        Load data for all intersections in one folder
        :param folder:
        :return: a list of logging data of one intersection for one folder
        '''
        # load settings
        self.logging_data_list_per_gen = []
        print("Load data for system in ", folder)
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        print(f'Log per intersection {i}:', len(logging_data))
        return 1

    def construct_state(self,features,time,i):
        '''

        :param features:
        :param time:
        :param i:  intersection id
        :return:
        '''

        state = self.logging_data_list_per_gen[i][time]
        traffic_light_phases = self.dic_traffic_env_conf['traffic_light_phases'][i]
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = traffic_light_phases[value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        # print(state_after_selection)
        return state_after_selection


    def _construct_state_process(self, features, time, state, i):
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection, i


    def get_reward_from_features(self, rs):
        reward = {}
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(rs["lane_sum_duration_vehicle_left"])
        reward["sum_num_vehicle_been_stopped_thres01"] = np.sum(rs["lane_num_vehicle_been_stopped_thres01"])
        reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_vehicle_been_stopped_thres1"])
        ##TODO pressure
        reward['pressure'] = np.sum(rs["pressure"])
        reward['sum_delays'] = np.sum(rs["delay"])
        return reward



    def cal_reward(self, rs, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r


    def construct_reward(self, rewards_components, time, i):
        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            #print("t is ", t)
            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self,time,i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']


    def make_reward(self, folder, i):
        '''
        make reward for one folder and one intersection,
        add the samples of one intersection into the list.sample[i]
        samples are built from logs. Logs are cumulative
        :param i: intersection id
        :return:
        '''
        if len(self._sample[i]) == 0:
            last_time = 0
        else:
            last_time = len(self._sample[i])  * self.interval

        if i % 100 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))
        list_samples = []
        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            # construct samples
            for time in range(last_time, total_time - self.measure_time + 1, self.interval):
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant, reward_average = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                action = self.judge_action(time, i)

                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)
                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)


            # list_samples = self.evaluate_sample(list_samples)
            self._sample[i].extend(list_samples)
            return 1
        except Exception as e:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"),"ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)),"ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)),'wb') as f:
                pickle.dump(samples, f, -1)

