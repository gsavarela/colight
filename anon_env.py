'''
Interactions with CityFlow, get/set values from CityFlow, pass it to RL agents
'''

import sys
import os, json
import pickle
import time
import threading
from collections import defaultdict
from multiprocessing import Process, Pool

import pandas as pd
import numpy as np
import cityflow as engine

from roadnet import RoadNet
from intersection import Intersection
from script import get_traffic_volume
from copy import deepcopy

def g_lst(): return []

class AnonEnv:
    list_intersection_id = [
        "intersection_1_1"
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.info_dict = defaultdict(g_lst)
        self.feature_name_for_neighbor = self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        print("# self.eng.reset() to be implemented")

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
            "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
            "roadnetLogFile": "frontend/web/roadnetLogFile.json",
            "replayLogFile": "frontend/web/replayLogFile.txt"
        }
        print("=========================")
        print(cityflow_config)

        with open(os.path.join(self.path_to_work_directory,"cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)
        self.eng = engine.Engine(os.path.join(self.path_to_work_directory,"cityflow.config"), thread_num=1)
        # self.load_roadnet()
        # self.load_flow()

        # get adjacency
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            self.traffic_light_node_dict = self._adjacency_extraction_lane()
        else:
            self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],self.path_to_log)
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]

        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]

        # set index for intersections and global index for lanes
        self.id_to_index = {}
        count_inter = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i+1, j+1)] = count_inter
                count_inter += 1

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)): # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        # build adjacency_matrix_lane in index from _adjacency_matrix_lane
        for inter in self.list_intersection:
            inter.build_adjacency_row_lane(self.lane_id_to_index)


        # get new measurements
        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": None,
                              "get_vehicle_distance": None
                              }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }
        print("Get system state time: ", time.time()-system_state_start_time)

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)
        print("Update_current_measurements_map time: ", time.time()-update_start_time)

        #update neighbor's info
        neighbor_start_time = time.time()
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters,deepcopy(inter.dic_feature))
        print("Update_neighbor time: ", time.time()-neighbor_start_time)

        state, done = self.get_state()
        # print(state)
        return state

    def step(self, action):
        step_start_time = time.time()
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        # 1. This loops builds the control actions.
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())
        # list_action_in_sec_display: control actions
        # but what does fill_value=-1  stands for?

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            if self.dic_traffic_env_conf['DEBUG']:
                print("time: {0}".format(instant_time))
            else:
                if i == 0:
                    print("time: {0}".format(instant_time))

            self._inner_step(action_in_sec)


            # get reward
            if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()

            reward = self.get_reward()

            
            print(self.current_time)
            if self.dic_traffic_env_conf['DEBUG']:
                print("Reward time: {}".format(time.time()-start_time))


            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)

            # average_reward_action = (average_reward_action*i + reward[0])/(i+1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state, done = self.get_state()
            # Collect info dict
            self._get_info_dict(reward, action, next_state)

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": None,
                              "get_vehicle_distance": None
                              }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        # print("Get system state time: ", time.time()-system_state_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Get system state time: {}".format(time.time()-start_time))
        # get new measurements

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)

        #update neighbor's info
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))


        # print("Update_current_measurements_map time: ", time.time()-update_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Update measurements time: {}".format(time.time()-start_time))

        #self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        #self.log_phase()

    def _get_info_dict(self, rewards, actions, states):
        """Gets information dict."""
        self.info_dict['rewards'].append({
            k: v for k, v in zip(self.id_to_index.keys(), rewards)
        })
        self.info_dict['vehicles'].append(sum(
            len(v) for v in self.system_states['get_lane_vehicles'].values()
        ))
        self.info_dict['velocities'].append(sum(
            v for v in self.system_states['get_vehicle_speed'].values()
        ))
        if self.info_dict['vehicles'][-1] > 0:
            self.info_dict['velocities'][-1] /=  self.info_dict['vehicles'][-1]
        # self.info_dict['states'].append(
        #     self._to_dict(states)
        # )

    # def _format_states(self, feature_list):
    #     return 

    def load_roadnet(self, roadnetFile=None):
        print("Start load roadnet")
        start_time = time.time()
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        #print("/n/n", os.path.join(self.path_to_work_directory, roadnetFile))
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        print("successfully load roadnet:{0}, time: {1}".format(roadnetFile,time.time()-start_time))

    def load_flow(self, flowFile=None):
        print("Start load flowFile")
        start_time = time.time()
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        print("successfully load flowFile: {0}, time: {1}".format(flowFile, time.time()-start_time))

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = self._check_episode_done(list_state)

        # print(list_state)

        return list_state, done

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0","1","2","3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]

        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            if int(inter_ind)%100 == 0:
                print("Batch log for inter ",inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle,orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start,stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()

        print("end join")

    def bulk_log(self):

        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

        self.eng.print_log(os.path.join(self.path_to_log, self.dic_traffic_env_conf["ROADNET_FILE"]),
                           os.path.join(self.path_to_log, "replay_1_1.txt"))

        #print("log files:", os.path.join("data", "frontend", "roadnet_1_1_test.json"))

    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()

    def log_hidden_state(self, hidden_states):
        path_to_log_file = os.path.join(self.path_to_log, "hidden_states.pkl")

        with open(path_to_log_file, "wb") as f:
            pickle.dump(hidden_states, f)

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt"%dic_lane_map[lane]), "a"))

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt"%dic_lane_map[lane]), "a"))

    def log_first_vehicle(self):
        _veh_id = "flow_0_"
        _veh_id_2 = "flow_2_"
        _veh_id_3 = "flow_4_"
        _veh_id_4 = "flow_6_"

        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt" % i), "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt" % i), "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                       'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,}


            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1


            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])


            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

        return traffic_light_node_dict

    def _adjacency_extraction_lane(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])

        roadnet = RoadNet('{0}'.format(file))
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                       'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,
                                                            "total_lane_num": None, 'adjacency_matrix_lane': None,
                                                            "lane_id_to_index": None,
                                                            "lane_ids_in_intersction": [],
                                                            # FIXME: 'incomplete' definition
                                                            'roadlinks_incoming': [],
                                                            'roadlinks_outgoing': [],
                                                            'roadlinks_num_lanes': {},
                                                            'light_phases': {}
                                                            }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            top_k_lane = self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]
            total_inter_num = len(traffic_light_node_dict.keys())

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            # set inter id to index dict
            inter_id_to_index = {}
            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            # set the neighbor_ENWS nodes and entring_lane_ENWS for intersections
            # iterates across intersections.
            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    # if road_id in edge_id_dict:
                    # FIXME: 'incomplete' roadnets 
                    neighboring_node = edge_id_dict[road_id]['to'] if road_id in edge_id_dict else None
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])

                # FIXME: 'incomplete' roadnets 
                traffic_light_node_dict[i]['roadlinks_incoming'] = \
                    [roadlink for roadlink, roaddata in edge_id_dict.items() if roaddata['to'] == i]
                traffic_light_node_dict[i]['roadlinks_outgoing'] = \
                    [roadlink for roadlink, roaddata in edge_id_dict.items() if roaddata['from'] == i]
                roadlinks = traffic_light_node_dict[i]['roadlinks_incoming'] + traffic_light_node_dict[i]['roadlinks_outgoing']
                traffic_light_node_dict[i]['roadlinks_num_lanes'] = \
                    {roadlink: edge_id_dict[roadlink]['num_of_lane'] for roadlink in roadlinks}



            lane_id_dict = roadnet.net_lane_dict
            total_lane_num = len(lane_id_dict.keys())
            # output an adjacentcy matrix for all the intersections
            # each row stands for a lane id,
            # each column is a list with two elements: first is the lane's entering_lane_LSR, second is the lane's leaving_lane_LSR
            def _get_top_k_lane(lane_id_list, top_k_input):
                top_k_lane_indexes = []
                for i in range(top_k_input):
                    lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                    top_k_lane_indexes.append(lane_id)
                return top_k_lane_indexes

            adjacency_matrix_lane = {}
            for i in lane_id_dict.keys(): # Todo lane_ids should be in an order
                adjacency_matrix_lane[i] = [_get_top_k_lane(lane_id_dict[i]['input_lanes'], top_k_lane),
                                            _get_top_k_lane(lane_id_dict[i]['output_lanes'], top_k_lane)]



            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num
                traffic_light_node_dict[i]['total_lane_num'] = total_lane_num
                traffic_light_node_dict[i]['adjacency_matrix_lane'] = adjacency_matrix_lane
                traffic_light_node_dict[i]['light_phases'].update(roadnet.light_phases_dict[i])



        return traffic_light_node_dict



    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))

    def end_sumo(self):
        print("anon process end")
        pass

if __name__ == '__main__':
    dic_agent_conf = {
    "PRIORITY": True,
    "nan_code":True,
    "att_regularization":False,
    "rularization_rate":0.03,
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    #special care for pretrain
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,

    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
        }

    dic_exp_conf = {
        "RUN_COUNTS": 3600,
        "MODEL_NAME": "STGAT",


        "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),

        "NUM_ROUNDS": 200,
        "NUM_GENERATORS": 4,

        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 3,

        "PRETRAIN_NUM_ROUNDS": 0,
        "PRETRAIN_NUM_GENERATORS": 15,

        "AGGREGATE": False,
        "PRETRAIN": False,
        "DEBUG": False,
        "EARLY_STOP": True
    }

    dic_traffic_env_conf  = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "/mnt/RLSignal_general/records/test/anon_3_3_test/anon_3_3_700_1.0.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 9,
        "ACTION_PATTERN": "switch",
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 10,
        "YELLOW_TIME": 5,
        "DEBUG": False,
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "TOP_K_ADJACENCY":9,
        "TOP_K_ADJACENCY_LANE": 6,




            "SAVEREPLAY": False,
            "NUM_ROW": 4,
            "NUM_COL": 3,

            "VOLUME": 300,
            "ROADNET_FILE": "roadnet_{0}.json".format("3_4"),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure"

                # "adjacency_matrix",
                # "lane_queue_length",
                "adjacency_matrix_lane",
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (12,),
                    D_LEAVING_VEHICLE = (12,),

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),
                    D_NEXT_PHASE=(1,),
                    D_TIME_THIS_PHASE=(1,),
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),
                    D_VEHICLE_POSITION_IMG=(4, 60,),
                    D_VEHICLE_SPEED_IMG=(4, 60,),
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                    D_PRESSURE=(1,),

                    D_ADJACENCY_MATRIX=(2,),
                    D_ADJACENCY_MATRIX_LANE=(6,)

                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },
                "anon": {
                    # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                    1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                    2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                    3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                    4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                    # 'WSWL',
                    # 'ESEL',
                    # 'WSES',
                    # 'NSSS',
                    # 'NSNL',
                    # 'SSSL',
                },
            }
        }

    dic_path= {
            "PATH_TO_MODEL": "/Users/Wingslet/PycharmProjects/RLSignal/model/test/anon_3_3_test",
            "PATH_TO_WORK_DIRECTORY": "records/test/jinan_3_4",

            "PATH_TO_DATA": "data/test/",
            "PATH_TO_ERROR": "error/test/"
        }
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                    "round_" + str(0), "generator_" + str(0))

    env = AnonEnv(path_to_log, dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf)
    env.reset()
    print("finish")






