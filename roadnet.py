""" Python representation for roadnet urban network """
import json

class RoadNet:

    def __init__(self, roadnet_file):
        self.roadnet_dict = json.load(open(roadnet_file,"r"))
        self.net_edge_dict = {}
        self.net_node_dict = {}
        self.net_lane_dict = {}

        self.generate_node_dict()
        self.generate_edge_dict()
        self.generate_lane_dict()
        self.generate_light_phases_dict()

    def generate_node_dict(self):
        '''
        node dict has key as node id, value could be the dict of input nodes and output nodes
        :return:
        '''

        for node_dict in self.roadnet_dict['intersections']:
            node_id = node_dict['id']
            road_links = node_dict['roads']
            input_nodes = []
            output_nodes = []
            input_edges = []
            output_edges = {}
            for road_link_id in road_links:
                road_link_dict = self._get_road_dict(road_link_id)
                if road_link_dict['startIntersection'] == node_id:
                    end_node = road_link_dict['endIntersection']
                    output_nodes.append(end_node)
                    # todo add output edges
                elif road_link_dict['endIntersection'] == node_id:
                    input_edges.append(road_link_id)
                    start_node = road_link_dict['startIntersection']
                    input_nodes.append(start_node)
                    output_edges[road_link_id] = set()
                    pass

            # update roadlinks
            actual_roadlinks = node_dict['roadLinks']
            for actual_roadlink in actual_roadlinks:
                output_edges[actual_roadlink['startRoad']].add(actual_roadlink['endRoad'])

            net_node = {
                'node_id': node_id,
                'input_nodes': list(set(input_nodes)),
                'input_edges': list(set(input_edges)),
                'output_nodes': list(set(output_nodes)),
                'output_edges': output_edges# should be a dict, with key as an input edge, value as output edges
            }
            if node_id not in self.net_node_dict.keys():
                self.net_node_dict[node_id] = net_node

    def _get_road_dict(self, road_id):
        for item in self.roadnet_dict['roads']:
            if item['id'] == road_id:
                return item
        print("Cannot find the road id {0}".format(road_id))
        sys.exit(-1)
        # return None

    def generate_edge_dict(self):
        '''
        edge dict has key as edge id, value could be the dict of input edges and output edges
        :return:
        '''
        for edge_dict in self.roadnet_dict['roads']:
            edge_id = edge_dict['id']
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']

            net_edge = {
                'edge_id': edge_id,
                'input_node': input_node,
                'output_node': output_node,
                'input_edges': self.net_node_dict[input_node]['input_edges'],
                'output_edges': self.net_node_dict[output_node]['output_edges'][edge_id],

            }
            if edge_id not in self.net_edge_dict.keys():
                self.net_edge_dict[edge_id] = net_edge

    def generate_lane_dict(self):
        lane_dict = {}
        for node_dict in self.roadnet_dict['intersections']:
            for road_link in node_dict["roadLinks"]:
                lane_links = road_link["laneLinks"]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in lane_links:
                    start_lane = start_road + "_" + str(lane_link['startLaneIndex'])
                    end_lane = end_road + "_" +str(lane_link["endLaneIndex"])
                    if start_lane not in lane_dict:
                        lane_dict[start_lane] = {
                            "output_lanes": [end_lane],
                            "input_lanes": []
                        }
                    else:
                        lane_dict[start_lane]["output_lanes"].append(end_lane)
                    if end_lane not in lane_dict:
                        lane_dict[end_lane] = {
                            "output_lanes": [],
                            "input_lanes": [start_lane]
                        }
                    else:
                        lane_dict[end_lane]["input_lanes"].append(start_lane)

        self.net_lane_dict = lane_dict

    def generate_light_phases_dict(self):
        """ Builds phase plans

            Phase plan 2:   1. go_straight 0-2 + turn_right 1-3
                            2. go_straight 1-3 + turn_right 0-2

            Phase plan 4:   1. go_straight 0-2 + turn_right 1-3
                            2. turn_left 0-2
                            3. go_straight 1-3 + turn_right 0-2
                            4. turn_left 1-3
        """
        self.light_phases_dict = {}
        for node_dict in self.roadnet_dict['intersections']:
            if not node_dict['virtual']:
                light_phase_id = node_dict['id']
                # for road_link in node_dict["roadLinks"]:
                roadlinks = node_dict["roadLinks"]
                self.light_phases_dict[light_phase_id] = {
                'phase_plan_2':
                {
                     1: self._generate_light_phase_plan_2(roadlinks, 1),
                     2: self._generate_light_phase_plan_2(roadlinks, 2)
                },
                'phase_plan_4': {
                     1: self._generate_light_phase_plan_4(roadlinks, 1),
                     2: self._generate_light_phase_plan_4(roadlinks, 2),
                     3: self._generate_light_phase_plan_4(roadlinks, 3),
                     4: self._generate_light_phase_plan_4(roadlinks, 4)
                    }
                }


    def _generate_light_phase_plan_2(self, roadlinks, index):
        if index == 1:
            return [
                int((roadlink['type'] in ('go_straight',) and roadlink['direction'] in (0,2)) or \
                    (roadlink['type'] in ('turn_right',) and roadlink['direction'] in (1,3)))
                for roadlink in roadlinks
            ]

        if index == 2:
            return [
                int((roadlink['type'] in ('go_straight',) and roadlink['direction'] in (1,3)) or \
                    (roadlink['type'] in ('turn_right',) and roadlink['direction'] in (0,2)))
                for roadlink in roadlinks
            ]

        raise ValueError

    def _generate_light_phase_plan_4(self, roadlinks, index):
        if index == 1:
            return self._generate_light_phase_plan_2(roadlinks, 1)
        if index == 2:
            return [
                int(roadlink['type'] in ('turn_left',) and roadlink['direction'] in (0,2))
                for roadlink in roadlinks
            ]
        if index == 3:
            return self._generate_light_phase_plan_2(roadlinks, 2)

        if index == 4:
            return [
                int(roadlink['type'] in ('turn_left',) and roadlink['direction'] in (1,3))
                for roadlink in roadlinks
            ]
        raise ValueError

    def hasEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return True
        else:
            return False

    def getEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return edge_id
        else:
            return None

    def getOutgoing(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return self.net_edge_dict[edge_id]['output_edges']
        else:
            return []
